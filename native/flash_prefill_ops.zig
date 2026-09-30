const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Weight = @import("flash_ops.zig").Weight;
const A = mx.Array;

pub const Pending = struct { branch: A, inject: A };
pub const Hyper = struct { residual: A, mixed: A, inject: ?A };

fn geometry(h: A, streams: i32) !struct { rows: i32, wide: i32, dims: i32 } {
    if (mx.shape(h).len != 2 or streams < 1 or streams > 256) return error.InvalidTensorShape;
    const rows = mx.dim(h, 0);
    const wide = mx.dim(h, 1);
    if (rows < 1 or wide < 1 or @mod(wide, streams * 256) != 0) return error.InvalidTensorShape;
    if (mx.dtype(h) != mx.bf16) return error.InvalidTensorDType;
    return .{ .rows = rows, .wide = wide, .dims = @divExact(wide, streams) };
}

pub fn writeBack(kernels: *mx.Kernels, s: *mx.Scope, h: A, pending: ?Pending, streams: i32) ![2]A {
    const g = try geometry(h, streams);
    const params = [_]mx.Template{ mx.ti("S", streams), mx.ti("D", g.dims) };
    const outputs = [_]mx.Output{ .{ .shape = &.{ g.rows, g.wide } }, .{ .shape = &.{ g.rows, @divExact(g.dims, 256), streams }, .dtype = mx.f32t } };
    if (pending) |p| {
        if (!std.mem.eql(i32, mx.shape(p.branch), &.{ g.rows, g.dims }) or !std.mem.eql(i32, mx.shape(p.inject), &.{ g.rows, streams })) return error.InvalidTensorShape;
        if (mx.dtype(p.branch) != mx.bf16 or mx.dtype(p.inject) != mx.bf16) return error.InvalidTensorDType;
        const out = try kernels.run(s, src.q4_hc_norm_plain, &.{ h, p.inject, p.branch }, &params, .{ g.dims, g.rows, 1 }, .{ 256, 1, 1 }, &outputs);
        return out[0..2].*;
    }
    const out = try kernels.run(s, src.q4_hc_norm_none, &.{h}, &params, .{ g.dims, g.rows, 1 }, .{ 256, 1, 1 }, &outputs);
    return out[0..2].*;
}

pub fn normalize(kernels: *mx.Kernels, s: *mx.Scope, h: A, ssp: A, scale: A, eps: A, streams: i32) !A {
    const g = try geometry(h, streams);
    if (!std.mem.eql(i32, mx.shape(ssp), &.{ g.rows, @divExact(g.dims, 256), streams }) or mx.c.mlx_array_size(scale) != g.wide or mx.c.mlx_array_size(eps) != 1) return error.InvalidTensorShape;
    if (mx.dtype(ssp) != mx.f32t or mx.dtype(scale) != mx.f32t or mx.dtype(eps) != mx.f32t) return error.InvalidTensorDType;
    return (try kernels.run(s, src.flash_prefill_hc_normed, &.{ h, ssp, scale, eps }, &.{ mx.ti("S", streams), mx.ti("D", g.dims) }, .{ g.wide, g.rows, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ g.rows, g.wide } }}))[0];
}

pub fn activate(kernels: *mx.Kernels, s: *mx.Scope, down: A, streams: i32, low: i32) !struct { act: A, inject: ?A } {
    if (mx.shape(down).len != 2 or streams < 1 or streams > 1024 or low < 1 or low > std.math.maxInt(i32) - streams) return error.InvalidTensorShape;
    const rows = mx.dim(down, 0);
    const width = mx.dim(down, 1);
    if (rows < 1 or (width != low and width != low + streams)) return error.InvalidTensorShape;
    if (mx.dtype(down) != mx.bf16) return error.InvalidTensorDType;
    const out = try kernels.run(s, src.flash_prefill_hc_act, &.{down}, &.{ mx.ti("S", streams), mx.ti("LOW", low), mx.ti("ND", width) }, .{ width, rows, 1 }, .{ @min(width, 256), 1, 1 }, &.{ .{ .shape = &.{ rows, low } }, .{ .shape = &.{ rows, streams } } });
    // Upstream leaves INJ unwritten for the final mixer.
    return .{ .act = out[0], .inject = if (width > low) out[1] else null };
}

pub fn mix(kernels: *mx.Kernels, s: *mx.Scope, up: A, normed: A, streams: i32) !A {
    const g = try geometry(normed, streams);
    if (!std.mem.eql(i32, mx.shape(up), mx.shape(normed))) return error.InvalidTensorShape;
    if (mx.dtype(up) != mx.bf16) return error.InvalidTensorDType;
    return (try kernels.run(s, src.flash_prefill_hc_mix, &.{ up, normed }, &.{ mx.ti("S", streams), mx.ti("D", g.dims) }, .{ g.dims, g.rows, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ g.rows, g.dims } }}))[0];
}

pub fn matmul(s: *mx.Scope, x: A, weight: Weight) !A {
    const g = try weight.geometry(2);
    if (mx.shape(x).len < 2 or mx.dim(x, -1) != g.k) return error.InvalidTensorShape;
    if (mx.dtype(x) != mx.bf16) return error.InvalidTensorDType;
    var out = mx.c.mlx_array_new();
    const w = weight.arrays;
    const rc = mx.c.mlx_quantized_matmul(&out, x, w[0], w[1], w[2], true, mx.opt(weight.format.group_size), mx.opt(weight.format.bits), "affine", mx.stream);
    return s.result(rc, out);
}

pub fn hyper(kernels: *mx.Kernels, s: *mx.Scope, h: A, pending: ?Pending, down: Weight, up: Weight, scale: A, eps: A, streams: i32, low: i32) !Hyper {
    const g = try geometry(h, streams);
    const dg = try down.geometry(2);
    const ug = try up.geometry(2);
    if (low < 1 or low > std.math.maxInt(i32) - streams or dg.k != g.wide or ug.n != g.wide or ug.k != low or (dg.n != low and dg.n != low + streams)) return error.InvalidTensorShape;
    const written = try writeBack(kernels, s, h, pending, streams);
    const normed = try normalize(kernels, s, written[0], written[1], scale, eps, streams);
    const active = try activate(kernels, s, try matmul(s, normed, down), streams, low);
    const projected = try matmul(s, active.act, up);
    return .{ .residual = written[0], .mixed = try mix(kernels, s, projected, normed, streams), .inject = active.inject };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/hyper.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, streams: i32, low: i32, pending: bool, inject: bool, down_bits: i32, down_group: i32, up_bits: i32, up_group: i32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    for (cases.value) |case| {
        errdefer std.debug.print("Flash prefill hyper-connection failed: {s}\n", .{case.name});
        var store = @import("checkpoint.zig").Store.init(32);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        const down = Weight{ .arrays = .{ try store.get("down.weight"), try store.get("down.scales"), try store.get("down.biases") }, .format = .{ .bits = case.down_bits, .group_size = case.down_group } };
        const up = Weight{ .arrays = .{ try store.get("up.weight"), try store.get("up.scales"), try store.get("up.biases") }, .format = .{ .bits = case.up_bits, .group_size = case.up_group } };
        const h = try store.get("h");
        const pending: ?Pending = if (case.pending) .{ .branch = try store.get("branch"), .inject = try store.get("pending_inject") } else null;
        const scale = try store.get("scale");
        const eps = try store.get("eps");
        const equal = @import("variant_checks.zig").equalBits;
        const written = try writeBack(&kernels, &s, h, pending, case.streams);
        try equal(&s, written[0], try store.get("residual"));
        try equal(&s, written[1], try store.get("ssp"));
        const normed = try normalize(&kernels, &s, written[0], written[1], scale, eps, case.streams);
        try equal(&s, normed, try store.get("normed"));
        const dn = try matmul(&s, normed, down);
        try equal(&s, dn, try store.get("dn"));
        const active = try activate(&kernels, &s, dn, case.streams, case.low);
        try equal(&s, active.act, try store.get("act"));
        try std.testing.expectEqual(case.inject, active.inject != null);
        if (active.inject) |inj| try equal(&s, inj, try store.get("inject"));
        const projected = try matmul(&s, active.act, up);
        try equal(&s, projected, try store.get("projected"));
        try equal(&s, try mix(&kernels, &s, projected, normed, case.streams), try store.get("mixed"));
        const result = try hyper(&kernels, &s, h, pending, down, up, scale, eps, case.streams, case.low);
        try equal(&s, result.residual, try store.get("residual"));
        try equal(&s, result.mixed, try store.get("mixed"));
        try std.testing.expectEqual(case.inject, result.inject != null);
        if (result.inject) |inj| try equal(&s, inj, try store.get("inject"));
        try std.testing.expectError(error.InvalidTensorShape, writeBack(&kernels, &s, h, null, 0));
        try std.testing.expectError(error.InvalidTensorShape, writeBack(&kernels, &s, h, null, 257));
        try std.testing.expectError(error.InvalidTensorDType, writeBack(&kernels, &s, try s.cast(h, mx.f32t), null, case.streams));
        try std.testing.expectError(error.InvalidTensorShape, activate(&kernels, &s, dn, case.streams, mx.dim(dn, 1) + 1));
        try std.testing.expectError(error.InvalidTensorShape, normalize(&kernels, &s, h, written[1], try s.slice(scale, 0, 0, 1), eps, case.streams));
        try std.testing.expectError(error.InvalidTensorShape, hyper(&kernels, &s, h, pending, up, down, scale, eps, case.streams, case.low));
    }
    std.debug.print("PASS: {d} Flash prefill hyper-connections, exact intermediate bits, residual write-back, mixed affine formats and input validation\n", .{cases.value.len});
}
