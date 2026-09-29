const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Quant = @import("quantization.zig").Spec;
const A = mx.Array;

pub const Weight = struct {
    arrays: [3]A,
    format: Quant = .{ .bits = 4, .group_size = 32 },

    fn geometry(w: Weight, rank: usize) !struct { n: i32, k: i32 } {
        const dims = mx.shape(w.arrays[0]);
        const scales = mx.shape(w.arrays[1]);
        const biases = mx.shape(w.arrays[2]);
        if (dims.len != rank or scales.len != rank or biases.len != rank or rank < 2) return error.InvalidTensorShape;
        if (!std.mem.eql(i32, dims[0 .. rank - 2], scales[0 .. rank - 2]) or !std.mem.eql(i32, scales, biases)) return error.InvalidTensorShape;
        if (mx.dtype(w.arrays[0]) != mx.c.MLX_UINT32 or mx.dtype(w.arrays[1]) != mx.bf16 or mx.dtype(w.arrays[2]) != mx.bf16) return error.InvalidTensorDType;
        const shape = try w.format.shape(dims[rank - 2 ..], scales[rank - 2 ..], biases[rank - 2 ..]);
        return .{ .n = shape.n, .k = shape.k };
    }
    fn q4(w: Weight) bool {
        return w.format.bits == 4 and w.format.group_size == 32;
    }

    pub fn widened(w: Weight, s: *mx.Scope, format: Quant) !Weight {
        const dims = try w.geometry(2);
        try format.validate();
        if (format.bits < w.format.bits or format.group_size > w.format.group_size or @mod(w.format.group_size, format.group_size) != 0) return error.UnsupportedQuantization;
        var arrays = w.arrays;
        if (format.bits != w.format.bits) {
            try mx.eval(w.arrays[0]);
            const input = mx.c.mlx_array_data_uint32(w.arrays[0])[0..mx.c.mlx_array_size(w.arrays[0])];
            const words: usize = @intCast(@divExact(dims.k * format.bits, 32));
            const output = try mx.allocator.alloc(u32, @as(usize, @intCast(dims.n)) * words);
            defer mx.allocator.free(output);
            try repack(input, output, @intCast(dims.n), @intCast(dims.k), @intCast(w.format.bits), @intCast(format.bits));
            arrays[0] = try s.data(output.ptr, &.{ dims.n, @intCast(words) }, mx.c.MLX_UINT32);
        }
        if (format.group_size != w.format.group_size) for (1..3) |i| {
            var out = mx.c.mlx_array_new();
            const rc = mx.c.mlx_repeat_axis(&out, arrays[i], @intCast(@divExact(w.format.group_size, format.group_size)), -1, mx.stream);
            arrays[i] = try s.result(rc, out);
        };
        return .{ .arrays = arrays, .format = format };
    }
};

pub fn repack(input: []const u32, output: []u32, rows: usize, columns: usize, bits: u6, wider: u6) !void {
    try (Quant{ .bits = bits }).validate();
    try (Quant{ .bits = wider }).validate();
    if (wider < bits or columns % 32 != 0 or rows == 0 or columns == 0 or input.len != rows * (columns / 32 * bits) or output.len != rows * (columns / 32 * wider)) return error.InvalidTensorShape;
    @memset(output, 0);
    const mask = (@as(u64, 1) << bits) - 1;
    for (0..rows) |row| for (0..columns) |col| {
        const at = col * bits;
        const wi = row * (columns / 32 * bits) + at / 32;
        var code = @as(u64, input[wi]) >> @intCast(at % 32);
        if (at % 32 + bits > 32) code |= @as(u64, input[wi + 1]) << @intCast(32 - at % 32);
        const dest = col * wider;
        const di = row * (columns / 32 * wider) + dest / 32;
        const shifted = (code & mask) << @intCast(dest % 32);
        output[di] |= @truncate(shifted);
        if (dest % 32 + wider > 32) output[di + 1] |= @intCast(shifted >> 32);
    };
}

pub fn stack(s: *mx.Scope, parts: []const Weight) !Weight {
    if (parts.len == 0) return error.InvalidTensorShape;
    var format = parts[0].format;
    const first = try parts[0].geometry(2);
    for (parts) |part| {
        if ((try part.geometry(2)).k != first.k) return error.InvalidTensorShape;
        format.bits = @max(format.bits, part.format.bits);
        format.group_size = @min(format.group_size, part.format.group_size);
    }
    const widened = try mx.allocator.alloc(Weight, parts.len);
    defer mx.allocator.free(widened);
    for (parts, widened) |part, *out| out.* = try part.widened(s, format);
    const arrays = try mx.allocator.alloc(A, parts.len);
    defer mx.allocator.free(arrays);
    var result: Weight = .{ .arrays = undefined, .format = format };
    for (0..3) |i| {
        for (widened, arrays) |part, *array| array.* = part.arrays[i];
        result.arrays[i] = if (parts.len == 1) arrays[0] else try s.cat(arrays, 0);
    }
    return result;
}

pub const Kind = enum { qmv, hc_down, hc_up, expert_gateup, expert_down };

pub fn pleGate(kernels: *mx.Kernels, s: *mx.Scope, kv: A, h: A, scales: [3]A, eps: A, streams: i32) ![2]A {
    if (mx.shape(h).len != 2 or mx.shape(kv).len != 2 or streams < 1 or mx.dtype(h) != mx.bf16 or mx.dtype(kv) != mx.bf16) return error.InvalidTensorShape;
    const rows = mx.dim(h, 0);
    const wide = mx.dim(h, 1);
    if (rows < 1 or wide < 1 or @mod(wide, streams * 256) != 0 or mx.dim(kv, 0) != rows or mx.dim(kv, 1) != wide + @divExact(wide, streams)) return error.InvalidTensorShape;
    for (scales) |scale| if (mx.c.mlx_array_size(scale) != wide or mx.dtype(scale) != mx.f32t) return error.InvalidTensorShape;
    if (mx.c.mlx_array_size(eps) != 1 or mx.dtype(eps) != mx.f32t) return error.InvalidTensorShape;
    const out = try kernels.run(s, src.flash_q4_ple_gate, &.{ kv, h, scales[0], scales[1], scales[2], eps }, &.{ mx.ti("S", streams), mx.ti("D", @divExact(wide, streams)) }, .{ 256 * streams, rows, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, wide } }, .{ .shape = &.{ rows, wide } } });
    return out[0..2].*;
}

pub fn pleConv(kernels: *mx.Kernels, s: *mx.Scope, input: A, weight: A, gated: A, h: A, streams: i32, dilation: i32) !A {
    if (mx.shape(h).len != 2 or mx.shape(weight).len != 2 or mx.shape(input).len != 2 or !std.mem.eql(i32, mx.shape(h), mx.shape(gated)) or streams < 1 or dilation < 1) return error.InvalidTensorShape;
    const rows = mx.dim(h, 0);
    const wide = mx.dim(h, 1);
    const taps = mx.dim(weight, 1);
    if (rows < 1 or wide < 1 or taps < 1 or @mod(wide, streams) != 0 or mx.dim(weight, 0) != wide or mx.dim(input, 1) != wide or mx.dim(input, 0) != rows + dilation * (taps - 1)) return error.InvalidTensorShape;
    if (mx.dtype(h) != mx.bf16 or mx.dtype(gated) != mx.bf16 or mx.dtype(input) != mx.bf16 or mx.dtype(weight) != mx.f32t) return error.InvalidTensorDType;
    return (try kernels.run(s, src.flash_q4_ple_conv, &.{ input, weight, gated, h }, &.{ mx.ti("S", streams), mx.ti("D", @divExact(wide, streams)), mx.ti("TAPS", taps), mx.ti("DIL", dilation) }, .{ wide, rows, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, wide } }}))[0];
}

pub fn select(kind: Kind, rows: i32, generation: u32) src.Spec {
    const variant: usize = if (rows < 2 or generation >= 17) 0 else if (generation == 15 or generation == 16) 1 else 2;
    return switch (kind) {
        .qmv => ([_]src.Spec{ src.flash_q4_qmv_rows, src.flash_q4_qmv_rows_h, src.flash_q4_qmv_rows_x })[variant],
        .hc_down => ([_]src.Spec{ src.flash_q4_hc_down_split, src.flash_q4_hc_down_split_h, src.flash_q4_hc_down_split_x })[variant],
        .hc_up => ([_]src.Spec{ src.flash_q4_hc_up2, src.flash_q4_hc_up2_h, src.flash_q4_hc_up2_x })[variant],
        .expert_gateup => ([_]src.Spec{ src.flash_q4_expert_gateup, src.flash_q4_expert_gateup_h, src.flash_q4_expert_gateup_x })[variant],
        .expert_down => ([_]src.Spec{ src.flash_q4_expert_down_y, src.flash_q4_expert_down_y_h, src.flash_q4_expert_down_y_x })[variant],
    };
}

pub fn project(kernels: *mx.Kernels, s: *mx.Scope, x: A, w: Weight, generation: u32, requested_rps: i32) !A {
    const geometry = try w.geometry(2);
    const dims = mx.shape(x);
    if (dims.len == 0 or dims.len > 8 or mx.dim(x, -1) != geometry.k or mx.dtype(x) != mx.bf16) return error.InvalidTensorShape;
    const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(geometry.k)));
    if (rows < 1 or requested_rps < 1) return error.InvalidLaneWidth;
    var rps = requested_rps;
    if (w.q4()) {
        if (@mod(geometry.k, 512) != 0 or @mod(geometry.n, rps) != 0) return error.InvalidTensorShape;
    } else {
        if (@mod(geometry.k, 16) != 0) return error.InvalidTensorShape;
        if (@mod(geometry.n, rps) != 0) rps = if (@mod(geometry.n, 2) == 0) 2 else 1;
    }
    const params = [_]mx.Template{ mx.ti("K", geometry.k), mx.ti("N", geometry.n), mx.ti("RPS", rps), mx.ti("BITS", w.format.bits), mx.ti("GS", w.format.group_size) };
    const spec = if (w.q4()) select(.qmv, rows, generation) else src.flash_qa_qmv_rows;
    const flat = try s.reshape(x, &.{ rows, geometry.k });
    var parts: std.ArrayList(A) = .empty;
    defer parts.deinit(mx.allocator);
    var at: i32 = 0;
    while (at < rows) : (at += 32) {
        const count = @min(rows - at, 32);
        const result = (try kernels.run(s, spec, &.{ try s.slice(flat, 0, at, at + count), w.arrays[0], w.arrays[1], w.arrays[2] }, params[0..if (w.q4()) @as(usize, 3) else 5], .{ 32 * count, @divExact(geometry.n, rps), 1 }, .{ 32 * count, 1, 1 }, &.{.{ .shape = &.{ count, geometry.n } }}))[0];
        try parts.append(mx.allocator, result);
    }
    var output: [8]i32 = undefined;
    @memcpy(output[0..dims.len], dims);
    output[dims.len - 1] = geometry.n;
    return s.reshape(if (parts.items.len == 1) parts.items[0] else try s.cat(parts.items, 0), output[0..dims.len]);
}

pub fn hyper(kernels: *mx.Kernels, s: *mx.Scope, h: A, ssp: A, down: Weight, up: Weight, norm: A, eps: A, streams: i32, low: i32, generation: u32) ![2]A {
    const dg = try down.geometry(2);
    const ug = try up.geometry(2);
    if (mx.shape(h).len != 2 or streams <= 0 or low <= 0 or @mod(low, 32) != 0) return error.InvalidTensorShape;
    const rows = mx.dim(h, 0);
    const wide = mx.dim(h, 1);
    if (rows <= 0 or @mod(wide, streams * 256) != 0 or @mod(wide, 1024) != 0 or dg.k != wide or ug.n != wide or ug.k != low or (dg.n != low and dg.n != low + streams)) return error.InvalidTensorShape;
    const dims = @divExact(wide, streams);
    const splits = @divExact(wide, 1024);
    const generic = !down.q4() or !up.q4();
    const dp = [_]mx.Template{ mx.ti("S", streams), mx.ti("D", dims), mx.ti("ND", dg.n), mx.ti("BITS", down.format.bits), mx.ti("GS", down.format.group_size) };
    const count = try s.ints(&.{rows});
    const part = (try kernels.run(s, if (generic) src.flash_qa_hc_down_split else select(.hc_down, rows, generation), &.{ h, ssp, norm, down.arrays[0], down.arrays[1], down.arrays[2], eps, count }, dp[0..if (generic) @as(usize, 5) else 3], .{ @divTrunc(dg.n + 7, 8) * 256, splits, rows }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ splits, rows, dg.n }, .dtype = mx.f32t }}))[0];
    const up_params = [_]mx.Template{ mx.ti("S", streams), mx.ti("D", dims), mx.ti("LOW", low), mx.ti("ND", dg.n), mx.ti("KS", splits), mx.ti("BITS", up.format.bits), mx.ti("GS", up.format.group_size) };
    const threads = streams * 8 * @divExact(low, 32);
    if (threads > 1024) return error.InvalidThreadgroup;
    const out = try kernels.run(s, if (generic) src.flash_qa_hc_up2 else select(.hc_up, rows, generation), &.{ h, ssp, part, up.arrays[0], up.arrays[1], up.arrays[2], norm, eps, count }, up_params[0..if (generic) @as(usize, 7) else 5], .{ @divExact(dims, 8) * threads, rows, 1 }, .{ threads, 1, 1 }, &.{ .{ .shape = &.{ rows, dims } }, .{ .shape = &.{ @max(rows, 2), streams } } });
    return .{ out[0], try s.slice(out[1], 0, 0, rows) };
}

pub fn gateUp(kernels: *mx.Kernels, s: *mx.Scope, x: A, logits: A, gate: Weight, up: Weight, shared: ?[2]Weight, top: i32, generation: u32, rps: i32, groups: i32) ![3]A {
    const g = try gate.geometry(3);
    const u = try up.geometry(3);
    const sg = if (shared) |pair| pair[0] else gate;
    const su = if (shared) |pair| pair[1] else up;
    const extra: i32 = @intFromBool(shared != null);
    const sg_shape = try sg.geometry(if (shared != null) 2 else 3);
    const su_shape = try su.geometry(if (shared != null) 2 else 3);
    if (!std.meta.eql(gate.format, up.format) or !std.meta.eql(sg.format, su.format)) return error.UnsupportedQuantization;
    if (!std.meta.eql(g, u) or !std.meta.eql(g, sg_shape) or !std.meta.eql(g, su_shape) or mx.dim(gate.arrays[0], 0) != mx.dim(up.arrays[0], 0)) return error.InvalidTensorShape;
    if (mx.shape(x).len != 2 or mx.shape(logits).len != 2 or mx.dim(x, 1) != g.k or mx.dim(x, 0) != mx.dim(logits, 0)) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    const experts = mx.dim(gate.arrays[0], 0);
    if (top < 1 or top > experts or @mod(experts, 32) != 0 or mx.dim(logits, 1) != experts + extra or rps < 1 or groups < 1 or @mod(g.k, 512) != 0 or @mod(g.n, 8) != 0 or @mod(g.n, rps * groups) != 0) return error.InvalidTensorShape;
    const generic = !gate.q4() or !sg.q4();
    const params = [_]mx.Template{ mx.ti("K", g.k), mx.ti("N", g.n), mx.ti("TOPK", top), mx.ti("SHARED", extra), mx.ti("NE", experts), mx.ti("NL", mx.dim(logits, 1)), mx.ti("RPS", rps), mx.ti("SG", groups), mx.ti("WB", gate.format.bits), mx.ti("WG", gate.format.group_size), mx.ti("SWB", sg.format.bits), mx.ti("SWG", sg.format.group_size) };
    const out = try kernels.run(s, if (generic) src.flash_qa_expert_gateup else select(.expert_gateup, rows, generation), &.{ x, logits, gate.arrays[0], gate.arrays[1], gate.arrays[2], up.arrays[0], up.arrays[1], up.arrays[2], sg.arrays[0], sg.arrays[1], sg.arrays[2], su.arrays[0], su.arrays[1], su.arrays[2] }, params[0..if (generic) @as(usize, 12) else 8], .{ 32 * groups, @divExact(g.n, rps * groups), rows * (top + extra) }, .{ 32 * groups, 1, 1 }, &.{ .{ .shape = &.{ rows, top + extra, g.n } }, .{ .shape = &.{ rows, top }, .dtype = mx.c.MLX_UINT32 }, .{ .shape = &.{ rows, top }, .dtype = mx.f32t } });
    return out[0..3].*;
}

pub fn expertDown(kernels: *mx.Kernels, s: *mx.Scope, act: A, picks: A, down: Weight, shared: Weight, generation: u32, groups: i32) !A {
    const dg = try down.geometry(3);
    const sg = try shared.geometry(2);
    if (!std.meta.eql(dg, sg) or mx.shape(act).len != 3 or mx.shape(picks).len != 2 or mx.dim(act, 0) != mx.dim(picks, 0) or mx.dim(act, 1) != mx.dim(picks, 1) + 1 or mx.dim(act, 2) != dg.k or @mod(dg.n, 8) != 0 or groups < 1) return error.InvalidTensorShape;
    const rows = mx.dim(act, 0);
    const slots = mx.dim(act, 1);
    const generic = !down.q4() or !shared.q4();
    if (!generic and (dg.k > 1024 or @mod(dg.k, 32) != 0)) return error.InvalidTensorShape;
    const params = [_]mx.Template{ mx.ti("NI", dg.k), mx.ti("D", dg.n), mx.ti("TOPK", mx.dim(picks, 1)), mx.ti("SG", groups), mx.ti("WB", down.format.bits), mx.ti("WG", down.format.group_size), mx.ti("SWB", shared.format.bits), mx.ti("SWG", shared.format.group_size) };
    return (try kernels.run(s, if (generic) src.flash_qa_expert_down_y else select(.expert_down, rows, generation), &.{ act, picks, down.arrays[0], down.arrays[1], down.arrays[2], shared.arrays[0], shared.arrays[1], shared.arrays[2], try s.ints(&.{rows}) }, params[0..if (generic) @as(usize, 8) else 4], .{ 32 * groups, @divExact(dg.n, 8), @divTrunc(rows * slots + groups - 1, groups) }, .{ 32 * groups, 1, 1 }, &.{.{ .shape = &.{ rows, slots, dg.n } }}))[0];
}

test "Flash nibble dispatch follows GPU generation and total row count" {
    inline for (.{ Kind.qmv, Kind.hc_down, Kind.hc_up, Kind.expert_gateup, Kind.expert_down }) |kind| {
        for ([_]u32{ 0, 13, 14, 15, 16, 17, 18 }) |generation| for ([_]i32{ 1, 2, 16, 32, 33, 128 }) |rows| {
            const name = select(kind, rows, generation).name;
            if (rows == 1 or generation >= 17) {
                try std.testing.expect(!std.mem.endsWith(u8, name, "_h") and !std.mem.endsWith(u8, name, "_x"));
            } else try std.testing.expect(std.mem.endsWith(u8, name, if (generation == 15 or generation == 16) "_h" else "_x"));
        };
    }
}

pub fn checkWeights(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    const cp = @import("checkpoint.zig");
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/weights.json", .{dir}));
    defer mx.allocator.free(bytes);
    const cases = try std.json.parseFromSlice([]const []const u8, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    for (cases.value) |name| {
        errdefer std.debug.print("Flash weight case failed: {s}\n", .{name});
        var s = mx.Scope{};
        defer s.deinit();
        var weights = cp.Store.init(32);
        defer weights.deinit();
        weights.flash_drafts = false;
        const config = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/{s}/config.json", .{ dir, name }));
        defer mx.allocator.free(config);
        try weights.configure(config);
        try weights.load(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, name }), "language_model.");
        var expected = cp.Store.init(32);
        defer expected.deinit();
        try expected.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}/expected.safetensors", .{ dir, name }), "", "");
        const combined = try stack(&s, &.{ try weights.affine("model.a"), try weights.affine("model.b") });
        const fmt = try expected.get("format");
        try mx.eval(fmt);
        const format = mx.c.mlx_array_data_int32(fmt)[0..2];
        try std.testing.expectEqual(Quant{ .bits = format[0], .group_size = format[1] }, combined.format);
        inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, i| {
            const want = try expected.get("stack." ++ suffix);
            if (i == 0) {
                try mx.evalMany(&.{ combined.arrays[0], want }, false);
                try std.testing.expectEqualSlices(u32, mx.c.mlx_array_data_uint32(want)[0..mx.c.mlx_array_size(want)], mx.c.mlx_array_data_uint32(combined.arrays[0])[0..mx.c.mlx_array_size(combined.arrays[0])]);
            } else try @import("sampling_checks.zig").equal(&s, combined.arrays[i], want);
        }
        try weights.putAffine("native_stack", combined);
        const x = try expected.get("input");
        const equal = @import("sampling_checks.zig").equal;
        try equal(&s, try weights.dequant(&s, "model.a"), try expected.get("a.dequant"));
        try equal(&s, try weights.embed(&s, "model.a", &.{ 0, 7 }), try expected.get("a.embed"));
        try equal(&s, try weights.linear(&kernels, &s, "model.b", x, true), try expected.get("b.projection"));
        try equal(&s, try weights.linear(&kernels, &s, "model.b", x, false), try expected.get("b.matmul"));
        try equal(&s, try weights.linear(&kernels, &s, "native_stack", x, true), try expected.get("stack.projection"));
        try equal(&s, try weights.linear(&kernels, &s, "model.dense", x, true), try expected.get("dense.projection"));
    }
    std.debug.print("PASS: {d} mixed Flash checkpoint formats, exact packed widening, regrouping, embeddings and projections match upstream\n", .{cases.value.len});
}
