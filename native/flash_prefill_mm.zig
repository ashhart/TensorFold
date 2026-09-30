const std = @import("std");
const mx = @import("mlx.zig");
const c = mx.c;
const A = mx.Array;
const Weight = @import("flash_ops.zig").Weight;
const src = @import("kernel_sources.zig");

pub var require_kernels = false;

pub const State = struct {
    decision: ?bool = null,

    pub fn tiles(state: *State, kernels: *mx.Kernels) !bool {
        if (state.decision) |value| return value;
        if (mx.tensor_units) {
            state.decision = false;
            return false;
        }
        const ok = selfCheck(kernels) catch |err| {
            std.debug.print("Flash prompt kernels failed to build or execute: {s}; using pinned MLX matmuls.\n", .{@errorName(err)});
            if (require_kernels) return error.PrefillKernelSelfCheckFailed;
            state.decision = false;
            return false;
        };
        if (!ok) {
            std.debug.print("Flash prompt kernels differ from pinned MLX; using MLX matmuls.\n", .{});
            if (require_kernels) return error.PrefillKernelSelfCheckFailed;
        }
        state.decision = ok;
        return ok;
    }
};

pub fn splitsK(rows: i32, columns: i32, width: i32) bool {
    var parts: i32 = @max(1, @divTrunc(512, @divTrunc(columns + 31, 32) * @divTrunc(rows + 31, 32)));
    parts = @min(parts, @divTrunc(width, 32));
    while (parts > 1 and @mod(width, parts * 32) != 0) parts -= 1;
    return parts > 1;
}

pub fn qmm(kernels: *mx.Kernels, s: *mx.Scope, x: A, w: Weight) !A {
    const g = try w.geometry(2);
    if (mx.shape(x).len != 2 or mx.dim(x, 1) != g.k or mx.dtype(x) != mx.bf16 or w.format.bits != 4 or w.format.group_size != 32) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    if (rows < 1) return error.InvalidTensorShape;
    const bn: i32 = if (g.n >= 8192) 64 else 32;
    return (try kernels.run(s, src.flash_prefill_qmm, &.{ x, w.arrays[0], w.arrays[1], w.arrays[2], try s.ints(&.{g.k}), try s.ints(&.{g.n}), try s.ints(&.{rows}) }, &.{ mx.ti("BM", 64), mx.ti("BN", bn), mx.ti("BK", 32), mx.ti("ALIGNED", @intFromBool(@mod(g.n, bn) == 0)) }, .{ @divTrunc(g.n + bn - 1, bn) * 128, @divTrunc(rows + 63, 64), 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ rows, g.n } }}))[0];
}

pub fn linear(kernels: *mx.Kernels, s: *mx.Scope, x: A, w: Weight) !A {
    if ((try w.geometry(2)).n < 32) return @import("flash_prefill_ops.zig").matmul(s, x, w);
    return matmul(kernels, s, x, w);
}

pub fn matmul(kernels: *mx.Kernels, s: *mx.Scope, x: A, w: Weight) !A {
    const g = try w.geometry(2);
    if (mx.shape(x).len < 2 or mx.dim(x, -1) != g.k or mx.dtype(x) != mx.bf16) return error.InvalidTensorShape;
    const rows: i32 = @intCast(@divExact(c.mlx_array_size(x), @as(usize, @intCast(g.k))));
    if (w.format.bits == 4 and w.format.group_size == 32 and rows >= 64 and !splitsK(rows, g.n, g.k) and try kernels.flash_prefill.tiles(kernels)) {
        const out = try qmm(kernels, s, try s.reshape(x, &.{ rows, g.k }), w);
        var shape: [8]i32 = undefined;
        const dims = mx.shape(x);
        if (dims.len > shape.len) return error.InvalidTensorShape;
        @memcpy(shape[0..dims.len], dims);
        shape[dims.len - 1] = g.n;
        return s.reshape(out, shape[0..dims.len]);
    }
    return @import("flash_prefill_ops.zig").matmul(s, x, w);
}

pub fn gatherSorted(kernels: *mx.Kernels, s: *mx.Scope, x: A, w: Weight, ids: A, tile: ?[4]i32) !A {
    const g = try w.geometry(3);
    if (mx.shape(x).len != 2 or mx.dim(x, 1) != g.k or mx.dtype(x) != mx.bf16 or w.format.bits != 4 or @mod(w.format.group_size, 32) != 0) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    const experts = mx.dim(w.arrays[0], 0);
    if (rows < 1 or experts < 1 or !std.mem.eql(i32, mx.shape(ids), &.{rows}) or mx.dtype(ids) != c.MLX_UINT32) return error.InvalidTensorShape;
    const shape = tile orelse if (@as(i64, rows) < @as(i64, 56) * experts) [4]i32{ 16, 32, 1, 2 } else [4]i32{ 32, 32, 1, 2 };
    const bm = shape[0];
    const bn = shape[1];
    const wm = shape[2];
    const wn = shape[3];
    if (!std.mem.eql(i32, &shape, &.{ 16, 32, 1, 2 }) and !std.mem.eql(i32, &shape, &.{ 32, 32, 1, 2 }) and !std.mem.eql(i32, &shape, &.{ 64, 64, 2, 2 })) return error.InvalidTileShape;
    const count = try s.ints(&.{experts});
    const m = try s.ints(&.{rows});
    const offsets = (try kernels.run(s, src.flash_prefill_offsets, &.{ ids, m, count }, &.{}, .{ experts + 1, 1, 1 }, .{ @min(256, experts + 1), 1, 1 }, &.{.{ .shape = &.{experts + 1}, .dtype = mx.i32t }}))[0];
    const most = @min(rows, @divTrunc(rows + bm - 1, bm) + experts);
    return (try kernels.run(s, src.flash_prefill_gather, &.{ x, w.arrays[0], w.arrays[1], w.arrays[2], offsets, m, try s.ints(&.{g.n}), try s.ints(&.{g.k}), count }, &.{ mx.ti("GS", w.format.group_size), mx.ti("BM", bm), mx.ti("BN", bn), mx.ti("BK", 32), mx.ti("WM", wm), mx.ti("WN", wn) }, .{ @divTrunc(g.n + bn - 1, bn) * 32, most * wn, wm }, .{ 32, wn, wm }, &.{.{ .shape = &.{ rows, g.n } }}))[0];
}

fn split(s: *mx.Scope, key: A, count: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_random_split_num(&out, key, count, mx.stream);
    return s.result(rc, out);
}
fn keyAt(s: *mx.Scope, keys: A, index: i32) !A {
    return s.reshape(try s.slice(keys, 0, index, index + 1), &.{2});
}
fn normal(s: *mx.Scope, dims: []const i32, key: A) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_random_normal(&out, dims.ptr, dims.len, mx.f32t, 0, 1, key, mx.stream);
    return s.result(rc, out);
}
fn randint(s: *mx.Scope, dims: []const i32, key: A, high: i64, dtype: c.mlx_dtype) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_random_randint(&out, try s.ints(&.{0}), try s.data(&high, &.{}, c.MLX_INT64), dims.ptr, dims.len, dtype, key, mx.stream);
    return s.result(rc, out);
}
fn weights(s: *mx.Scope, key: A, experts: ?i32, n: i32, k: i32) !Weight {
    const wd = [_]i32{ experts orelse 1, n, @divExact(k, 8) };
    const sd = [_]i32{ experts orelse 1, n, @divExact(k, 32) };
    const start: usize = if (experts != null) 0 else 1;
    const keys = try split(s, key, 2);
    const scale = try s.scalar(0.02);
    return .{ .arrays = .{ try randint(s, wd[start..], key, 2147483648, c.MLX_UINT32), try s.cast(try s.binary(c.mlx_multiply, try normal(s, sd[start..], try keyAt(s, keys, 0)), scale), mx.bf16), try s.cast(try s.binary(c.mlx_multiply, try normal(s, sd[start..], try keyAt(s, keys, 1)), scale), mx.bf16) }, .format = .{ .bits = 4, .group_size = 32 } };
}
fn equal(s: *mx.Scope, a: A, b: A) !bool {
    var out = c.mlx_array_new();
    const rc = c.mlx_array_equal(&out, a, b, false, mx.stream);
    out = try s.result(rc, out);
    try mx.eval(out);
    var result = false;
    try mx.check(c.mlx_array_item_bool(&result, out));
    return result;
}

pub fn selfCheck(kernels: *mx.Kernels) !bool {
    return selfCheckAgainst(kernels, null);
}

fn probeInputs(s: *mx.Scope, reference: ?*@import("checkpoint.zig").Store, index: usize, x: A, w: Weight, ids: ?A) !void {
    const ref = reference orelse return;
    var path: [64]u8 = undefined;
    for ([_][]const u8{ "input", "weight", "scales", "biases" }, [_]A{ x, w.arrays[0], w.arrays[1], w.arrays[2] }) |name, array| {
        try @import("variant_checks.zig").equalBits(s, array, try ref.get(try std.fmt.bufPrint(&path, "probe{d}.{s}", .{ index, name })));
    }
    if (ids) |array| try @import("variant_checks.zig").equalBits(s, array, try ref.get("probe2.ids"));
}

fn selfCheckAgainst(kernels: *mx.Kernels, reference: ?*@import("checkpoint.zig").Store) !bool {
    var s = mx.Scope{};
    defer s.deinit();
    var root = c.mlx_array_new();
    const rc = c.mlx_random_key(&root, 20260926);
    const keys = try split(&s, try s.result(rc, root), 4);
    const input_key = try keyAt(&s, keys, 3);
    var same = true;
    for ([_][2]i32{ .{ 512, 640 }, .{ 128, 8192 } }, 0..) |dims, i| {
        const x = try s.cast(try normal(&s, &.{ dims[0], 256 }, input_key), mx.bf16);
        const w = try weights(&s, try keyAt(&s, keys, @intCast(i)), null, dims[1], 256);
        try probeInputs(&s, reference, i, x, w, null);
        const matches = try equal(&s, try qmm(kernels, &s, x, w), try @import("flash_prefill_ops.zig").matmul(&s, x, w));
        same = same and matches;
    }
    const x = try s.cast(try normal(&s, &.{ 400, 256 }, input_key), mx.bf16);
    const key = try keyAt(&s, keys, 2);
    const w = try weights(&s, key, 16, 64, 256);
    const ids = try s.cast(try s.unary(c.mlx_sort, try randint(&s, &.{400}, key, 16, mx.i32t)), c.MLX_UINT32);
    try probeInputs(&s, reference, 2, x, w, ids);
    var ref = c.mlx_array_new();
    const rr = c.mlx_gather_qmm(&ref, try s.reshape(x, &.{ 400, 1, 256 }), w.arrays[0], w.arrays[1], w.arrays[2], mx.empty, ids, true, mx.opt(32), mx.opt(4), "affine", true, mx.stream);
    ref = try s.reshape(try s.result(rr, ref), &.{ 400, 64 });
    for ([_][4]i32{ .{ 16, 32, 1, 2 }, .{ 32, 32, 1, 2 } }) |tile| {
        const matches = try equal(&s, try gatherSorted(kernels, &s, x, w, ids, tile), ref);
        same = same and matches;
    }
    return same;
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/mm.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, mode: enum { qmm, matmul, linear, gather }, bits: i32, group: i32, tile: ?[4]i32, decision: bool };
    const Fixture = struct { self_check: bool, cases: []const Case };
    const fixture = try std.json.parseFromSlice(Fixture, mx.allocator, bytes, .{});
    defer fixture.deinit();
    if (fixture.value.cases.len == 0) return error.EmptyFixtures;
    var probes = @import("checkpoint.zig").Store.init(32);
    defer probes.deinit();
    try probes.loadFile(io, try std.fmt.bufPrint(&path, "{s}/self-check.safetensors", .{dir}), "", "");
    try std.testing.expectEqual(fixture.value.self_check, try selfCheckAgainst(&kernels, &probes));
    const before = mx.tensor_units;
    defer mx.tensor_units = before;
    mx.tensor_units = false;
    try std.testing.expectEqual(fixture.value.self_check, try kernels.flash_prefill.tiles(&kernels));
    try std.testing.expectEqual(fixture.value.self_check, try kernels.flash_prefill.tiles(&kernels));
    if (!fixture.value.self_check) {
        const strict = require_kernels;
        defer require_kernels = strict;
        require_kernels = true;
        var state = State{};
        try std.testing.expectError(error.PrefillKernelSelfCheckFailed, state.tiles(&kernels));
        try std.testing.expectEqual(@as(?bool, null), state.decision);
    }
    mx.tensor_units = true;
    var tensor = State{};
    try std.testing.expectEqual(false, try tensor.tiles(&kernels));
    mx.tensor_units = before;
    for (fixture.value.cases) |case| {
        errdefer std.debug.print("Flash prefill matmul fixture failed: {s}\n", .{case.name});
        var store = @import("checkpoint.zig").Store.init(32);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        const w = Weight{ .arrays = .{ try store.get("weight"), try store.get("scales"), try store.get("biases") }, .format = .{ .bits = case.bits, .group_size = case.group } };
        const x = try store.get("input");
        kernels.flash_prefill.decision = case.decision;
        const out = switch (case.mode) {
            .qmm => try qmm(&kernels, &s, x, w),
            .linear => try linear(&kernels, &s, x, w),
            .matmul => try matmul(&kernels, &s, x, w),
            .gather => try gatherSorted(&kernels, &s, x, w, try store.get("ids"), case.tile),
        };
        try @import("variant_checks.zig").equalBits(&s, out, try store.get("output"));
    }
    std.debug.print("PASS: {d} Flash tiled matmul cases, split-K fallback, expert tile scans and upstream seeded self-check ({any})\n", .{ fixture.value.cases.len, fixture.value.self_check });
}
