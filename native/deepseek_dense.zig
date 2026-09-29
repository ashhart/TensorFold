const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;

pub const Projection = struct {
    weights: [3]A,
    group: i32 = 64,

    pub fn validate(p: Projection) !void {
        if (p.group != 32 and p.group != 64) return error.UnsupportedQuantization;
        const g = try (@import("quantization.zig").Spec{ .bits = 4, .group_size = p.group }).shape(mx.shape(p.weights[0]), mx.shape(p.weights[1]), mx.shape(p.weights[2]));
        if (@mod(g.n, 8) != 0 or @mod(g.k, 64) != 0) return error.UnsupportedProjectionGeometry;
        if (mx.dtype(p.weights[0]) != mx.c.MLX_UINT32 or mx.dtype(p.weights[1]) != mx.bf16 or mx.dtype(p.weights[2]) != mx.bf16) return error.InvalidTensorDType;
    }
    fn key(p: Projection) [3]i32 {
        return .{ mx.dim(p.weights[0], 0), mx.dim(p.weights[0], 1) * 8, p.group };
    }
};

pub const Dense = struct {
    // The first projection of each shape controls dispatch, as in upstream prepare().
    checked: std.AutoHashMapUnmanaged([3]i32, bool) = .empty,

    pub fn deinit(d: *Dense) void {
        d.checked.deinit(mx.allocator);
    }
    pub fn prepare(d: *Dense, kernels: *mx.Kernels, projections: []const Projection) !void {
        for (projections) |p| {
            try p.validate();
            if (d.checked.contains(p.key())) continue;
            try d.checked.put(mx.allocator, p.key(), try calibrate(kernels, p));
        }
    }
    pub fn apply(d: *Dense, kernels: *mx.Kernels, s: *mx.Scope, x: A, p: Projection) !A {
        const scalar_ok = d.checked.get(p.key()) orelse return error.UncalibratedProjection;
        const rows = mx.dim(x, 0);
        const n = p.key()[0];
        const limit: i32 = if (p.group == 32 and n > 6144) 3 else 2;
        return launch(kernels, s, x, p, scalar_ok and rows <= limit and scalarBlock(rows, splits(n), p.group) != 0, mx.simd_groups);
    }
};

fn splits(n: i32) i32 {
    return if (n <= 64) 32 else if (n <= 6144) 16 else 8;
}
fn scalarBlock(rows: i32, split: i32, group: i32) i32 {
    if (rows < 1 or rows > 4) return 0;
    const xb = if (rows == 1) 32 else @max(split, 16);
    return if (@mod(xb, split) == 0 and rows * xb * @as(i32, if (group == 64) 76 else 44) * 4 <= 20480) xb else 0;
}

pub fn launch(kernels: *mx.Kernels, s: *mx.Scope, x: A, p: Projection, scalar: bool, max_groups: i32) !A {
    try p.validate();
    const n, const k, const group = p.key();
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != k) return error.InvalidProjectionInput;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 65536 or max_groups < 1 or max_groups > 16) return error.InvalidLaneWidth;
    const split = splits(n);
    const inputs = [_]A{ try s.contiguous(x), p.weights[0], p.weights[1], p.weights[2], try s.scalar(1) };
    if (scalar) {
        const xb = scalarBlock(rows, split, group);
        if (xb == 0) return error.InvalidScalarRows;
        const nr: i32 = if (n > 2048) 2 else 1;
        const sgs = if (n > 2048) @max(1, @divTrunc(16, @divExact(32, split) * nr)) else 8;
        const per = sgs * @divExact(32, split) * nr;
        return (try kernels.run(s, src.simd_qmm_scalar, &inputs, &.{ ti("K", k), ti("N", n), ti("S", split), ti("SGS", sgs), ti("NR", nr), ti("XB", xb), ti("GS", group), ti("RS", rows) }, .{ @divTrunc(n + per - 1, per) * sgs * 32, 1, 1 }, .{ sgs * 32, 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
    }
    const rt = if (rows > 16 and rows <= 24) 3 else @min(2, @divTrunc(rows + 7, 8));
    var nt: i32 = if (@mod(n, 32) == 0) 4 else if (@mod(n, 16) == 0) 2 else 1;
    if (rt > 2) nt = @min(nt, 2);
    while (nt > 1 and split * rt * nt * 64 * 4 > 16384) nt = @divExact(nt, 2);
    const sgs = @min(split, max_groups);
    return (try kernels.run(s, src.simd_qmm_mma, &inputs, &.{ ti("K", k), ti("N", n), ti("S", split), ti("SGS", sgs), ti("NT", nt), ti("RT", rt), ti("GS", group) }, .{ @divTrunc(n + 8 * nt - 1, 8 * nt) * sgs * 32, @divTrunc(rows + 8 * rt - 1, 8 * rt), 1 }, .{ sgs * 32, 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
}

fn calibrate(kernels: *mx.Kernels, p: Projection) !bool {
    var s = mx.Scope{};
    defer s.deinit();
    var key = mx.c.mlx_array_new();
    const key_rc = mx.c.mlx_random_key(&key, 0);
    key = try s.result(key_rc, key);
    var normal = mx.c.mlx_array_new();
    const dims = [_]c_int{ 8, p.key()[1] };
    const normal_rc = mx.c.mlx_random_normal(&normal, &dims, 2, mx.f32t, 0, 1, key, mx.stream);
    normal = try s.result(normal_rc, normal);
    const x = try s.cast(try s.binary(mx.c.mlx_multiply, normal, try s.scalar(0.5)), mx.bf16);
    const full = try launch(kernels, &s, x, p, false, mx.simd_groups);
    for (1..5) |count| {
        const m: i32 = @intCast(count);
        if (scalarBlock(m, splits(p.key()[0]), p.group) == 0) continue;
        var row: i32 = 0;
        while (row <= 8 - m) : (row += if (m == 1) 1 else 8 - m) {
            const out = try launch(kernels, &s, try s.slice(x, 0, row, row + m), p, true, mx.simd_groups);
            var equal = mx.c.mlx_array_new();
            const rc = mx.c.mlx_array_equal(&equal, out, try s.slice(full, 0, row, row + m), false, mx.stream);
            equal = try s.result(rc, equal);
            var agrees = false;
            try mx.check(mx.c.mlx_array_item_bool(&agrees, equal));
            if (!agrees) return false;
        }
    }
    return true;
}

test "scalar staging respects upstream shared memory limits" {
    try std.testing.expectEqual(@as(i32, 32), scalarBlock(1, 32, 64));
    try std.testing.expectEqual(@as(i32, 16), scalarBlock(4, 16, 64));
    try std.testing.expectEqual(@as(i32, 0), scalarBlock(3, 32, 64));
    try std.testing.expectEqual(@as(i32, 0), scalarBlock(5, 8, 32));
}
