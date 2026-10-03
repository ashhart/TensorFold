const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;

pub const Projection = struct {
    weights: [3]A,
    group: i32 = 64,
    bits: i32 = 4,
    reduction: ?i32 = null,

    pub fn validate(p: Projection) !void {
        if (p.bits != 4 and p.bits != 5 and p.bits != 6 and p.bits != 8) return error.UnsupportedQuantization;
        if (p.group != 64 and p.group != (if (p.bits == 4) @as(i32, 32) else 128)) return error.UnsupportedQuantization;
        if (p.reduction) |split| if (split != 8 and split != 16 and split != 32) return error.InvalidReduction;
        const g = try (@import("quantization.zig").Spec{ .bits = p.bits, .group_size = p.group }).shape(mx.shape(p.weights[0]), mx.shape(p.weights[1]), mx.shape(p.weights[2]));
        if (@mod(g.n, 8) != 0 or @mod(g.k, 64) != 0) return error.UnsupportedProjectionGeometry;
        if (mx.dtype(p.weights[0]) != mx.c.MLX_UINT32 or mx.dtype(p.weights[1]) != mx.bf16 or mx.dtype(p.weights[2]) != mx.bf16) return error.InvalidTensorDType;
    }
    pub fn key(p: Projection) [5]i32 {
        const n = mx.dim(p.weights[0], 0);
        return .{ n, @divExact(mx.dim(p.weights[0], 1) * 32, p.bits), p.group, p.bits, p.reduction orelse splits(n) };
    }
};

pub const Dense = struct {
    // The first projection of each shape controls dispatch, as in upstream prepare().
    checked: std.AutoHashMapUnmanaged([5]i32, bool) = .empty,

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
    pub fn prepareAdditional(d: *Dense, kernels: *mx.Kernels, projections: []const Projection) !void {
        var additional = Dense{};
        defer additional.deinit();
        try additional.prepare(kernels, projections);
        var it = additional.checked.iterator();
        while (it.next()) |entry| {
            const agrees = entry.value_ptr.* and (d.checked.get(entry.key_ptr.*) orelse true);
            try d.checked.put(mx.allocator, entry.key_ptr.*, agrees);
        }
    }
    pub fn apply(d: *Dense, kernels: *mx.Kernels, s: *mx.Scope, x: A, p: Projection) !A {
        const scalar_ok = d.checked.get(p.key()) orelse return error.UncalibratedProjection;
        if (p.bits != 4 and !scalar_ok) return error.SimdScalarMismatch;
        const rows = mx.dim(x, 0);
        const n = p.key()[0];
        const limit: i32 = if (p.bits != 4) 1 else if (p.group == 32 and n > 6144) 3 else 2;
        return launch(kernels, s, x, p, scalar_ok and rows <= limit and scalarBlock(rows, p.key()[4], p.group) != 0, mx.simd_groups);
    }
};

fn splits(n: i32) i32 {
    return if (n <= 64) 32 else if (n <= 6144) 16 else 8;
}
fn scalarBlock(rows: i32, split: i32, group: i32) i32 {
    if (rows < 1 or rows > 4) return 0;
    const xb = if (rows == 1) 32 else @max(split, 16);
    return if (@mod(xb, split) == 0 and rows * xb * @as(i32, if (group == 32) 44 else 76) * 4 <= 20480) xb else 0;
}

pub fn launch(kernels: *mx.Kernels, s: *mx.Scope, x: A, p: Projection, scalar: bool, max_groups: i32) !A {
    try p.validate();
    const n, const k, const group, const bits, const split = p.key();
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != k) return error.InvalidProjectionInput;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 65536 or max_groups < 1 or max_groups > 16) return error.InvalidLaneWidth;
    const inputs = [_]A{ try s.contiguous(x), p.weights[0], p.weights[1], p.weights[2], try s.scalar(1) };
    if (scalar) {
        const xb = scalarBlock(rows, split, group);
        if (xb == 0) return error.InvalidScalarRows;
        const nr: i32 = if (bits != 8 and n > 2048) 2 else 1;
        const sgs = if (n > 2048) @max(1, @divTrunc(16, @divExact(32, split) * nr)) else 8;
        const per = sgs * @divExact(32, split) * nr;
        return (try kernels.run(s, if (bits == 4) src.simd_qmm_scalar else src.simd_qmm_bits_scalar, &inputs, &.{ ti("K", k), ti("N", n), ti("S", split), ti("SGS", sgs), ti("NR", nr), ti("XB", xb), ti("GS", group), ti("B", bits), ti("RS", rows) }, .{ @divTrunc(n + per - 1, per) * sgs * 32, 1, 1 }, .{ sgs * 32, 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
    }
    const rt = @min(2, @divTrunc(rows + 7, 8));
    var nt: i32 = if (@mod(n, 32) == 0) 4 else if (@mod(n, 16) == 0) 2 else 1;
    while (nt > 1 and split * rt * nt * 64 * 4 > 16384) nt = @divExact(nt, 2);
    const sgs = @min(split, max_groups);
    return (try kernels.run(s, if (bits == 4) src.simd_qmm_mma else src.simd_qmm_bits_mma, &inputs, &.{ ti("K", k), ti("N", n), ti("S", split), ti("SGS", sgs), ti("NT", nt), ti("RT", rt), ti("GS", group), ti("B", bits) }, .{ @divTrunc(n + 8 * nt - 1, 8 * nt) * sgs * 32, @divTrunc(rows + 8 * rt - 1, 8 * rt), 1 }, .{ sgs * 32, 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
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
        if (scalarBlock(m, p.key()[4], p.group) == 0) continue;
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
    try std.testing.expectEqual(@as(i32, 0), scalarBlock(3, 32, 128));
    try std.testing.expectEqual(@as(i32, 0), scalarBlock(5, 8, 32));
}
