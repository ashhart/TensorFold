const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;

pub fn project(k: *mx.Kernels, s: *mx.Scope, x: A, weights: [3]A, bits: i32, float_output: bool, rps: i32) !A {
    if (mx.shape(x).len != 2 or mx.shape(weights[0]).len != 2) return error.InvalidTensorShape;
    const geometry = try (@import("quantization.zig").Spec{ .bits = bits }).shape(mx.shape(weights[0]), mx.shape(weights[1]), mx.shape(weights[2]));
    const rows = mx.dim(x, 0);
    const width = geometry.k;
    const n = geometry.n;
    const v: i32 = switch (bits) {
        4, 5 => 16,
        6, 8 => 8,
        else => return error.UnsupportedQuantization,
    };
    if (rows < 1 or rows > 32 or rps < 1 or rps > 8 or @mod(n, rps) != 0 or mx.dim(x, 1) != width or @mod(width, 32 * v) != 0 or @mod(n, if (bits == 4) @as(i32, 4) else 8) != 0) return error.UnsupportedProjectionGeometry;
    if (mx.dtype(weights[0]) != mx.c.MLX_UINT32 or mx.dtype(weights[1]) != mx.bf16 or mx.dtype(weights[2]) != mx.bf16) return error.InvalidTensorDType;
    if (float_output and bits != 4) return error.UnsupportedQuantization;
    const input = if (float_output) (if (mx.dtype(x) == mx.bf16 or mx.dtype(x) == mx.f32t) x else try s.cast(x, mx.f32t)) else try s.cast(x, mx.bf16);
    var params = [_]mx.Template{ ti("K", width), ti("N", n), ti("RPS", rps), ti("BITS", bits), ti("V", v), ti("LB", @divExact(v * bits, 8)) };
    const spec = if (float_output) src.ds4_qmv_rows_f32 else if (bits == 4) src.glm_qmv_rows64 else src.glm_qmv_rows_b;
    return (try k.run(s, spec, &.{ input, weights[0], weights[1], weights[2] }, params[0..if (bits == 4) @as(usize, 3) else 6], .{ 32 * rows, @divExact(n, rps), 1 }, .{ 32 * rows, 1, 1 }, &.{.{ .shape = &.{ rows, n }, .dtype = if (float_output) mx.f32t else mx.bf16 }}))[0];
}

pub fn normRope(k: *mx.Kernels, s: *mx.Scope, x: A, weight: ?A, positions: A, frequencies: A, eps: A, norm: bool, inverse: bool) !A {
    if (mx.shape(x).len < 2 or mx.shape(frequencies).len != 1 or mx.dtype(x) != mx.bf16 or mx.dtype(frequencies) != mx.f32t or mx.dtype(positions) != mx.i32t or mx.dtype(eps) != mx.f32t) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    const dims = mx.dim(x, -1);
    const pe = 2 * mx.dim(frequencies, 0);
    if (rows < 1 or dims < 64 or @mod(dims, 64) != 0 or pe < 2 or pe > dims or mx.c.mlx_array_size(positions) < rows or mx.c.mlx_array_size(eps) < 1) return error.InvalidRotaryGeometry;
    if (weight) |w| if (mx.c.mlx_array_size(w) != dims or mx.dtype(w) != mx.bf16) return error.InvalidTensorShape;
    const heads: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(rows * dims)));
    const result = (try k.run(s, src.ds4_norm_rope, &.{ x, weight orelse frequencies, positions, frequencies, eps }, &.{ ti("D", dims), ti("H", heads), ti("PE", pe), ti("NORM", @intFromBool(norm)), ti("WEIGHTED", @intFromBool(weight != null)), ti("INVERSE", @intFromBool(inverse)) }, .{ 32, rows * heads, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ rows * heads, dims } }}))[0];
    return s.reshape(result, mx.shape(x));
}

pub fn indexedAttention(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, indices: A, length: i32, scale: f32) !A {
    if (mx.shape(q).len != 3 or mx.shape(keys).len != 2 or mx.shape(indices).len != 2 or mx.dtype(q) != mx.bf16 or mx.dtype(keys) != mx.bf16 or mx.dtype(indices) != mx.i32t) return error.InvalidTensorShape;
    const rows = mx.dim(q, 0);
    const heads = mx.dim(q, 1);
    const dims = mx.dim(q, 2);
    const top = mx.dim(indices, 1);
    if (rows < 1 or heads < 1 or dims < 32 or dims > 512 or @mod(dims, 32) != 0 or mx.dim(keys, 1) != dims or mx.dim(indices, 0) != rows or top < 1 or length < 0 or length > mx.dim(keys, 0) or !std.math.isFinite(scale)) return error.InvalidAttentionGeometry;
    return (try k.run(s, src.glm_indexed_attention, &.{ q, keys, indices, try s.scalar(scale), try s.ints(&.{length}) }, &.{ ti("QK_DIM", dims), ti("TOPK", top), ti("HEADS", heads) }, .{ 1024, rows * heads, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = mx.shape(q) }}))[0];
}

pub fn hcStep(k: *mx.Kernels, s: *mx.Scope, streams: A, pending: ?[3]A, mix_weight: ?A, scale: A, base: A, norm: A, eps: f32, hc_eps: f32, iters: i32) ![4]A {
    if (mx.shape(streams).len != 3 or mx.dim(streams, 1) != 4 or mx.dim(streams, 2) != 4096 or mx.dtype(streams) != mx.bf16) return error.UnsupportedHyperconnectionGeometry;
    const rows = mx.dim(streams, 0);
    if (rows < 1 or rows > 16 or iters < 1 or iters > 1024) return error.UnsupportedHyperconnectionGeometry;
    const split = mix_weight != null;
    const branch = if (pending) |p| p[0] else try s.reshape(try s.slice(streams, 1, 0, 1), &.{ rows, 4096 });
    const post = if (pending) |p| p[1] else try s.zeros(&.{ rows, 4 }, mx.f32t);
    const comb = if (pending) |p| p[2] else try s.zeros(&.{ rows, 4, 4 }, mx.f32t);
    const epsilon = try s.scalar(eps);
    const expanded = try k.run(s, src.glm_hc_expand, &.{ streams, branch, post, comb, epsilon }, &.{ ti("D", 4096), ti("EXPAND", @intFromBool(pending != null)), ti("SPLIT", @intFromBool(split)), ti("SQ_FMA", 0), ti("ZOUT", 0) }, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 }, &.{ .{ .shape = if (pending != null) mx.shape(streams) else &.{1} }, .{ .shape = &.{rows}, .dtype = mx.f32t }, .{ .shape = &.{1}, .dtype = mx.f32t } });
    const x = if (pending != null) expanded[0] else streams;
    const fnw = mix_weight orelse return .{ x, mx.empty, mx.empty, mx.empty };
    const is_packed = mx.dtype(fnw) == mx.bf16;
    const mixes = (try k.run(s, if (is_packed) src.glm_hc_mix_packed else src.glm_hc_mix, &.{ x, expanded[1], fnw }, if (is_packed) &.{ ti("D", 4096), ti("U", 8) } else &.{ti("D", 4096)}, .{ 1536, rows, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, 24 }, .dtype = mx.f32t }}))[0];
    const result = try k.run(s, src.glm_hc_split_norm, &.{ x, mixes, scale, base, norm, epsilon }, &.{ ti("D", 4096), ti("SQ_FMA", 0), ti("ITERS", iters), ti("HC_EPS_INT", @intFromFloat(@round(hc_eps / 1e-9))) }, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 }, &.{ .{ .shape = &.{ rows, 4096 } }, .{ .shape = &.{ rows, 4 }, .dtype = mx.f32t }, .{ .shape = &.{ rows, 4, 4 }, .dtype = mx.f32t } });
    return .{ x, result[0], result[1], result[2] };
}

pub const Kda = struct {
    heads: i32,
    dims: i32,
    taps: i32,
    f_bits: i32,
    g_bits: i32,
    pub fn validate(g: Kda) !void {
        if (g.heads < 1 or g.heads > 256 or (g.dims != 64 and g.dims != 128) or g.taps < 1 or g.taps > 16 or (g.f_bits != 4 and g.f_bits != 8) or (g.g_bits != 4 and g.g_bits != 8)) return error.UnsupportedKdaGeometry;
    }
};
pub fn kda(k: *mx.Kernels, s: *mx.Scope, g: Kda, inputs: [15]A) ![3]A {
    try g.validate();
    const width = g.heads * g.dims;
    if (mx.shape(inputs[0]).len != 2) return error.InvalidTensorShape;
    const rows = mx.dim(inputs[0], 0);
    if (rows < 1 or rows > 16 or mx.dim(inputs[0], 1) != 3 * width + 2 * g.dims + g.heads) return error.InvalidTensorShape;
    const sizes = [_]usize{ @intCast(rows * (3 * width + 2 * g.dims + g.heads)), @intCast((g.taps - 1) * 3 * width), @intCast(g.taps * 3 * width), @intCast(@divExact(width * g.dims * g.f_bits, 32)), @intCast(@divExact(width * g.dims, 64)), @intCast(@divExact(width * g.dims, 64)), @intCast(@divExact(width * g.dims * g.g_bits, 32)), @intCast(@divExact(width * g.dims, 64)), @intCast(@divExact(width * g.dims, 64)), @intCast(g.heads), @intCast(width), @intCast(g.heads * g.dims * g.dims), @intCast(g.dims), 1, 1 };
    const types = [_]mx.c.mlx_dtype{ mx.bf16, mx.bf16, mx.f32t, mx.c.MLX_UINT32, mx.bf16, mx.bf16, mx.c.MLX_UINT32, mx.bf16, mx.bf16, mx.f32t, mx.f32t, mx.f32t, mx.f32t, mx.f32t, mx.f32t };
    for (inputs, sizes, types) |value, size, dtype| if (mx.c.mlx_array_size(value) != size or mx.dtype(value) != dtype) return error.InvalidTensorShape;
    const result = try k.run(s, src.glm_kda_rows, &inputs, &.{ ti("H", g.heads), ti("D", g.dims), ti("TAPS", g.taps), ti("TY", 32), ti("FB", g.f_bits), ti("GB", g.g_bits) }, .{ 32, 32, g.heads }, .{ 32, 32, 1 }, &.{ .{ .shape = &.{ rows, width } }, .{ .shape = mx.shape(inputs[11]), .dtype = mx.f32t }, .{ .shape = mx.shape(inputs[1]) } });
    return .{ result[0], result[1], result[2] };
}

test "KDA rejects unsupported widths and quantization" {
    try (Kda{ .heads = 64, .dims = 128, .taps = 4, .f_bits = 4, .g_bits = 8 }).validate();
    try std.testing.expectError(error.UnsupportedKdaGeometry, (Kda{ .heads = 64, .dims = 32, .taps = 4, .f_bits = 4, .g_bits = 4 }).validate());
    try std.testing.expectError(error.UnsupportedKdaGeometry, (Kda{ .heads = 64, .dims = 128, .taps = 4, .f_bits = 6, .g_bits = 4 }).validate());
}
