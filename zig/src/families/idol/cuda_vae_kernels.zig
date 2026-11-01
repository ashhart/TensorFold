//! Idol's image encoder kernels on CUDA (idol_vae.cu, fp32): convolutions, GroupNorm, attention's softmax and the rows.
const std = @import("std");
const cuda = @import("cuda");

/// GroupNorm's groups, the pixels a statistics part sums, the convolution's tile, and the softmax's widest row.
pub const groups = 32;
pub const chunk = 256;
pub const tile_m = 128;
pub const tile_n = 128;
pub const tile_k = 8;
pub const softmax_max = 256 * 64;

/// Each function by its symbol, `tf_iv_` and the field's name.
pub const names = [_][]const u8{ "conv_in", "conv", "group_part", "group_finish", "norm", "bias", "softmax", "tokens" };

pub const Kernels = struct {
    mod: cuda.Module,
    conv_in: cuda.Function,
    conv: cuda.Function,
    group_part: cuda.Function,
    group_finish: cuda.Function,
    norm: cuda.Function,
    bias: cuda.Function,
    softmax: cuda.Function,
    tokens: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.idol_vae);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (names) |n| @field(k, n) = try mod.function("tf_iv_" ++ n);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, block: cuda.Dim3, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = block }, s, a);
}

fn addAll(a: *cuda.Args, ptrs: []const u64) void {
    for (ptrs) |p| a.add(p);
}

fn addInts(a: *cuda.Args, vals: []const usize) void {
    for (vals) |v| a.add(int(v));
}

/// tf_iv_conv_in: y [rows * cols, 128] = conv3x3(2 x - 1) + b of the picture x [rows * cols, 3], w [27, 128].
pub fn convIn(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, cols: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, y });
    addInts(&a, &.{ rows, cols });
    try go(k.conv_in, s, .{ .x = ceil(rows * cols, 2) }, .{ .x = 128, .y = 2 }, &a);
}

/// A convolution's output and input pictures, channels, weights' row, taps (1 or 9), stride and padding above and left.
pub const ConvShape = struct { rows: usize, cols: usize, in_rows: usize, in_cols: usize, cin: usize, cout: usize, ldw: usize, taps: usize = 1, stride: usize = 1, pad: usize = 0 };

/// The convolutions the encoder runs: a same-size 3x3, the stride-2 downsample, a 1x1 at `R` x `R`.
pub fn same(R: usize, cin: usize, cout: usize, ldw: usize) ConvShape {
    return .{ .rows = R, .cols = R, .in_rows = R, .in_cols = R, .cin = cin, .cout = cout, .ldw = ldw, .taps = 9, .pad = 1 };
}
pub fn down(R: usize, cin: usize, cout: usize, ldw: usize) ConvShape {
    return .{ .rows = R / 2, .cols = R / 2, .in_rows = R, .in_cols = R, .cin = cin, .cout = cout, .ldw = ldw, .taps = 9, .stride = 2 };
}
pub fn one(R: usize, cin: usize, cout: usize, ldw: usize) ConvShape {
    return .{ .rows = R, .cols = R, .in_rows = R, .in_cols = R, .cin = cin, .cout = cout, .ldw = ldw };
}

/// The weights' row length for `cout` output channels: whole column tiles, so no weight load needs a bound.
pub fn ldwOf(cout: usize) usize {
    return std.mem.alignForward(usize, cout, tile_n);
}

/// What tf_iv_conv requires of a shape: K in whole steps of one tap, float4 rows, whole weight tiles.
pub fn convOk(c: ConvShape) bool {
    return c.cin % tile_k == 0 and c.cout % 4 == 0 and c.ldw % tile_n == 0 and c.ldw >= c.cout and (c.taps == 1 or c.taps == 9) and
        (c.stride == 1 or c.stride == 2) and c.rows * c.cols > 0;
}

/// The grid of a convolution: 128-pixel tiles on x, 128-channel tiles of the weights' row on y.
pub fn convGrid(c: ConvShape) cuda.Dim3 {
    return .{ .x = ceil(c.rows * c.cols, tile_m), .y = @intCast(c.ldw / tile_n) };
}

/// tf_iv_conv: y = conv(x) + b (+ r when given), an implicit GEMM in fp32 FMAs (no tensor cores, no TF32).
pub fn conv(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, r: ?u64, y: u64, c: ConvShape) !void {
    std.debug.assert(convOk(c));
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, r orelse y, y });
    addInts(&a, &.{ c.rows, c.cols, c.in_rows, c.in_cols, c.cin, c.cout, c.ldw, c.taps, c.stride, c.pad, @intFromBool(r != null) });
    try go(k.conv, s, convGrid(c), .{ .x = 256 }, &a);
}

/// GroupNorm's parts over `n` pixels: one sum a chunk of `chunk` pixels and a group.
pub fn chunks(n: usize) usize {
    return (n + chunk - 1) / chunk;
}

/// tf_iv_group_part: part [32, chunks] = each chunk's sum of a group's values, or (centered) of (x - mean)^2.
pub fn groupPart(k: *const Kernels, s: cuda.Stream, x: u64, stats: u64, part: u64, n: usize, channels: usize, centered: bool) !void {
    std.debug.assert(channels % groups == 0 and channels <= 512);
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, stats, part });
    addInts(&a, &.{ n, channels, chunk, chunks(n), @intFromBool(centered) });
    try go(k.group_part, s, .{ .x = @intCast(chunks(n)) }, .{ .x = @intCast(channels) }, &a);
}

/// tf_iv_group_finish: stats' 32 means from part, or (centered) its 32 rstd.
pub fn groupFinish(k: *const Kernels, s: cuda.Stream, part: u64, stats: u64, n: usize, channels: usize, centered: bool, eps: f32) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ part, stats });
    addInts(&a, &.{ n, channels, chunks(n), @intFromBool(centered) });
    a.add(eps);
    try go(k.group_finish, s, .{ .x = 1 }, .{ .x = 32 * groups }, &a);
}

/// tf_iv_norm: y = GroupNorm(x) by stats (32 means, then 32 rstd) with g and beta, then SiLU when `silu`.
pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, stats: u64, g: u64, beta: u64, y: u64, n: usize, channels: usize, silu: bool) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, stats, g, beta, y });
    addInts(&a, &.{ n, channels, @intFromBool(silu) });
    try go(k.norm, s, .{ .x = ceil(n * channels, 256) }, .{ .x = 256 }, &a);
}

/// tf_iv_bias: y [rows, cols] = acc + b by column (or by row when `by_row`) (+ r when given); y may be acc.
pub fn bias(k: *const Kernels, s: cuda.Stream, acc: u64, b: u64, r: ?u64, y: u64, rows: usize, cols: usize, by_row: bool) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ acc, b, r orelse y, y });
    addInts(&a, &.{ rows, cols, @intFromBool(by_row), @intFromBool(r != null) });
    try go(k.bias, s, .{ .x = ceil(rows * cols, 256) }, .{ .x = 256 }, &a);
}

/// tf_iv_softmax: each of `rows` rows of s [rows, n] becomes softmax(s scale), in place; n <= softmax_max.
pub fn softmax(k: *const Kernels, s: cuda.Stream, scores: u64, rows: usize, n: usize, scale: f32) !void {
    std.debug.assert(n <= softmax_max);
    var a: cuda.Args = .{};
    a.add(scores);
    a.add(int(n));
    a.add(scale);
    try go(k.softmax, s, .{ .x = @intCast(rows) }, .{ .x = 256 }, &a);
}

/// tf_iv_tokens: out [pad + rows * cols / 4, 128] from moments [rows, cols, 64], noise [rows, cols, 32], m and s [128].
pub fn tokens(k: *const Kernels, s: cuda.Stream, moments: u64, noise: u64, m: u64, sd: u64, out: u64, rows: usize, cols: usize, pad: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ moments, noise, m, sd, out });
    addInts(&a, &.{ rows, cols, pad });
    try go(k.tokens, s, .{ .x = ceil(pad + rows * cols / 4, 2) }, .{ .x = 128, .y = 2 }, &a);
}

test "launch geometry: 128 x 128 conv tiles over the weights' padded row, a group-part block a chunk" {
    const c = same(1024, 128, 128, ldwOf(128));
    try std.testing.expect(convOk(c));
    try std.testing.expectEqual(@as(u32, 8192), convGrid(c).x);
    try std.testing.expectEqual(@as(u32, 1), convGrid(c).y);
    const o = same(128, 512, 64, ldwOf(64));
    try std.testing.expectEqual(@as(usize, 128), o.ldw);
    try std.testing.expect(convOk(o));
    const d = down(1024, 128, 128, ldwOf(128));
    try std.testing.expectEqual(@as(usize, 512), d.rows);
    try std.testing.expectEqual(@as(u32, 2048), convGrid(d).x);
    try std.testing.expect(!convOk(.{ .rows = 4, .cols = 4, .in_rows = 4, .in_cols = 4, .cin = 3, .cout = 128, .ldw = 128 }));
    try std.testing.expectEqual(@as(usize, 4096), chunks(1024 * 1024));
    try std.testing.expectEqual(@as(usize, 16384), softmax_max);
    // every grid dimension and kernel index fits its type at the widest stage
    try std.testing.expect(1024 * 1024 * 128 < std.math.maxInt(c_int));
}
