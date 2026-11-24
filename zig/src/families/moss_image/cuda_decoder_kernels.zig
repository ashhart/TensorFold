//! MOSS-Image's image decoder kernels (moss_image_decoder.cu): their launches and picture geometry.
const std = @import("std");
const cuda = @import("cuda");

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

/// Pixels a GroupNorm statistics part sums, as the metal decoder's chunk.
pub const chunk = 256;

/// The chunks of `n` pixels a GroupNorm's statistics take.
pub fn chunks(n: usize) usize {
    return (n + chunk - 1) / chunk;
}

pub const Kernels = struct {
    mod: cuda.Module,
    unpack: cuda.Function,
    gemm: cuda.Function,
    conv3x3: cuda.Function,
    group_part: cuda.Function,
    group_finish: cuda.Function,
    norm: cuda.Function,
    softmax: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.moss_image_decoder);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .unpack = try mod.function("tf_md_unpack"),
            .gemm = try mod.function("tf_md_gemm"),
            .conv3x3 = try mod.function("tf_md_conv3x3"),
            .group_part = try mod.function("tf_md_group_part"),
            .group_finish = try mod.function("tf_md_group_finish"),
            .norm = try mod.function("tf_md_norm"),
            .softmax = try mod.function("tf_md_softmax"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// z = the latents [rows, columns, 128] with the batch norm undone, each 2 x 2 patch unpacked: [2 rows, 2 columns, 32].
pub fn unpack(k: *const Kernels, s: cuda.Stream, x: u64, scale: u64, mean: u64, z: u64, rows: usize, columns: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, scale, mean, z }) |v| a.add(v);
    for ([_]usize{ rows, columns }) |v| a.add(int(v));
    try go(k.unpack, s, .{ .x = ceil(rows * columns * 4, 8) }, 256, &a);
}

/// y [m, n] = x [m, k] . w + bias (+ r): w is [k, n] rows apart by `ldb`, or [n, k] when `transposed`.
pub fn gemm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, bias: u64, r: u64, y: u64, m: usize, n: usize, width: usize, lda: usize, ldb: usize, transposed: bool, add: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, bias, r, y }) |v| a.add(v);
    for ([_]usize{ m, n, width, lda, ldb }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(transposed)));
    a.add(@as(c_int, @intFromBool(add)));
    try go(k.gemm, s, .{ .x = ceil(n, 64), .y = ceil(m, 64) }, 256, &a);
}

/// A 3x3 convolution's size: its output picture, its channels, and whether x is read nearest from half size.
pub const ConvShape = struct { rows: usize, cols: usize, cin: usize, cout: usize, up: bool = false };

/// y = conv3x3(x) + b (+ r) at shape.rows x shape.cols, u [9 taps][cin][cout], the taps row major.
pub fn conv3x3(k: *const Kernels, s: cuda.Stream, x: u64, u: u64, bias: u64, r: u64, y: u64, shape: ConvShape, add: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, u, bias, r, y }) |v| a.add(v);
    for ([_]usize{ shape.rows, shape.cols, shape.cin, shape.cout }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(shape.up)));
    a.add(@as(c_int, @intFromBool(add)));
    try go(k.conv3x3, s, .{ .x = ceil(shape.cout, 32), .y = ceil(shape.rows, 8), .z = ceil(shape.cols, 16) }, 256, &a);
}

/// A chunk of pixels' sums of a GroupNorm's group, or of (x - mean)^2 when centered, into part [32, chunks].
pub fn groupPart(k: *const Kernels, s: cuda.Stream, x: u64, stats: u64, part: u64, pixels: usize, channels: usize, centered: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, stats, part }) |v| a.add(v);
    for ([_]usize{ pixels, channels, chunk, chunks(pixels) }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(centered)));
    try go(k.group_part, s, .{ .x = @intCast(chunks(pixels)) }, @intCast(channels), &a);
}

/// stats' 32 means from part, or (centered) its 32 rstd: 1 / sqrt(var + eps).
pub fn groupFinish(k: *const Kernels, s: cuda.Stream, part: u64, stats: u64, pixels: usize, channels: usize, centered: bool, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ part, stats }) |v| a.add(v);
    for ([_]usize{ pixels, channels, chunks(pixels) }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(centered)));
    a.add(eps);
    try go(k.group_finish, s, .{ .x = 1 }, 1024, &a);
}

/// y = GroupNorm(x) by stats (mean, rstd a group) with g and beta, then SiLU when silu; x, y [pixels, channels].
pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, stats: u64, g: u64, beta: u64, y: u64, pixels: usize, channels: usize, silu: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, stats, g, beta, y }) |v| a.add(v);
    a.add(int(channels));
    a.add(@as(c_int, @intFromBool(silu)));
    try go(k.norm, s, .{ .x = @intCast(pixels) }, @intCast(channels), &a);
}

/// scores [rows, rows] = softmax(scores * scale) of each row, in place.
pub fn softmax(k: *const Kernels, s: cuda.Stream, scores: u64, rows: usize, scale: f32) !void {
    var a: cuda.Args = .{};
    a.add(scores);
    a.add(int(rows));
    a.add(scale);
    try go(k.softmax, s, .{ .x = @intCast(rows) }, 256, &a);
}

test "launch geometry: the channels' tiles, the pixel chunks and the blocks a picture launches" {
    try std.testing.expectEqual(@as(u32, 18), ceil(1152, 64));
    try std.testing.expectEqual(@as(u32, 2), ceil(128, 64));
    try std.testing.expectEqual(@as(u32, 64), ceil(1024, 16));
    try std.testing.expectEqual(@as(u32, 128), ceil(1024, 8));
    try std.testing.expectEqual(@as(u32, 2048), ceil(64 * 64 * 4, 8));
    try std.testing.expectEqual(@as(usize, 4096), chunks(1024 * 1024));
    try std.testing.expectEqual(@as(usize, 64), chunks(16384));
    try std.testing.expectEqual(@as(usize, 1), chunks(1));
}
