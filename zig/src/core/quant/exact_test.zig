//! Every cut of an MLX affine projection dequantizes to the same bits as the uncut rows and columns it covers.
const std = @import("std");
const mlx = @import("mlx.zig");
const slice = @import("slice.zig");
const ref = @import("../affine4_host.zig");
const Tensor = @import("types.zig").Tensor;

const n = 8;
const k = 512;
const groups = k / ref.group;

fn tensor(dtype: anytype, rows: usize, cols: usize, bytes: []u8) Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, cols, 1, 1, 1 }, .bytes = bytes };
}

/// The cut's values, row-major, by the reference dequantizer.
fn values(a: std.mem.Allocator, h: mlx.Host) ![]f32 {
    const out = try a.alloc(f32, h.words.shape[0] * h.scales.shape[1] * h.group);
    ref.dequantize(try aligned(a, u32, h.words.bytes), try aligned(a, u16, h.scales.bytes), try aligned(a, u16, h.biases.bytes), out);
    return out;
}

/// A cut's bytes as typed elements (a cut is a byte slice of any alignment).
fn aligned(a: std.mem.Allocator, comptime T: type, bytes: []const u8) ![]T {
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn expectCut(full: []const f32, cut: []const f32, rows: []const usize, col0: usize, cols: usize) !void {
    try std.testing.expectEqual(rows.len * cols, cut.len);
    for (rows, 0..) |row, r| for (0..cols) |c| {
        const want: u32 = @bitCast(full[row * k + col0 + c]);
        const got: u32 = @bitCast(cut[r * cols + c]);
        try std.testing.expectEqual(want, got);
    };
}

test "row spans, rank column groups and K halves keep every dequantized bit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7f0d);
    const x = try a.alloc(f32, n * k);
    for (x) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
    const words = try a.alloc(u32, n * k / 8);
    const scales = try a.alloc(u16, n * groups);
    const biases = try a.alloc(u16, n * groups);
    ref.quantize(x, words, scales, biases);
    const h: mlx.Host = .{
        .words = tensor(.i32, n, k / 8, std.mem.sliceAsBytes(words)),
        .scales = tensor(.bf16, n, groups, std.mem.sliceAsBytes(scales)),
        .biases = tensor(.bf16, n, groups, std.mem.sliceAsBytes(biases)),
        .bits = 4,
        .group = ref.group,
    };
    const full = try values(a, h);
    const all = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };

    const spans = [_]slice.Span{ .{ .from = 1, .to = 3 }, .{ .from = 5, .to = 8 } };
    try expectCut(full, try values(a, try mlx.sliceRows(a, h, &spans)), &.{ 1, 2, 5, 6, 7 }, 0, k);

    for ([_]usize{ 1, 2, 4, 8 }) |world| for (0..world) |rank| {
        const cut = try mlx.sliceCols(a, h, .{ .rank = rank, .world = world }, "exact");
        try expectCut(full, try values(a, cut), &all, rank * k / world, k / world);
    };

    const pair = try mlx.halves(a, h);
    for (pair, 0..) |half, i| try expectCut(full, try values(a, half), &all, i * k / 2, k / 2);
}
