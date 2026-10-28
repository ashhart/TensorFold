//! The drafter's host quantization: MLX affine-4 g64 codes and tables, in the layout the HIP MLX projections read.

const std = @import("std");
const quant = @import("core").quant;
const weights = @import("weights.zig");
const config = @import("config.zig");

const gpa = std.testing.allocator;

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// The MLX rule written out once more: f32 min and max, bf16 scale and bias, codes from the rounded tables.
fn expected(w: []const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    @memset(words, 0);
    for (0..n) |r| for (0..k / 64) |g| {
        const row = w[r * k + g * 64 ..][0..64];
        var lo: f32 = bf16(row[0]);
        var hi: f32 = lo;
        for (row) |h| {
            lo = @min(lo, bf16(h));
            hi = @max(hi, bf16(h));
        }
        const s = quant.convert.bf16((hi - lo) / 15.0);
        const b = quant.convert.bf16(lo);
        scales[r * k / 64 + g] = s;
        biases[r * k / 64 + g] = b;
        const inv: f32 = if (bf16(s) == 0) 0 else 1.0 / bf16(s);
        for (row, 0..) |h, i| {
            const q: u32 = @intFromFloat(std.math.clamp(@round((bf16(h) - bf16(b)) * inv), 0, 15));
            const at = r * k + g * 64 + i;
            words[at / 8] |= q << @intCast(4 * (at % 8));
        }
    };
}

test "a hand-checked group: codes 0..15 low nibble first, scale 0.5, bias -1; a flat group has scale 0" {
    var w: [128]u16 = undefined;
    for (w[0..64], 0..) |*h, i| h.* = quant.convert.bf16(-1.0 + 0.5 * @as(f32, @floatFromInt(i % 16)));
    @memset(w[64..], 0x3f80);
    var q = try weights.quantize(gpa, &w, 1, 128);
    defer q.deinit(gpa);
    try std.testing.expectEqualSlices(u16, &.{ 0x3f00, 0 }, q.scales);
    try std.testing.expectEqualSlices(u16, &.{ 0xbf80, 0x3f80 }, q.biases);
    for (q.words[0..8], 0..) |word, j|
        try std.testing.expectEqual(@as(u32, if (j % 2 == 0) 0x76543210 else 0xfedcba98), word);
    for (q.words[8..]) |word| try std.testing.expectEqual(@as(u32, 0), word);
}

test "random bf16 rows quantize to the MLX rule's bytes and dequantize within the affine bound" {
    const n = 5;
    const k = 192;
    var prng = std.Random.DefaultPrng.init(0xD1A5);
    const rnd = prng.random();
    var w: [n * k]u16 = undefined;
    for (&w) |*h| h.* = quant.convert.bf16((rnd.float(f32) - 0.5) * 0.2);
    var q = try weights.quantize(gpa, &w, n, k);
    defer q.deinit(gpa);
    var words: [n * k / 8]u32 = undefined;
    var scales: [n * k / 64]u16 = undefined;
    var biases: [n * k / 64]u16 = undefined;
    expected(&w, n, k, &words, &scales, &biases);
    try std.testing.expectEqualSlices(u32, &words, q.words);
    try std.testing.expectEqualSlices(u16, &scales, q.scales);
    try std.testing.expectEqualSlices(u16, &biases, q.biases);
    // the HIP MLX format's own fp64 dequantization reads the codes back: the layout is the one its kernels take
    const h = q.host();
    try std.testing.expectEqual(@as(usize, n * k), quant.mlx.elements(h));
    var deq: [n * k]f64 = undefined;
    try quant.mlx.reference(h, &deq);
    for (0..n) |r| for (0..k / 64) |g| {
        const row = w[r * k + g * 64 ..][0..64];
        var lo: f64 = bf16(row[0]);
        var hi: f64 = lo;
        for (row) |x| {
            lo = @min(lo, bf16(x));
            hi = @max(hi, bf16(x));
        }
        const s: f64 = bf16(q.scales[r * k / 64 + g]);
        const b: f64 = bf16(q.biases[r * k / 64 + g]);
        const bound = s / 2 + 15 * @abs(s - (hi - lo) / 15) + @abs(lo - b) + 1e-9;
        for (row, 0..) |x, i| try std.testing.expect(@abs(deq[r * k + g * 64 + i] - bf16(x)) <= bound);
    };
}

test "a group holding an infinity or a NaN is refused" {
    for ([_]u16{ 0x7f80, 0xff80, 0x7fc0 }) |bad| {
        var w: [128]u16 = @splat(0x3f80);
        w[69] = bad;
        try std.testing.expectError(error.BadDraftWeights, weights.quantize(gpa, &w, 1, 128));
    }
}

test "a projection whose inputs are not whole groups of 64, or with no rows, is refused" {
    const w: [96]u16 = @splat(0x3f80);
    try std.testing.expectError(error.BadDraftShape, weights.quantize(gpa, &w, 1, 96));
    try std.testing.expectError(error.BadDraftShape, weights.quantize(gpa, &w, 0, 64));
    try std.testing.expectError(error.BadDraftShape, weights.quantize(gpa, w[0..64], 2, 64));
}

test "the loader's shape follows the drafter config" {
    var c: config.Config = undefined;
    c.hidden = 5120;
    c.intermediate = 17408;
    c.layers = 5;
    c.vocab = 248320;
    c.heads = 32;
    c.kv_heads = 8;
    c.head_dim = 128;
    c.group = 16;
    c.rank = 256;
    c.n_taps = 5;
    const s = weights.Shape.of(c);
    try std.testing.expectEqual(@as(usize, 4096), s.q_width);
    try std.testing.expectEqual(@as(usize, 1024), s.kv_width);
    try std.testing.expectEqual(@as(usize, 1280), s.dynamic);
    try std.testing.expectEqual(@as(usize, 5), s.taps);
}
