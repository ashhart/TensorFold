//! Serial one-row EXL3 dot from the scalar decoder: the bits the Metal one-row kernel must reproduce.
const std = @import("std");
const fmt = @import("exl3_format.zig");

/// One product into a column's TwoSum pair: `next` rounds once and its error is added to `low`.
fn add(high: *f32, low: *f32, product: f32) void {
    const next = high.* + product;
    const part = next - high.*;
    low.* += (high.* - (next - part)) + (product - part);
    high.* = next;
}

/// Tile columns [tn0, tn0 + out.len / 16) of one rotated row; stored tiles [x.len/16][tiles_n], k ascending.
pub fn dot(width: fmt.Width, codebook: fmt.Codebook, trellis: []const i16, tiles_n: usize, x: []const f16, tn0: usize, out: []f32) void {
    const words = width.tileWords();
    var wq: [256]f16 = undefined;
    for (0..out.len / 16) |t| {
        var high: [16]f32 = @splat(0);
        var low: [16]f32 = @splat(0);
        for (0..x.len / 16) |tk| {
            fmt.unpack(width, codebook, trellis[(tk * tiles_n + tn0 + t) * words ..][0..words], 1, 1, &wq);
            for (0..16) |r| {
                const xr: f32 = x[tk * 16 + r];
                // fp16 times fp16 is exact in fp32
                for (0..16) |c| add(&high[c], &low[c], xr * @as(f32, wq[r * 16 + c]));
            }
        }
        for (out[t * 16 ..][0..16], high, low) |*o, h, l| o.* = h + l;
    }
}

test "the compensated chain keeps small terms that a large cancelling pair hides from a plain fp32 chain" {
    const width = fmt.Width{ .halves = 8 };
    const words = width.tileWords();
    var prng = std.Random.DefaultPrng.init(0x524f5752);
    var trellis: [4 * 128]i16 = undefined;
    for (trellis[0..words]) |*v| v.* = prng.random().int(i16);
    for (1..4) |t| @memcpy(trellis[t * words ..][0..words], trellis[0..words]);
    // Row 0 adds 4096 * w, row 48 (the same tile row) takes it back; the 62 rows between are 2^-12 * w.
    var x: [64]f16 = @splat(0x1p-12);
    x[0] = 4096;
    x[48] = -4096;
    var got: [16]f32 = undefined;
    dot(width, .mcg, &trellis, 1, &x, 0, &got);
    var wq: [256]f16 = undefined;
    fmt.unpack(width, .mcg, trellis[0..words], 1, 1, &wq);
    var plain_differs = false;
    for (got, 0..) |g, c| {
        var exact: f64 = 0;
        var plain: f32 = 0;
        for (x, 0..) |xv, k| {
            exact += @as(f64, xv) * @as(f64, wq[(k % 16) * 16 + c]);
            plain += @as(f32, xv) * @as(f32, wq[(k % 16) * 16 + c]);
        }
        try std.testing.expectEqual(@as(f32, @floatCast(exact)), g);
        plain_differs = plain_differs or plain != g;
    }
    try std.testing.expect(plain_differs);
}
