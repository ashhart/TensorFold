//! key.zig against vLLM's forms of the indexer's k: F.layer_norm, then RotaryEmbedding's NeoX halves; its refusals.
const std = @import("std");
const k = @import("key.zig");

const d = k.head_dim;

/// F.layer_norm, then vLLM's NeoX ApplyRotaryEmb: dims below 64 in halves x1, x2, cat(x1 c - x2 s, x2 c + x1 s).
fn vendor(x: []const f64, w: []const f64, b: []const f64, inv: []const f32, pos: i64, out: []f64) void {
    var mean: f64 = 0;
    for (x) |v| mean += v / d;
    var v2: f64 = 0;
    for (x) |v| v2 += (v - mean) * (v - mean) / d;
    var y: [d]f64 = undefined;
    for (&y, x, w, b) |*o, v, g, h| o.* = (v - mean) / @sqrt(v2 + k.eps) * g + h;
    const x1 = y[0 .. k.rope / 2];
    const x2 = y[k.rope / 2 .. k.rope];
    for (inv, 0..) |f, i| {
        const t: f32 = @floatFromInt(pos); // the cache's fp32 positions times fp32 inv_freq
        const a: f64 = t * f;
        out[i] = x1[i] * @cos(a) - x2[i] * @sin(a);
        out[i + k.rope / 2] = x2[i] * @cos(a) + x1[i] * @sin(a);
    }
    @memcpy(out[k.rope..], y[k.rope..]);
}

test "a key is the vendor's LayerNorm, then its NeoX turn of the first 64 dims, at the start and deep in context" {
    var prng = std.Random.DefaultPrng.init(0x6b6579);
    const rng = prng.random();
    var inv: [k.rope / 2]f32 = undefined;
    for (&inv, 0..) |*f, i| f.* = 1.0 / @as(f32, @floatCast(std.math.pow(f64, 8.0e6, @as(f32, @floatFromInt(2 * i)) / k.rope)));
    var x: [d]f64 = undefined;
    var w: [d]f64 = undefined;
    var b: [d]f64 = undefined;
    // rows offset by 0, 64 and 128, then a row of one value: no variance, so its keys are its bias turned
    for (0..4) |row| {
        for (&x, &w, &b) |*v, *g, *h| {
            v.* = if (row == 3) 2.5 else 64 * @as(f64, @floatFromInt(row)) + rng.float(f64) * 8 - 4;
            g.* = rng.float(f64) * 3 - 0.5;
            h.* = rng.float(f64) * 6 - 3;
        }
        const n = k.norm(&x, k.exact);
        for ([_]i64{ 0, 1, 4095, 1_048_575 }) |pos| {
            var got: [d]f64 = undefined;
            var want: [d]f64 = undefined;
            k.key(&x, &w, &b, &inv, pos, k.exact, &got);
            vendor(&x, &w, &b, &inv, pos, &want);
            for (got, want) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-12 * (1 + @abs(v)));
            if (pos == 0) for (got, x, w, b) |g, v, s, h| try std.testing.expectEqual((v - n.mean) * n.rstd * s + h, g);
            if (row == 3) for (got[k.rope..], b[k.rope..]) |g, h| try std.testing.expectEqual(h, g);
        }
    }
}

test "check refuses rows narrower than a key and more rows than the grid" {
    try k.check(.{ .rows = 1 });
    try k.check(.{ .rows = 4096, .ks = d + 64, .os = d });
    try std.testing.expectError(error.KeyShape, k.check(.{ .rows = 1, .ks = d - 1 }));
    try std.testing.expectError(error.KeyShape, k.check(.{ .rows = 1, .os = k.rope }));
    try std.testing.expectError(error.KeyShape, k.check(.{ .rows = @as(usize, std.math.maxInt(i32)) + 1 }));
}
