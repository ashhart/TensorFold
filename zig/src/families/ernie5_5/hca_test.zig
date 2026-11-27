//! hca.zig against vLLM's C128 compressor as its kernels write it: the two-stage pool, the norm, GPT-J's rope at 128j.
const std = @import("std");
const h = @import("hca.zig");

const eps: f32 = 1e-6;

/// SparseAttnCompressC128Block8Kernel's pool: 8 warps of 16 tokens, each one's max and sums, rescaled by the top max.
fn vendorPool(kv: []const f64, s: []const f64, c: usize) f64 {
    var m: [8]f64 = undefined;
    var sw: [8]f64 = @splat(0);
    var pw: [8]f64 = @splat(0);
    for (0..8) |w| {
        m[w] = -std.math.inf(f64);
        for (0..16) |i| m[w] = @max(m[w], s[(16 * w + i) * h.head_dim + c]);
        for (0..16) |i| {
            const k = (16 * w + i) * h.head_dim + c;
            const e = std.math.exp2((s[k] - m[w]) * std.math.log2e);
            sw[w] += e;
            pw[w] += kv[k] * e;
        }
    }
    var top = m[0];
    for (m[1..]) |v| top = @max(top, v);
    var ss: f64 = 0;
    var ps: f64 = 0;
    for (0..8) |w| {
        const sc = std.math.exp2((m[w] - top) * std.math.log2e);
        ss += sw[w] * sc;
        ps += pw[w] * sc;
    }
    return ps / ss;
}

test "a compressed row is vLLM's: the two-stage pool, RMSNorm 1 / sqrt(mean + eps), GPT-J's pairs at the block's start" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x686361);
    const rng = prng.random();
    const n = h.ratio * h.head_dim;
    const kv = try a.alloc(f64, n);
    defer a.free(kv);
    const score = try a.alloc(f64, n);
    defer a.free(score);
    const ape = try a.alloc(f32, n);
    defer a.free(ape);
    const s = try a.alloc(f64, n);
    defer a.free(s);
    for (kv, score, ape, s) |*x, *y, *z, *q| {
        x.* = rng.float(f64) * 4 - 2;
        y.* = rng.float(f64) * 16 - 8;
        z.* = rng.float(f32) * 2 - 1;
        q.* = y.* + z.*;
    }
    var w: [h.head_dim]f64 = undefined;
    for (&w) |*g| g.* = 0.5 + rng.float(f64);
    var freq: [h.rope / 2]f32 = undefined;
    for (&freq, 0..) |*f, l| f.* = @floatCast(std.math.pow(f64, 160000, -@as(f64, @floatFromInt(2 * l)) / h.rope));
    // the boundary token 128 j + 127 writes row j, its rope at cpos = (pos // 128) 128, the block's first position
    const boundary: i64 = 128 * 4321 + 127;
    const cpos = @divFloor(boundary, h.ratio) * h.ratio;
    var t: [2 * h.ratio]f64 = undefined;
    var got: [h.head_dim]f64 = undefined;
    h.compress(kv, score, ape, &w, &freq, cpos, eps, h.exact, &t, &got);
    var want: [h.head_dim]f64 = undefined;
    var ss: f64 = 0;
    for (&want, 0..) |*o, c| {
        o.* = vendorPool(kv, s, c);
        ss += o.* * o.*;
    }
    const rr = 1 / @sqrt(ss / h.head_dim + eps);
    for (&want, w) |*o, g| o.* = o.* * rr * g;
    const unturned = want;
    for (0..h.rope / 2) |l| {
        const ang: f64 = freq[l] * @as(f32, @floatFromInt(cpos));
        const x0 = want[h.nope + 2 * l];
        const x1 = want[h.nope + 2 * l + 1];
        want[h.nope + 2 * l] = x0 * @cos(ang) - x1 * @sin(ang);
        want[h.nope + 2 * l + 1] = x0 * @sin(ang) + x1 * @cos(ang);
    }
    for (got, want) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-11 * (1 + @abs(v)));
    // the nope dims are as normed, and the rope dims are turned (cpos is far from 0)
    for (got[0..h.nope], unturned[0..h.nope]) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-11 * (1 + @abs(v)));
    var moved: f64 = 0;
    for (got[h.nope..], unturned[h.nope..]) |g, v| moved = @max(moved, @abs(g - v));
    try std.testing.expect(moved > 1e-3);
}

test "the pool weighs kv by the softmax of score + ape over the tokens, for each channel alone" {
    const a = std.testing.allocator;
    const n = h.ratio * h.head_dim;
    const kv = try a.alloc(f64, n);
    defer a.free(kv);
    const score = try a.alloc(f64, n);
    defer a.free(score);
    const ape = try a.alloc(f32, n);
    defer a.free(ape);
    @memset(score, 0);
    @memset(ape, 0);
    for (kv, 0..) |*x, k| x.* = @floatFromInt(k / h.head_dim);
    // equal scores: the mean of the tokens' kv, 0 to 127
    var t: [2 * h.ratio]f64 = undefined;
    try std.testing.expectApproxEqAbs(63.5, h.pool(kv, score, ape, 3, h.exact, &t), 1e-12);
    // ape alone picks token 9 in channel 5: kv 9
    ape[9 * h.head_dim + 5] = 60;
    try std.testing.expectApproxEqAbs(9, h.pool(kv, score, ape, 5, h.exact, &t), 1e-12);
    try std.testing.expectApproxEqAbs(63.5, h.pool(kv, score, ape, 4, h.exact, &t), 1e-12);
}

test "check refuses narrow or odd rows and more rows than the grid" {
    try h.check(.{ .rows = 1 });
    try h.check(.{ .rows = 64, .ts = 2 * h.head_dim + 2, .os = h.head_dim + 2 });
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .ts = 2 * h.head_dim - 2 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .ts = 2 * h.head_dim + 1 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .os = h.head_dim - 2 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = std.math.maxInt(i32) / h.ratio + 1 }));
}

test "check refuses strides past i64, and ts or os whose furthest offset at these rows would pass it" {
    const most: usize = std.math.maxInt(i64);
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .ts = most + 1 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .os = most + 1 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 0, .ts = most + 1, .os = most + 1 }));
    // the largest even strides one row allows, then one step more, then the same strides at two rows
    const ts = (most - 2 * h.head_dim) / h.ratio / 2 * 2;
    const os = (most - h.head_dim) / 2 * 2;
    try h.check(.{ .rows = 1, .ts = ts, .os = os });
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .ts = ts + 2 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 1, .os = os + 2 }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 2, .ts = ts }));
    try std.testing.expectError(error.HcaShape, h.check(.{ .rows = 2, .os = os }));
}
