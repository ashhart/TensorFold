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

/// vLLM's eager reference for the DSv4 sink: softmax over [scores, sink], the sink's column dropped, kv weighted.
fn vendorAttend(q: []const f64, kv: []const f64, sink: f64, scale: f64, out: []f64) void {
    var buf: [600]f64 = undefined;
    const n = kv.len / h.head_dim;
    const full = buf[0 .. n + 1];
    for (full[0..n], 0..) |*v, j| {
        var dot: f64 = 0;
        for (0..h.head_dim) |d| dot += q[d] * kv[j * h.head_dim + d];
        v.* = dot * scale;
    }
    full[n] = sink;
    var top = -std.math.inf(f64);
    for (full) |v| top = @max(top, v);
    var z: f64 = 0;
    for (full) |*v| {
        v.* = @exp(v.* - top);
        z += v.*;
    }
    for (full) |*v| v.* /= z;
    @memset(out, 0);
    for (full[0..n], 0..) |pv, j| for (out, 0..) |*o, d| {
        o.* += pv * kv[j * h.head_dim + d];
    };
}

test "a query's keys are vLLM's: the window max(p - 127, 0) .. p, then (p + 1) // 128 compressed rows" {
    const cases = [_][4]u64{ .{ 0, 0, 0, 1 }, .{ 126, 0, 0, 127 }, .{ 127, 0, 1, 129 }, .{ 128, 1, 1, 129 }, .{ 255, 128, 2, 130 }, .{ 128 * 5 + 127, 128 * 5, 6, 134 }, .{ 49279, 49152, 385, 513 }, .{ (1 << 20) - 1, (1 << 20) - 128, 8192, 8320 } };
    for (cases) |x| {
        try std.testing.expectEqual(x[1], h.windowStart(x[0]));
        try std.testing.expectEqual(x[2], h.compressedRows(x[0]));
        try std.testing.expectEqual(x[3], h.keys(x[0]));
    }
}

test "a head is vLLM's softmax over its scores and its sink, the sink a zero-value key not scaled by 512^-0.5" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x61747464);
    const rng = prng.random();
    const scale = 1 / @sqrt(@as(f64, h.head_dim));
    for ([_]usize{ 1, 3, 130 }) |n| {
        const kv = try a.alloc(f64, n * h.head_dim);
        defer a.free(kv);
        for (kv, 0..) |*x, i| x.* = rng.float(f64) * 2 - 1 + @as(f64, if ((i / h.head_dim) % 3 == 1) 2 else 0);
        var q: [h.head_dim]f64 = undefined;
        for (&q) |*x| x.* = rng.float(f64) * 2 - 1;
        const t = try a.alloc(f64, 2 * n + 1 + @max(n, h.head_dim));
        defer a.free(t);
        for ([_]f64{ -std.math.inf(f64), -3, 0.7, 6 }) |sink| {
            var got: [h.head_dim]f64 = undefined;
            var want: [h.head_dim]f64 = undefined;
            h.attend(&q, kv, sink, scale, h.exact, t, &got);
            vendorAttend(&q, kv, sink, scale, &want);
            for (got, want) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-12 * (1 + @abs(v)));
        }
    }
}

test "the sink only adds to the denominator: one key at score 0 and sink 2 gives the key times 1 / (1 + e^2)" {
    const a = std.testing.allocator;
    const q: [h.head_dim]f64 = @splat(0);
    const kv = try a.alloc(f64, h.head_dim);
    defer a.free(kv);
    for (kv, 0..) |*x, d| x.* = @floatFromInt(d);
    var t: [2 + 1 + h.head_dim]f64 = undefined;
    var got: [h.head_dim]f64 = undefined;
    h.attend(&q, kv, 2, 0.5, h.exact, &t, &got);
    for (got, kv) |g, k| try std.testing.expectApproxEqAbs(k / (1 + @exp(2.0)), g, 1e-12 * (1 + k));
    h.attend(&q, kv, -std.math.inf(f64), 0.5, h.exact, &t, &got);
    for (got, kv) |g, k| try std.testing.expectEqual(k, g);
}

test "attnCheck refuses no heads, keys or chunks, chunks short of the lists' width, and grids or partials too large" {
    const no = error.HcaShape;
    try h.attnCheck(.{ .rows = 2, .heads = 64, .w = 8320, .nch = 17 });
    try h.attnCheck(.{ .rows = 0, .heads = 32, .w = 1, .nch = 1 });
    try std.testing.expectError(no, h.attnCheck(.{ .rows = 1, .heads = 0, .w = 1, .nch = 1 }));
    try std.testing.expectError(no, h.attnCheck(.{ .rows = 1, .heads = 64, .w = 0, .nch = 1 }));
    try std.testing.expectError(no, h.attnCheck(.{ .rows = 1, .heads = 64, .w = 1, .nch = 0 }));
    try std.testing.expectError(no, h.attnCheck(.{ .rows = 2, .heads = 64, .w = 8320, .nch = 16 })); // 8,192 keys
    try std.testing.expectError(no, h.attnCheck(.{ .rows = 1, .heads = 64, .w = 1, .nch = 65536 }));
    try std.testing.expectError(no, h.attnCheck(.{ .rows = std.math.maxInt(i32) + 1, .heads = 1, .w = 1, .nch = 1 }));
    try std.testing.expectError(no, h.attnCheck(.{ .rows = std.math.maxInt(i32), .heads = 16 * 65535, .w = 1, .nch = 65535 }));
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
