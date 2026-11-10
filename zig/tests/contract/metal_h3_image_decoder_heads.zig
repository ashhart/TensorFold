//! h3d_heads and h3d_attention under the contract: q/k norms and rotary at their bf16 points, attention's range.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const parts = @import("metal_h3_image_decoder_parts.zig");

const c = parts.c;
const ar = contract.attention_range;
const Range = parts.Range;
const Ends = parts.Ends;
const Sum = parts.Sum;
const bf = parts.bf;
const value = parts.value;
const W = parts.W;
const D = parts.D;
const H = parts.heads;
/// The rotary dims at the front of each q and k head, as half-style pairs (i, i + 16).
const R = 32;

/// One tile of 261 tokens, the batch of 16 tiles, one small tile of 65 tokens.
const tiles_served = [_][2]usize{ .{ 1, 261 }, .{ 16, 261 }, .{ 1, 65 } };

// --- h3d_heads: per row and head, q and k RMS-normed (no weight) to bf16, the first 32 dims turned; v copied ------

pub const Heads = struct {
    qkv: []u16 = &.{},
    cs: []u16 = &.{},
    sn: []u16 = &.{},

    /// A lane's two squares, then the simdgroup's 5-level tree.
    fn depth(_: c.Shape) usize {
        return 2 + 5;
    }

    /// Rows: tiles; n: tokens a tile; k: the head dim each norm sums. Outputs: q, k and v [tiles, 16, tokens, 64].
    pub fn spec(name: []const u8, _: Heads) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 2 * (parts.div_ulps + parts.rsqrt_ulps) + 3,
            .mids = 3,
            .out = .bf16,
            .served = &.{ .{ .rows = tiles_served[0][0], .n = tiles_served[0][1], .k = D }, .{ .rows = tiles_served[1][0], .n = tiles_served[1][1], .k = D }, .{ .rows = tiles_served[2][0], .n = tiles_served[2][1], .k = D } },
        };
    }

    pub fn prepare(k: *Heads, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.n == 0 or s.k != D) return false;
        const m = s.rows * s.n;
        const al = t.host();
        k.qkv = try al.alloc(u16, m * 3 * W);
        k.cs = try al.alloc(u16, s.n * R);
        k.sn = try al.alloc(u16, s.n * R);
        parts.uniform(rng, k.qkv, -3, 3);
        // gate 4: q and k integers whose squares' sums pass fp16's range and stay exact in fp32
        if (t.fill == .stress) for (0..m * H) |i| parts.integers(rng, k.qkv[i * 3 * D ..][0 .. 2 * D], -64, 64);
        for (k.cs, k.sn) |*cv, *sv| {
            const angle = rng.float(f32) * 2 * std.math.pi;
            cv.* = c.toBf16(@cos(angle));
            sv.* = c.toBf16(@sin(angle));
        }
        if (t.fill == .nonfinite) {
            // NaN in a q's first half, +Inf in a k's second half, -Inf in a v, both infinities in one q
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 3 * D + D + 40 }, .{ 2, 6 * D + 2 * D + 7 }, .{ 3, 9 * D }, .{ 3, 9 * D + 33 } };
            for (at, parts.plants) |p, b| if (p[0] < m) {
                k.qkv[p[0] * 3 * W + p[1]] = b;
            };
        }
        try t.input(k.qkv);
        try t.input(k.cs);
        try t.input(k.sn);
        for (0..3) |_| try t.output(m * W * 2);
        return true;
    }

    /// A block's heads: a simdgroup a row and head.
    pub fn launch(_: *Heads, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        for (0..6) |i| t.bind(e, i, i);
        e.setValue(parts.Shape{ .tiles = @intCast(s.rows), .tokens = @intCast(s.n), .m = @intCast(s.rows * s.n) }, 6);
        e.dispatchGroups(mtl.Size.of(s.rows * s.n, H, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn count(_: *Heads, s: c.Shape) usize {
        return 3 * s.rows * s.n * W;
    }

    const Job = struct {
        k: *const Heads,
        s: c.Shape,
        sum: Sum,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        /// Rows [from, to): each head's q and k normed and turned, and v.
        fn rows(j: Job, from: usize, to: usize) void {
            const m = j.s.rows * j.s.n;
            for (from..to) |r| for (0..H) |hd| {
                const tk = r % j.s.n;
                const to_at = ((r / j.s.n * H + hd) * j.s.n + tk) * D;
                const src = j.k.qkv[r * 3 * W + hd * 3 * D ..][0 .. 3 * D];
                for (0..2) |part| j.norm(src[part * D ..][0..D], j.k.cs[tk * R ..][0..R], j.k.sn[tk * R ..][0..R], part * m * W + to_at);
                for (src[2 * D ..][0..D], 0..) |b, i| j.ends.put(j.want, j.mag, 2 * m * W + to_at + i, value(b), Range.of(value(b)));
            };
        }

        fn norm(j: Job, src: []const u16, cs: []const u16, sn: []const u16, at: usize) void {
            var x: [D]f64 = undefined;
            var terms: [D]f64 = undefined;
            for (&x, &terms, src) |*v, *q, b| {
                v.* = value(b);
                q.* = v.* * v.*;
            }
            const ss = j.sum.of(&terms, depth(j.s));
            const inv = parts.invNorm(ss.r, D);
            const inv64 = 1 / @sqrt(ss.v / D + parts.eps);
            var na: [R]f64 = undefined;
            var nr: [R]Range = undefined;
            for (&na, &nr, x[0..R]) |*v, *vr, xv| {
                v.* = bf(xv * inv64);
                vr.* = Range.of(xv).mul(inv).fp32(0.5).bf16();
            }
            for (0..R) |i| {
                // rotate-half: the partner 16 dims away, negated below the half
                const p = i ^ 16;
                const sign: f64 = if (i < 16) -1 else 1;
                const cv = value(cs[i]);
                const sv = sign * value(sn[i]);
                const y = bf(bf(na[i] * cv) + bf(na[p] * sv));
                const yr = nr[i].mul(Range.of(cv)).fp32(0.5).bf16().add(nr[p].mul(Range.of(sv)).fp32(0.5).bf16()).fp32(0.5).bf16();
                j.ends.put(j.want, j.mag, at + i, y, yr);
                const z = x[R + i];
                j.ends.put(j.want, j.mag, at + R + i, bf(z * inv64), Range.of(z).mul(inv).fp32(0.5).bf16());
            }
        }
    };

    pub fn outputs(k: *Heads, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        const s = t.shape;
        parts.parallel(s.rows * s.n, Job{ .k = k, .s = s, .sum = sum, .ends = ends, .want = want, .mag = mag }, Job.rows);
    }
};

// --- h3d_attention: softmax(q k^T / 8) v over a tile's tokens, fp32 scores, softmax and sums ---------------------

/// Keys the kernel holds a query: 9 a lane.
const slots = 9 * 32;

/// The kernel's fp32 steps: a 64-deep score, precise exp, a lane's 9 keys and a tree, a chain over the keys, division.
fn form(tokens: usize) ar.Form {
    return .{ .score = D + 1, .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = @floatFromInt(tokens + 1), .den = 9 + 5 + 1, .div = 2 * parts.div_ulps, .exp_ulps = .{ parts.exp_ulps, 0 } };
}

pub const Attention = struct {
    q: []u16 = &.{},
    kk: []u16 = &.{},
    v: []u16 = &.{},

    /// A score's 64-deep chain, the softmax's exp and total, then the values' chain over every key.
    fn depth(s: c.Shape) usize {
        return D + s.k + 16;
    }

    /// Rows: tiles; n: the 1,024 outputs a query (16 heads of 64); k: tokens a tile, each a query and a key.
    pub fn spec(name: []const u8, _: Attention) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 2 * (parts.exp_ulps + parts.div_ulps) + 2,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 32, .min = parts.suffix + 1 } },
            .served = &.{ .{ .rows = tiles_served[0][0], .n = W, .k = tiles_served[0][1] }, .{ .rows = tiles_served[1][0], .n = W, .k = tiles_served[1][1] }, .{ .rows = tiles_served[2][0], .n = W, .k = tiles_served[2][1] } },
        };
    }

    pub fn prepare(k: *Attention, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        const T = s.k;
        if (s.rows == 0 or s.n != W or T == 0 or T > slots) return false;
        const n = s.rows * H * T * D;
        const al = t.host();
        k.q = try al.alloc(u16, n);
        k.kk = try al.alloc(u16, n);
        k.v = try al.alloc(u16, n);
        parts.uniform(rng, k.q, -1, 1);
        parts.uniform(rng, k.kk, -1, 1);
        parts.uniform(rng, k.v, -2, 2);
        if (t.fill == .stress) {
            // gate 4: every score 0, so p is 1; each value column's integers reach 2^17 and cancel to 4
            @memset(k.q, 0);
            for (0..s.rows * H) |pair| for (0..D) |d| for (0..T) |key| {
                k.v[(pair * T + key) * D + d] = parts.integer(even(key, T));
            };
        }
        if (t.fill == .nonfinite) {
            // (tile, head, token, dim): a NaN query, an infinite key, -Inf and both infinities and a NaN as values
            const at = [_][4]usize{ .{ 0, 0, 1, 3 }, .{ 1, 1, 2, 5 }, .{ 2, 2, 4, 7 }, .{ 3, 3, 0, 9 }, .{ 3, 3, 1, 9 }, .{ 4, 4, 5, 11 } };
            const into = [_]*[]u16{ &k.q, &k.kk, &k.v, &k.v, &k.v, &k.v };
            for (at, into, parts.plants ++ [_]u16{parts.nan16}) |p, buf, b| {
                buf.*[((@min(p[0], s.rows - 1) * H + p[1]) * T + p[2] % T) * D + p[3]] = b;
            }
        }
        try t.input(k.q);
        try t.input(k.kk);
        try t.input(k.v);
        try t.output(s.rows * T * W * 2);
        return true;
    }

    /// Gate 4's value `key` of a column: 127 or 2,040, then their negatives plus 1 on 4; integer partials, sum 4.
    fn even(key: usize, tokens: usize) i32 {
        const half = tokens / 2;
        if (key >= 2 * half) return 0;
        const at = if (key < half) key else key - half;
        const big: i32 = if (at % 2 == 0) 127 else 2040;
        return if (key < half) big else @as(i32, if (at < 8 and at % 2 == 0) 1 else 0) - big;
    }

    /// A block's attention: groups of 8 queries (256 threads) by 16 heads by tiles.
    pub fn launch(_: *Attention, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        for (0..4) |i| t.bind(e, i, i);
        e.setValue(parts.Shape{ .tiles = @intCast(s.rows), .tokens = @intCast(s.k), .m = @intCast(s.rows * s.k) }, 4);
        e.dispatchGroups(mtl.Size.of((s.k + 7) / 8, H, s.rows), mtl.Size.of(256, 1, 1));
    }

    pub fn count(_: *Attention, s: c.Shape) usize {
        return s.rows * s.k * W;
    }

    const Job = struct {
        k: *const Attention,
        s: c.Shape,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        /// (tile, head) pairs [from, to): each query's outputs within attention_range's float64 range.
        fn pairs(j: Job, from: usize, to: usize) void {
            const T = j.s.k;
            var keys: [slots * D]f64 = undefined;
            var vals: [slots * D]f64 = undefined;
            var w: [slots]f64 = undefined;
            var tau: [slots]f64 = undefined;
            var q: [D]f64 = undefined;
            var point: [D]f64 = undefined;
            var half: [D]f64 = undefined;
            for (from..to) |pair| {
                const base = pair * T * D;
                for (keys[0 .. T * D], vals[0 .. T * D], j.k.kk[base..][0 .. T * D], j.k.v[base..][0 .. T * D]) |*a, *b, x, y| {
                    a.* = value(x);
                    b.* = value(y);
                }
                for (0..T) |qi| {
                    for (&q, j.k.q[base + qi * D ..][0..D]) |*a, x| a.* = value(x);
                    ar.head(&q, keys[0 .. T * D], vals[0 .. T * D], T, D, parts.scale, form(T), w[0..T], tau[0..T], &point, &half);
                    const at = ((pair / H) * T + qi) * W + (pair % H) * D;
                    for (point, half, 0..) |p, hf, d| {
                        const e = ar.intervalBf16(p, hf, 0, 0);
                        j.ends.put(j.want, j.mag, at + d, bf(p), .{ .lo = e[0], .hi = e[1] });
                    }
                }
            }
        }
    };

    pub fn outputs(k: *Attention, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        const s = t.shape;
        if (t.fill == .stress) return k.equal(s, sum, ends, want, mag);
        parts.parallel(s.rows * H, Job{ .k = k, .s = s, .ends = ends, .want = want, .mag = mag }, Job.pairs);
    }

    /// Gate 4's equal scores: each output is its value column's sum over the keys' total, by the fast division.
    fn equal(k: *const Attention, s: c.Shape, sum: Sum, ends: *Ends, want: []f64, mag: []f64) void {
        const T = s.k;
        var col: [slots]f64 = undefined;
        const ones: [slots]f64 = @splat(1);
        const total = sum.of(ones[0..T], T);
        for (0..s.rows * H) |pair| for (0..D) |d| {
            for (col[0..T], 0..) |*x, key| x.* = value(k.v[(pair * T + key) * D + d]);
            const o = sum.of(col[0..T], T).v / total.v;
            const range = Range.of(o).fp32(parts.div_ulps).bf16();
            for (0..T) |qi| ends.put(want, mag, ((pair / H) * T + qi) * W + (pair % H) * D + d, bf(o), range);
        };
    }
};
