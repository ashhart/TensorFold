//! Inkling 2's attention in the order of #4125's kernels, in fp32 on the host (docs/inkling2-attention-reference.md).
const std = @import("std");
const attn = @import("attn.zig");
const bf16_round = @import("bf16_round");

/// Keys a tile, a global sub-chunk, a sliding chunk and a global chunk hold (attention.cu kKeys, kSub; kernels.zig).
pub const tile = 64;
pub const sub = 1024;
pub const sliding_chunk = 256;
pub const global_chunk = 16384;
/// (row, head) pairs a decode block holds (kernels.zig tile_pairs): its rows decide which keys a row's block loads.
pub const block_pairs = 64;

const log2e: f32 = 1.4426950408889634;
const inf = std.math.inf(f32);

/// `round_out` false leaves the gated output unrounded (fp32), for the bound.
pub const Options = struct { round_out: bool = true };

fn rn(x: f32) f32 {
    return bf16_round.round32(x);
}

/// The kernels' scale constant: fp32 1 / sqrt(hd) (kernels.zig:180).
pub fn scale(hd: usize) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
}

/// A running (max, sum, weighted values) in log2 units.
const Acc = struct {
    m: f32 = -inf,
    l: f32 = 0,
    o: [attn.most_hd]f32 = @splat(0),

    /// A part (m, l, o) folded in: both sides scaled by exp2 of their max less the joint max, each operation rounded.
    fn fold(a: *Acc, hd: usize, m: f32, l: f32, o: []const f32) void {
        const mn = @max(a.m, m);
        const x = @exp2(a.m - mn);
        const y = @exp2(m - mn);
        a.l = a.l * x + l * y;
        for (a.o[0..hd], o) |*u, w| u.* = u.* * x + w * y;
        a.m = mn;
    }
};

/// One (row, head) pair: its inputs, its key head, its position and the keys its block loads.
const Pair = struct {
    s: attn.Shape,
    in: attn.Inputs,
    q: []const f32,
    rel: []const f32,
    kh: usize,
    t: usize,
    lo: usize,
    tile_lo: usize,
    tile_hi: usize,

    /// The bias table's entry: the sum over the taps in order, rounded once to bf16.
    fn table(c: *const Pair, dist: usize) f32 {
        var acc: f32 = 0;
        for (c.rel, 0..) |a, j| acc += a * c.in.proj[j * c.s.rext + dist];
        return rn(acc);
    }

    /// The key's score in log2 units: (dot * scale + bias) * log2e, the dot a chain of k16 mma (here: in order).
    fn logit(c: *const Pair, key: usize) f32 {
        const k = c.in.k[(key * c.s.kv_heads + c.kh) * c.s.hd ..][0..c.s.hd];
        var dot: f32 = 0;
        for (c.q, k) |a, b| dot += a * b;
        const dist = c.t - key;
        const bias: f32 = if (dist < c.s.rext) c.table(dist) else 0;
        return (dot * scale(c.s.hd) + bias) * log2e;
    }

    /// Keys s_lo..s_hi as the kernel's block runs them: tiles of 64 from the block's first loaded key, a rescale each.
    fn step(c: *const Pair, s_lo: usize, s_hi: usize, a: *Acc) void {
        const hd = c.s.hd;
        a.* = .{};
        var lane: [4]f32 = @splat(0);
        const k_lo = @max(s_lo, c.tile_lo);
        const k_hi = @min(s_hi, c.tile_hi);
        var kt = k_lo / tile * tile;
        while (kt <= k_hi) : (kt += tile) {
            var sc: [tile]f32 = undefined;
            var mx: f32 = -inf;
            for (&sc, 0..) |*x, i| {
                const key = kt + i;
                const live = key >= c.lo and key <= c.t and key >= s_lo and key <= s_hi;
                x.* = if (live) c.logit(key) else -inf;
                mx = @max(mx, x.*);
            }
            const mn = @max(a.m, mx);
            const mu: f32 = if (mn == -inf) 0 else mn;
            const alpha = @exp2(a.m - mu);
            a.m = mn;
            for (&lane) |*x| x.* *= alpha;
            var p: [tile]f32 = undefined;
            for (&p, sc) |*x, v| x.* = rn(@exp2(v - mu));
            for (0..4) |t4| {
                for (0..tile / 8) |j| {
                    for (0..2) |b| lane[t4] += p[8 * j + 2 * t4 + b];
                }
            }
            for (0..hd) |d| {
                var acc = a.o[d] * alpha;
                for (&p, 0..) |w, i| {
                    const key = kt + i;
                    if (key < k_lo or key > k_hi) continue;
                    acc += w * c.in.v[(key * c.s.kv_heads + c.kh) * hd + d];
                }
                a.o[d] = acc;
            }
        }
        a.l = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    }
};

/// Row `row`, head `h` of a call into `out[0..hd]`; its block is the 64 pairs (row, head) of its key head it sits in.
fn pair(s: attn.Shape, in: attn.Inputs, row: usize, h: usize, out: []f32, opt: Options) void {
    const hd = s.hd;
    const g = s.heads / s.kv_heads;
    const t = s.pos0 + row;
    const hrow = row * s.heads + h;
    const m = row * g + h % g;
    const m0 = m / block_pairs * block_pairs;
    const c: Pair = .{
        .s = s,
        .in = in,
        .q = in.q[hrow * hd ..][0..hd],
        .rel = in.rel[hrow * s.drel ..][0..s.drel],
        .kh = h / g,
        .t = t,
        .lo = attn.first(s, t),
        .tile_lo = if (s.window > 0) (s.pos0 + m0 / g + 1) -| s.window else 0,
        .tile_hi = s.pos0 + @min(m0 + block_pairs - 1, s.rows * g - 1) / g,
    };
    const chunk: usize = if (s.window > 0) sliding_chunk else global_chunk;
    var run: Acc = .{};
    var ch = c.lo / chunk;
    while (ch <= t / chunk) : (ch += 1) {
        var part: Acc = .{};
        if (s.window > 0) {
            c.step(ch * chunk, ch * chunk + chunk - 1, &part);
        } else {
            var s_lo = ch * chunk;
            while (s_lo < (ch + 1) * chunk and s_lo <= t) : (s_lo += sub) {
                var one: Acc = undefined;
                c.step(s_lo, s_lo + sub - 1, &one);
                part.fold(hd, one.m, one.l, one.o[0..hd]);
            }
        }
        run.fold(hd, part.m, part.l, part.o[0..hd]);
    }
    const sk = in.sink[hrow] * log2e;
    const mf = @max(run.m, sk);
    const a = @exp2(run.m - mf);
    const den = run.l * a + @exp2(sk - mf);
    for (out, 0..) |*o, d| {
        const y = run.o[d] * a / den;
        const r = y * (1 / (1 + @exp(-in.gate[hrow * hd + d])));
        o.* = if (opt.round_out) rn(r) else r;
    }
}

/// The attention of `s` in the kernels' order into `out` [rows, heads, hd]: bf16 values in fp32 unless `!round_out`.
pub fn attendOrder(s: attn.Shape, in: attn.Inputs, out: []f32, opt: Options) attn.Error!void {
    try attn.shapedWithin(s, in, out.len, std.math.maxInt(usize));
    for (0..s.rows) |row| for (0..s.heads) |h| pair(s, in, row, h, out[(row * s.heads + h) * s.hd ..][0..s.hd], opt);
}
