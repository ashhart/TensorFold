//! Attention launches: causal prefill and decode, the gate and the cache write.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

pub fn tf_causal(l: *const Launcher, q: CF, k: C, v: C, out: F, batch: c_int, qlen: c_int, span: c_int, heads: c_int, kv_heads: c_int, d: c_int, scale: f32, q_pos0: c_int, k_sb: c_longlong, k_sh: c_longlong, k_ss: c_longlong, v_sb: c_longlong, v_sh: c_longlong, v_ss: c_longlong, cache_kind: c_int, scores: F, stats: F, partials: F, s: S, pos: CI) Error!void {
    if (d < 1 or d > 256 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0 or batch < 1 or qlen < 1 or span < 1 or q_pos0 < 0) {
        return invalid("causal attention");
    }
    if (cache_kind < 0 or cache_kind > 2) return invalid("causal attention cache kind");
    const kind: usize = @intCast(cache_kind);
    const visible: c_int = @min(q_pos0 + 1, span);
    if (scores != null) {
        if (qlen != 1 or stats == null or partials == null) return invalid("the decode walk");
        // With a device position the keys in use are read on the device and the grid covers the whole cache.
        const tiles: c_int = @intCast(if (pos != null) cdiv(span, 128) else cdiv(visible, 128));
        const head_grid = dim(heads, batch, 1);
        var a: Args = .{};
        try a.add(ad(q));
        try a.add(ad(k));
        try a.add(ad(scores));
        try a.add(span);
        try a.add(heads);
        try a.add(kv_heads);
        try a.add(d);
        try a.add(scale);
        try a.add(visible);
        try a.add(k_sb);
        try a.add(k_sh);
        try a.add(k_ss);
        try a.add(ad(pos));
        // a warp scores a key: four warps a block
        try l.go(l.score_keys[kind], dim(cdiv(span, 4), heads, batch), dim(128, 1, 1), 0, s, &a);
        var b: Args = .{};
        try b.add(ad(scores));
        try b.add(ad(stats));
        try b.add(span);
        try b.add(visible);
        try b.add(ad(pos));
        try l.go(l.softmax_stats, head_grid, dim(256, 1, 1), 0, s, &b);
        var c: Args = .{};
        try c.add(ad(scores));
        try c.add(ad(stats));
        try c.add(ad(v));
        try c.add(ad(partials));
        try c.add(span);
        try c.add(heads);
        try c.add(kv_heads);
        try c.add(d);
        try c.add(visible);
        try c.add(v_sb);
        try c.add(v_sh);
        try c.add(v_ss);
        try c.add(ad(pos));
        try l.go(l.apply_values[kind], dim(tiles, heads, batch), dim(256, 1, 1), 0, s, &c);
        var e: Args = .{};
        try e.add(ad(partials));
        try e.add(ad(out));
        try e.add(tiles);
        try e.add(d);
        try l.go(l.sum_partials, head_grid, dim(256, 1, 1), 0, s, &e);
        return;
    }
    var a: Args = .{};
    try a.add(ad(q));
    try a.add(ad(k));
    try a.add(ad(v));
    try a.add(ad(out));
    try a.add(qlen);
    try a.add(span);
    try a.add(heads);
    try a.add(kv_heads);
    try a.add(d);
    try a.add(scale);
    try a.add(q_pos0);
    try a.add(k_sb);
    try a.add(k_sh);
    try a.add(k_ss);
    try a.add(v_sb);
    try a.add(v_sh);
    try a.add(v_ss);
    // a 16-bit cache of whole 64-wide heads takes the 64-row tile (attention=f32: 16-row); odd heads a wave a query
    if (kind < 2 and @rem(d, 64) == 0 and d <= 256 and @rem(k_ss, 8) == 0 and @rem(v_ss, 8) == 0 and l.wide) {
        try l.go(l.fa_wide[kind], dim(cdiv(qlen, 64), heads, batch), dim(256, 1, 1), 0, s, &a);
    } else if (d >= 2 and @rem(d, 2) == 0) {
        try l.go(l.fa_prefill[kind], dim(cdiv(qlen, 16), heads, batch), dim(256, 1, 1), 0, s, &a);
    } else {
        try l.go(l.causal[kind], dim(qlen, heads, batch), dim(32, 1, 1), 0, s, &a);
    }
}

/// The 64-row prefill tile over a stream's pages: q (heads, qlen, d) fp32 against `span` positions, 16-bit cache.
pub fn pagedCausal(l: *const Launcher, q: u64, k: u64, v: u64, table: u64, out: u64, qlen: usize, span: usize, heads: usize, kv_heads: usize, d: usize, scale: f32, q_pos0: usize, count: usize, cache_kind: c_int, s: S) Error!void {
    if (d % 64 != 0 or d > 256 or heads % kv_heads != 0 or cache_kind < 0 or cache_kind > 1) return invalid("paged attention");
    var a: Args = .{};
    try a.add(q);
    try a.add(k);
    try a.add(v);
    try a.add(out);
    try a.add(@as(c_int, @intCast(qlen)));
    try a.add(@as(c_int, @intCast(span)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(scale);
    try a.add(@as(c_int, @intCast(q_pos0)));
    try a.add(table);
    try a.add(@as(c_int, @intCast(count)));
    try l.go(l.fa_paged[@intCast(cache_kind)], dim(cdiv(qlen, 64), heads, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_attn_gate(l: *const Launcher, att: CF, qg: C, out: P, kind: c_int, len: c_int, heads: c_int, d: c_int, rows_major: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(att));
    try a.add(ad(qg));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(len);
    try a.add(heads);
    try a.add(d);
    try a.add(rows_major);
    try l.flat(l.op.attn_gate, @as(i64, len) * heads * d, s, &a);
}

pub fn tf_kv_write(l: *const Launcher, src: C, cache: P, kind: c_int, len: c_int, kv_heads: c_int, d: c_int, total: c_int, pos0: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(src));
    try a.add(ad(cache));
    try a.add(kind);
    try a.add(len);
    try a.add(kv_heads);
    try a.add(d);
    try a.add(total);
    try a.add(pos0);
    try l.flat(l.op.kv_write, @as(i64, len) * kv_heads * d, s, &a);
}

/// kv_write with the first slot read from the device (`pos`, one i32), so a captured graph replays at any position.
pub fn tf_kv_write_at(l: *const Launcher, src: C, cache: P, kind: c_int, len: c_int, kv_heads: c_int, d: c_int, total: c_int, pos: CI, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(src));
    try a.add(ad(cache));
    try a.add(kind);
    try a.add(len);
    try a.add(kv_heads);
    try a.add(d);
    try a.add(total);
    try a.add(ad(pos));
    try l.flat(l.op.kv_write_at, @as(i64, len) * kv_heads * d, s, &a);
}
