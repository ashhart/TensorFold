//! A lane round's launches over its device plan: cache writes, the attention walk, the recurrence and the keep.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

/// plan.hpp's PlanArgs: device addresses of the round's per-row and per-slot arrays (all zero: no plan).
pub const PlanArgs = extern struct {
    pos: u64 = 0,
    slot: u64 = 0,
    first: u64 = 0,
    count: u64 = 0,
    desc: u64 = 0,
    snaps: u64 = 0,
    pages: u32 = 0,
    pool: u32 = 0,
};

/// A plan's shape: `rows` rows (a bucket, padding included) over `slots` slots.
pub const PlanRef = struct { args: PlanArgs, rows: usize, slots: usize };

/// Rows of keys (fp32, rotated) and values (fp16 or bf16) into each row's slot at its position.
pub fn planKvWrite(l: *const Launcher, keys: u64, values: u64, kind: c_int, p: PlanRef, layer: usize, kv_heads: usize, d: usize, s: S) Error!void {
    var a: Args = .{};
    try a.add(keys);
    try a.add(values);
    try a.add(kind);
    try a.add(@as(c_int, @intCast(p.rows)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(p.args);
    try a.add(@as(c_int, @intCast(layer)));
    try l.flat(l.plan.kv_write, @intCast(p.rows * kv_heads * d), s, &a);
}

/// `planKvWrite` with the keys in the activation kind: the cache bytes of the fp32 keys a cast gives.
pub fn planKvWriteAct(
    l: *const Launcher,
    keys: u64,
    values: u64,
    kind: c_int,
    p: PlanRef,
    layer: usize,
    kv_heads: usize,
    d: usize,
    s: S,
) Error!void {
    if (kind < 1 or kind > 2) return invalid("planned keys");
    var a: Args = .{};
    try a.add(keys);
    try a.add(values);
    try a.add(kind);
    try a.add(@as(c_int, @intCast(p.rows)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(p.args);
    try a.add(@as(c_int, @intCast(layer)));
    try l.flat(l.plan.kv_write_act, @intCast(p.rows * kv_heads * d), s, &a);
}

/// Each row's `hidden` 16-bit residual words, as tap `tap` of `taps`, into its slot's tap ring (`tap_rows` rows).
pub fn planTap(l: *const Launcher, x: u64, p: PlanRef, hidden: usize, tap: usize, taps: usize, tap_rows: usize, s: S) Error!void {
    if (hidden % 2 != 0 or tap >= taps) return invalid("planned tap");
    var a: Args = .{};
    try a.add(x);
    try a.add(@as(c_int, @intCast(p.rows)));
    try a.add(@as(c_int, @intCast(hidden)));
    try a.add(@as(c_int, @intCast(tap)));
    try a.add(@as(c_int, @intCast(taps)));
    try a.add(@as(c_int, @intCast(tap_rows)));
    try a.add(p.args);
    try l.flat(l.plan.tap, @intCast(p.rows * hidden / 2), s, &a);
}

/// `planTap` for `rows` prompt rows of one stream at positions `pos0` on, into `ring`.
pub fn tapRows(l: *const Launcher, x: u64, ring: u64, rows: usize, hidden: usize, tap: usize, taps: usize, tap_rows: usize, pos0: usize, s: S) Error!void {
    if (hidden % 2 != 0 or tap >= taps) return invalid("prompt tap");
    var a: Args = .{};
    try a.add(x);
    try a.add(ring);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(hidden)));
    try a.add(@as(c_int, @intCast(tap)));
    try a.add(@as(c_int, @intCast(taps)));
    try a.add(@as(c_int, @intCast(tap_rows)));
    try a.add(@as(c_longlong, @intCast(pos0)));
    try l.flat(l.plan.tap_rows, @intCast(rows * hidden / 2), s, &a);
}

/// Keys a warp scores from score_span keys on, and below it (plan.hip's kScoreKeys and kScoreKeysShort).
const score_keys = 4;
const score_keys_short = 2;
const score_span = 16384;
/// Steps of its warps' keys a tf_plan_score256_* block walks (plan.hip's kScoreSteps).
const score_steps = 8;
/// Rows of one slot a tf_plan_rows_score256_* block scores from one read of their keys (plan_rows.hip's kRows).
pub const score_rows = 2;

/// Which blocks score a walk's rows: a block a row, or a slot's rows in chunks of `score_rows` a block.
pub const Rows = enum { one, shared };

/// One query a row (rows, heads, d) fp32 over its slot's caches up to its position; the walk covers `span` keys.
pub fn planCausal(l: *const Launcher, q: u64, out: u64, scores: u64, stats: u64, partials: u64, p: PlanRef, layer: usize, heads: usize, kv_heads: usize, d: usize, span: usize, scale: f32, kind: c_int, s: S) Error!void {
    return planCausalAs(l, q, out, scores, stats, partials, p, layer, heads, kv_heads, d, span, scale, kind, false, s);
}

/// `planCausal` with q and out in the cache's activation kind when `act`: the bytes of the casts around it.
pub fn planCausalAs(
    l: *const Launcher,
    q: u64,
    out: u64,
    scores: u64,
    stats: u64,
    partials: u64,
    p: PlanRef,
    layer: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    span: usize,
    scale: f32,
    kind: c_int,
    act: bool,
    s: S,
) Error!void {
    try planScore(l, q, scores, p, layer, heads, kv_heads, d, span, scale, kind, act, roundRows(p.rows), s);
    const which: usize = @intCast(kind - 1);
    const rows = p.rows;
    const tiles: c_int = @intCast(cdiv(span, 128));
    const span_c: c_int = @intCast(span);
    const heads_c: c_int = @intCast(heads);
    const kv_c: c_int = @intCast(kv_heads);
    const d_c: c_int = @intCast(d);
    const layer_c: c_int = @intCast(layer);
    var b: Args = .{};
    try b.add(scores);
    try b.add(stats);
    try b.add(span_c);
    try b.add(@as(c_int, 0));
    try b.add(p.args.pos);
    try l.go(l.softmax_stats, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &b);
    var c: Args = .{};
    try c.add(scores);
    try c.add(stats);
    try c.add(partials);
    try c.add(span_c);
    try c.add(heads_c);
    try c.add(kv_c);
    try c.add(d_c);
    try c.add(p.args);
    try c.add(layer_c);
    const apply = @divExact(heads, applyGroup(scoreGroup(heads, kv_heads), @intCast(tiles), heads, rows));
    try l.go(l.plan.apply[which], dim(tiles, apply, rows), dim(256, 1, 1), 0, s, &c);
    var e: Args = .{};
    try e.add(partials);
    try e.add(out);
    try e.add(tiles);
    try e.add(d_c);
    if (act) try e.add(kind);
    try l.go(if (act) l.plan.sum_act else l.sum_partials, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &e);
}

/// The score blocks of a walk's launch: one-row launches keep a block a row, wider ones share a slot's key loads.
fn roundRows(rows: usize) Rows {
    return if (rows > 1) .shared else .one;
}

/// Query heads a score block takes, of one KV head (at most plan.hip's kMaxGroup): the cache is read once for them.
fn scoreGroup(heads: usize, kv_heads: usize) usize {
    var group = @divExact(heads, kv_heads);
    while (group > 16) group = @divExact(group, smallestFactor(group));
    return group;
}

/// Rows a score block works: a chunk of its slot's when `rows` shares them at d = 256 on long spans with q fp32.
fn blockRows(rows: Rows, d: usize, span: usize, act: bool) usize {
    return if (rows == .shared and d == 256 and span >= score_span and !act) score_rows else 1;
}

/// planCausalAs's score step: each row's scores (rows, heads, span) fp32 against its slot's keys, -inf past its own.
pub fn planScore(
    l: *const Launcher,
    q: u64,
    scores: u64,
    p: PlanRef,
    layer: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    span: usize,
    scale: f32,
    kind: c_int,
    act: bool,
    rows: Rows,
    s: S,
) Error!void {
    if (d > 256 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0 or kind < 1 or kind > 2) return invalid("planned attention");
    const which: usize = @intCast(kind - 1);
    const group = scoreGroup(heads, kv_heads);
    var a: Args = .{};
    try a.add(q);
    try a.add(scores);
    try a.add(@as(c_int, @intCast(span)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(scale);
    try a.add(p.args);
    try a.add(@as(c_int, @intCast(layer)));
    // four warps a block score their keys for the block's heads; short spans take fewer keys a warp to fill the card
    const long = span >= score_span;
    const keys: usize = if (long) score_keys else score_keys_short;
    // d = 256 (the Qwen checkpoints) on long spans takes its own kernels, a block walking score_steps steps
    const wide = long and d == 256;
    const steps: usize = if (wide) score_steps else 1;
    const grid = dim(cdiv(span, 4 * keys * steps), @divExact(heads, group), p.rows);
    const lds: u32 = @intCast(group * d * 4);
    const chunk = blockRows(rows, d, span, act);
    if (chunk > 1) {
        // the rows of one-row slots at the one-row kernel's occupancy, then the chunks of the others, both on this grid
        try l.go(l.plan.score256_lone[which], grid, dim(128, 1, 1), lds, s, &a);
        return l.go(l.plan.score256_rows[which], grid, dim(128, 1, 1), @intCast(chunk * lds), s, &a);
    }
    const fns = if (wide) (if (act) l.plan.score256_act else l.plan.score256) else if (act) (if (long) l.plan.score_act else l.plan.score2_act) else if (long) l.plan.score else l.plan.score2;
    try l.go(fns[which], grid, dim(128, 1, 1), lds, s, &a);
}

/// Keys the one-block walk (`planAttend`) takes at most.
pub const attend_span = 4096;

/// `planCausalAs` as one launch a (row, query head), q and out fp32 (kind 0) or the cache's kind; false past its shape.
pub fn planAttend(
    l: *const Launcher,
    q: u64,
    q_kind: c_int,
    out: u64,
    out_kind: c_int,
    p: PlanRef,
    layer: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    span: usize,
    scale: f32,
    kind: c_int,
    s: S,
) Error!bool {
    if (span > attend_span or d > 128 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0) return false;
    if (kind < 1 or kind > 2) return false;
    var a: Args = .{};
    try a.add(q);
    try a.add(q_kind);
    try a.add(out);
    try a.add(out_kind);
    try a.add(@as(c_int, @intCast(span)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(scale);
    try a.add(p.args);
    try a.add(@as(c_int, @intCast(layer)));
    try l.go(l.plan.attend[@intCast(kind - 1)], dim(heads, p.rows, 1), dim(1024, 1, 1), 0, s, &a);
    return true;
}

/// Blocks under which the values' walk splits a KV head's query heads over more blocks.
const apply_blocks = 128;

/// Query heads an apply block takes: fewer while the grid is small (a head's sums do not depend on its block).
fn applyGroup(group: usize, tiles: usize, heads: usize, rows: usize) usize {
    var g = group;
    while (g > 1 and tiles * @divExact(heads, g) * rows < apply_blocks) g = @divExact(g, smallestFactor(g));
    return g;
}

fn smallestFactor(n: usize) usize {
    var f: usize = 2;
    while (@rem(n, f) != 0) f += 1;
    return f;
}

/// The DeltaNet recurrence of every slot's rows (q, k, v, gate, beta, y flat over rows), state in the slot's caches.
pub fn planGatedDelta(l: *const Launcher, q: u64, k: u64, v: u64, gate: u64, beta: u64, y: u64, p: PlanRef, layer: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize, s: S) Error!void {
    if ((dk != 16 and dk != 128) or @rem(dv, 8) != 0 or key_heads < 1 or @rem(value_heads, key_heads) != 0) return invalid("planned gated delta");
    var a: Args = .{};
    try a.add(q);
    try a.add(k);
    try a.add(v);
    try a.add(gate);
    try a.add(beta);
    try a.add(y);
    try a.add(@as(c_int, @intCast(key_heads)));
    try a.add(@as(c_int, @intCast(value_heads)));
    try a.add(@as(c_int, @intCast(dv)));
    try a.add(p.args);
    try a.add(@as(c_int, @intCast(layer)));
    const wide: usize = if (dk == 128) 0 else 1;
    try l.go(l.plan.gdn[wide], dim(@divExact(dv, 8), value_heads, p.slots), dim(32, 8, 1), 0, s, &a);
}

/// Row r of `dst` (`words` words each) from the address in `srcs[r]` (device u64s), for `rows` rows.
pub fn planGather(l: *const Launcher, srcs: u64, dst: u64, words: usize, rows: usize, s: S) Error!void {
    if (words % 4 != 0 or rows == 0) return invalid("planned gather");
    var a: Args = .{};
    try a.add(srcs);
    try a.add(dst);
    try a.add(@as(c_int, @intCast(words)));
    try l.go(l.plan.gather, dim(cdiv(words, 256), rows, 1), dim(256, 1, 1), 0, s, &a);
}

/// Writes positions pos0 .. pos0 + len of a stream's pages from 16-bit `src` at h * s_head + r * s_row + j.
pub fn pagesWrite(l: *const Launcher, src: u64, pool: u64, table: u64, len: usize, kv_heads: usize, d: usize, s_head: usize, s_row: usize, pos0: usize, count: usize, s: S) Error!void {
    var a: Args = .{};
    try a.add(src);
    try a.add(pool);
    try a.add(table);
    try a.add(@as(c_int, @intCast(len)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(@as(c_longlong, @intCast(s_head)));
    try a.add(@as(c_longlong, @intCast(s_row)));
    try a.add(@as(c_int, @intCast(pos0)));
    try a.add(@as(c_int, @intCast(count)));
    try l.flat(l.plan.page_write, @intCast(len * kv_heads * d), s, &a);
}

/// The first `len` positions of a stream's pages as (kv_heads, len, d) 16-bit values in `dst`.
pub fn pagesGather(l: *const Launcher, pool: u64, table: u64, dst: u64, len: usize, kv_heads: usize, d: usize, count: usize, s: S) Error!void {
    var a: Args = .{};
    try a.add(pool);
    try a.add(table);
    try a.add(dst);
    try a.add(@as(c_int, @intCast(len)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(@as(c_int, @intCast(count)));
    try l.flat(l.plan.page_gather, @intCast(len * kv_heads * d), s, &a);
}

/// What a keep reads: the plan, kept rows a slot (negative: unlisted), final rows, word counts (of four), layers.
pub const Keep = struct { keep: u64, hidden: u64, hidden_words: usize, channels: usize, taps: usize, layers: usize };

/// Keeps the listed slots' rows: conv windows from the round's first window and the kept rows' inputs, final rows.
pub fn planKeep(l: *const Launcher, p: PlanRef, k: Keep, s: S) Error!void {
    if (k.hidden_words % 4 != 0) return invalid("planned keep");
    var a: Args = .{};
    try a.add(p.args);
    try a.add(k.keep);
    try a.add(k.hidden);
    try a.add(@as(c_int, @intCast(k.hidden_words)));
    try a.add(@as(c_int, @intCast(k.channels)));
    try a.add(@as(c_int, @intCast(k.taps)));
    try a.add(@as(c_int, @intCast(k.layers)));
    try l.go(l.plan.keep, dim(64, k.layers + 1, p.slots), dim(256, 1, 1), 0, s, &a);
}

/// The kept slots' DeltaNet states run through their kept rows, every linear layer in one launch.
pub fn planGdnReplay(l: *const Launcher, keep: u64, p: PlanRef, key_heads: usize, value_heads: usize, dk: usize, dv: usize, layers: usize, s: S) Error!void {
    if ((dk != 16 and dk != 128) or @rem(dv, 8) != 0 or key_heads < 1 or @rem(value_heads, key_heads) != 0) return invalid("planned replay");
    var a: Args = .{};
    try a.add(keep);
    try a.add(@as(c_int, @intCast(p.rows)));
    try a.add(@as(c_int, @intCast(key_heads)));
    try a.add(@as(c_int, @intCast(value_heads)));
    try a.add(@as(c_int, @intCast(dv)));
    try a.add(p.args);
    try a.add(@as(c_int, @intCast(p.slots)));
    const wide: usize = if (dk == 128) 0 else 1;
    try l.go(l.plan.gdn_replay[wide], dim(@divExact(dv, 8), value_heads, p.slots * layers), dim(32, 8, 1), 0, s, &a);
}

test "a slot's rows share the score's key loads only past one row, at d 256 on long spans with q fp32" {
    const std = @import("std");
    try std.testing.expectEqual(Rows.one, roundRows(1));
    try std.testing.expectEqual(Rows.shared, roundRows(2));
    try std.testing.expectEqual(@as(usize, score_rows), blockRows(roundRows(4), 256, score_span, false));
    try std.testing.expectEqual(@as(usize, 1), blockRows(roundRows(1), 256, 1 << 18, false));
    try std.testing.expectEqual(@as(usize, 1), blockRows(.shared, 256, score_span - 1, false));
    try std.testing.expectEqual(@as(usize, 1), blockRows(.shared, 128, 1 << 18, false));
    try std.testing.expectEqual(@as(usize, 1), blockRows(.shared, 256, 1 << 18, true));
    try std.testing.expectEqual(@as(usize, 8), scoreGroup(16, 2));
    try std.testing.expectEqual(@as(usize, 16), scoreGroup(64, 2));
}

test "a small walk splits the heads of a KV head, a wide one keeps them together" {
    const std = @import("std");
    try std.testing.expectEqual(@as(usize, 1), applyGroup(16, 1, 32, 1));
    try std.testing.expectEqual(@as(usize, 4), applyGroup(16, 2, 32, 8));
    try std.testing.expectEqual(@as(usize, 16), applyGroup(16, 256, 32, 1));
    try std.testing.expectEqual(@as(usize, 3), applyGroup(6, 16, 24, 1));
}
