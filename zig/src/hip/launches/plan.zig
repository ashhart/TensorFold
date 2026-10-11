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

/// One query a row (rows, heads, d) fp32 over its slot's caches up to its position; the walk covers `span` keys.
pub fn planCausal(l: *const Launcher, q: u64, out: u64, scores: u64, stats: u64, partials: u64, p: PlanRef, layer: usize, heads: usize, kv_heads: usize, d: usize, span: usize, scale: f32, kind: c_int, s: S) Error!void {
    if (d > 256 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0 or kind < 1 or kind > 2) return invalid("planned attention");
    const which: usize = @intCast(kind - 1);
    const rows = p.rows;
    const tiles: c_int = @intCast(cdiv(span, 128));
    const span_c: c_int = @intCast(span);
    const heads_c: c_int = @intCast(heads);
    const kv_c: c_int = @intCast(kv_heads);
    const d_c: c_int = @intCast(d);
    const layer_c: c_int = @intCast(layer);
    // a block takes `group` query heads of one KV head (at most plan.hip's kMaxGroup): the cache is read once for them
    var group = @divExact(heads, kv_heads);
    while (group > 16) group = @divExact(group, smallestFactor(group));
    const blocks = @divExact(heads, group);
    var a: Args = .{};
    try a.add(q);
    try a.add(scores);
    try a.add(span_c);
    try a.add(heads_c);
    try a.add(kv_c);
    try a.add(d_c);
    try a.add(scale);
    try a.add(p.args);
    try a.add(layer_c);
    // a warp scores a key for the block's heads: four warps a block, the heads' queries in shared memory
    try l.go(l.plan.score[which], dim(cdiv(span, 4), blocks, rows), dim(128, 1, 1), @intCast(group * d * 4), s, &a);
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
    try l.go(l.plan.apply[which], dim(tiles, blocks, rows), dim(256, 1, 1), 0, s, &c);
    var e: Args = .{};
    try e.add(partials);
    try e.add(out);
    try e.add(tiles);
    try e.add(d_c);
    try l.go(l.sum_partials, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &e);
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
