//! The MTP drafter's shared rounds: every stream's absorbs and chained drafts in passes of up to 16 rows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const mtp = @import("cuda_mtp.zig");

const Mtp = mtp.Mtp;
const Ctx = mtp.Ctx;
const dr = lanes.drafter;
const head_dim = geometry.head_dim;

/// Rows a pass at most: the drafter's verify-tile and routed-chunk width, so every row keeps its solo bits.
pub const most = 16;

/// tf_q36m_attn_rows' RingRow: a packed row's ring, its slots [begin, end) of a ring of w, its segment's first row.
pub const RingRow = extern struct { rk: u64, rv: u64, begin: i64, end: i64, seg: i32, w: i32 };

/// A stream's rows in a pass: its context, its first packed row, its rows, and their first slot.
const Seg = struct { ctx: *Ctx, first: usize, rows: usize, p0: u64 };

/// Consecutive items into passes of at most `most` rows (out[i]: its items [from, to)); an item over `most` alone.
pub fn plan(rows: []const usize, out: [][2]usize) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < rows.len) {
        var j = i;
        var sum: usize = 0;
        while (j < rows.len and rows[j] <= most and sum + rows[j] <= most) : (j += 1) sum += rows[j];
        if (j == i) j = i + 1;
        out[n] = .{ i, j };
        n += 1;
        i = j;
    }
    return n;
}

/// A row's attention over its context's ring: slots [begin, end), then its segment's rows from `seg`.
fn ringRow(d: *Mtp, ctx: *Ctx, end: u64, seg: usize) RingRow {
    return .{ .rk = d.ringK(ctx), .rv = d.ringV(ctx), .begin = @intCast(ctx.begin), .end = @intCast(end), .seg = @intCast(seg), .w = @intCast(d.capacity) };
}

/// The pass's tokens and ring table up from the batch's own staging (the last pass's fetch has waited on it).
fn upload(d: *Mtp, ids: []const u32, tab: []const RingRow) !void {
    const e = d.e;
    const st = d.staged.bytes;
    @memcpy(std.mem.bytesAsSlice(u32, st[Mtp.bids_at..][0 .. ids.len * 4]), ids);
    @memcpy(st[Mtp.btab_at..][0 .. tab.len * @sizeOf(RingRow)], std.mem.sliceAsBytes(tab));
    try cuda.copy_source.enqueue(e.d, d.ids, st[Mtp.bids_at..][0 .. ids.len * 4], e.stream);
    try cuda.copy_source.enqueue(e.d, d.tab, st[Mtp.btab_at..][0 .. tab.len * @sizeOf(RingRow)], e.stream);
}

/// Mtp.layer over every segment's rows at once: d.ids their tokens, d.hin their states; x gets their normed outputs.
fn layerRows(d: *Mtp, segs: []const Seg, rows: usize) !void {
    const e = d.e;
    const g = e.g;
    const w = &d.w;
    const aw = g.attnWidth();
    const kw = d.kvWidth();
    try e.embed(d.ids, d.emb, rows, 0);
    try d.norm(d.emb, 0, w.norm_e, d.en, rows);
    try d.norm(d.hin, 0, w.norm_h, d.hn, rows);
    try d.proj(.{ .a4 = w.fc_e }, d.en, d.h, rows);
    try d.proj(.{ .a4 = w.fc_h }, d.hn, d.r, rows);
    try d.norm(d.h, d.r, w.input_norm, d.x, rows);
    try d.projQkv(rows);
    for (segs) |s| { // rope positions run from each segment's first slot
        const p0: c_longlong = @intCast(s.p0);
        try e.run(d.fns.norm_rope, .{ s.rows * g.q_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.qraw + s.first * 2 * aw * 2, w.q_norm, d.q + s.first * aw * 2, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), p0, @as(c_int, 1), g.eps, geometry.theta });
        try e.run(d.fns.norm_rope, .{ s.rows * g.kv_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.kraw + s.first * kw * 2, w.k_norm, d.kb + s.first * kw * 2, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(kw)), @as(c_int, head_dim), p0, @as(c_int, 1), g.eps, geometry.theta });
    }
    try e.run(d.fns.attn_rows, .{ rows, g.q_heads, 1 }, .{ 32, 1, 1 }, .{ d.q, d.qraw, @as(c_int, @intCast(2 * aw)), d.kb, d.vb, d.tab, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(g.kv_heads)), d.att });
    try d.proj(w.o, d.att, d.r, rows);
    try d.norm(d.h, d.r, w.post_norm, d.x, rows);
    try d.routed(w.mlp.moe.router, w.mlp.moe.experts, d.x, d.r, rows);
    try d.norm(d.h, d.r, w.norm, d.x, rows);
}

/// Each segment's keys and values from the pass into its context's ring, at its slots.
fn ringPut(d: *Mtp, segs: []const Seg) !void {
    const e = d.e;
    const kw = d.kvWidth();
    for (segs) |s| try e.run(d.ring_fns.ring_put, .{ s.rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kb + s.first * kw * 2, d.vb + s.first * kw * 2, d.ringK(s.ctx), d.ringV(s.ctx), @as(c_int, @intCast(kw)), @as(c_longlong, @intCast(s.p0)), @as(c_int, @intCast(d.capacity)) });
}

/// Rows of `input` through the target's head, and each row's greedy token: one projection, one argmax launch.
fn head(d: *Mtp, input: u64, rows: usize) !void {
    const e = d.e;
    try d.proj(e.head, input, d.logits, rows);
    try e.run(e.fns.argmax, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ d.logits, @as(c_int, @intCast(e.g.vocab)), d.tok });
}

/// The first `rows` drawn tokens, read back with one synchronization.
fn fetchRows(d: *Mtp, rows: usize) ![]align(1) const u32 {
    const e = d.e;
    const st = d.staged.bytes;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + Mtp.tok_at, d.tok, rows * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    return std.mem.bytesAsSlice(u32, st[Mtp.tok_at..][0 .. rows * 4]);
}

/// The round's absorbs: up to `most` rows a pass, each stream's rows one segment; larger or empty ones as before.
pub fn absorb(d: *Mtp, items: []const dr.Absorb) !void {
    const sizes = try d.gpa.alloc(usize, items.len);
    defer d.gpa.free(sizes);
    for (items, sizes) |it, *s| {
        const n = if (it.rows) |rs| rs.len else it.features.rows;
        s.* = if (n == 0) most + 1 else n; // an empty absorb keeps the serial path's handling
    }
    const passes = try d.gpa.alloc([2]usize, items.len);
    defer d.gpa.free(passes);
    for (passes[0..plan(sizes, passes)]) |p| {
        if (sizes[p[0]] > most) {
            try d.absorbOne(items[p[0]]);
            continue;
        }
        try absorbPass(d, items[p[0]..p[1]]);
    }
}

fn absorbPass(d: *Mtp, items: []const dr.Absorb) !void {
    const e = d.e;
    const g = e.g;
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = .live; // up to `most` rows: prompt and live rows take the same arithmetic
    var segs: [most]Seg = undefined;
    var tab: [most]RingRow = undefined;
    var ids: [most]u32 = undefined;
    var at: usize = 0;
    for (items, 0..) |it, si| {
        const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
        if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
        const n = if (it.rows) |rs| rs.len else it.features.rows;
        const start = it.features.offset;
        if (it.follow.len < n) return error.BadDraftRows;
        // absorbRole's bookkeeping; `most` rows never reach the ring's trim
        if (start < ctx.end) ctx.end = start;
        if (start > ctx.end) ctx.begin = start;
        ctx.end = start;
        const out = d.taps_in + at * g.hidden * 2;
        if (it.rows) |rs| {
            var tr: mtp.TapRows = .{ .n = @intCast(n), .rows = @splat(0) };
            for (rs, 0..) |r, i| tr.rows[i] = @intCast(r);
            try e.run(d.ring_fns.gather_rows, .{ n, 1, 1 }, .{ 256, 1, 1 }, .{ it.features.buffer, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), tr, @as(c_int, @intCast(g.hidden)), out });
        } else try e.run(d.ring_fns.gather_taps, .{ n, 1, 1 }, .{ 256, 1, 1 }, .{ it.features.buffer, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(g.hidden)), out });
        for (0..n) |r| {
            tab[at + r] = ringRow(d, ctx, start, at);
            ids[at + r] = it.follow[r];
        }
        segs[si] = .{ .ctx = ctx, .first = at, .rows = n, .p0 = start };
        at += n;
    }
    const s = segs[0..items.len];
    try d.norm(d.taps_in, 0, e.final_norm, d.hin, at); // the target's final norm: the rows the head read
    try upload(d, ids[0..at], tab[0..at]);
    try layerRows(d, s, at);
    try ringPut(d, s);
    for (s, 0..) |seg, j| { // each stream's newest row: its state kept for the drafts, its row of the head's input
        const last = d.x + (seg.first + seg.rows - 1) * g.hidden * 2;
        try seg.ctx.last.copyFrom(0, last, g.hidden * 2, e.stream.handle);
        try d.scratch.copyFrom(d.hn - d.scratch.ptr + j * g.hidden * 2, last, g.hidden * 2, e.stream.handle);
        seg.ctx.end = seg.p0 + seg.rows;
    }
    try head(d, d.hn, s.len);
    const toks = try fetchRows(d, s.len);
    for (s, toks) |seg, t| seg.ctx.cands[0] = t;
    d.batch_passes += 1;
}

/// A chain's stream and the drafts it asks for.
const Chain = struct { ctx: *Ctx, pos0: u64, depth: usize };

/// The round's holds: each draft level's chains in passes of up to `most` streams; trees keep the serial path.
pub fn hold(d: *Mtp, items: []const dr.Hold) !void {
    const chains = try d.gpa.alloc(Chain, items.len);
    defer d.gpa.free(chains);
    var n: usize = 0;
    var deepest: usize = 0;
    for (items) |it| {
        const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
        if (it.lanes != null) {
            try d.hold(ctx, it.pending, it.position - 1, it.depth, it.lanes);
            continue;
        }
        const pos0 = it.position - 1;
        if (ctx.end != pos0) return error.DraftContextBehind;
        ctx.held[0] = ctx.cands[0];
        ctx.held_n = 1;
        chains[n] = .{ .ctx = ctx, .pos0 = pos0, .depth = @min(it.depth, ctx.held.len) };
        deepest = @max(deepest, chains[n].depth);
        n += 1;
    }
    var k: usize = 1;
    while (k < deepest) : (k += 1) {
        var group: [most]Chain = undefined;
        var rows: usize = 0;
        for (chains[0..n]) |c| if (c.depth > k) {
            group[rows] = c;
            rows += 1;
            if (rows == most) {
                try holdPass(d, group[0..rows], k);
                rows = 0;
            }
        };
        if (rows > 0) try holdPass(d, group[0..rows], k);
    }
}

/// Draft level k of every chain in `group`, a row each: its previous state and draft at slot pos0 + k - 1.
fn holdPass(d: *Mtp, group: []const Chain, k: usize) !void {
    const e = d.e;
    const g = e.g;
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = .proposal;
    var segs: [most]Seg = undefined;
    var tab: [most]RingRow = undefined;
    var ids: [most]u32 = undefined;
    for (group, 0..) |c, r| {
        const src = if (k == 1) c.ctx.last.ptr else c.ctx.step.ptr;
        try d.scratch.copyFrom(d.hin - d.scratch.ptr + r * g.hidden * 2, src, g.hidden * 2, e.stream.handle);
        const p0 = c.pos0 + k - 1;
        ids[r] = c.ctx.held[k - 1];
        tab[r] = ringRow(d, c.ctx, p0, r);
        segs[r] = .{ .ctx = c.ctx, .first = r, .rows = 1, .p0 = p0 };
    }
    const s = segs[0..group.len];
    try upload(d, ids[0..group.len], tab[0..group.len]);
    try layerRows(d, s, group.len);
    try ringPut(d, s);
    for (s) |seg| try seg.ctx.step.copyFrom(0, d.x + seg.first * g.hidden * 2, g.hidden * 2, e.stream.handle);
    try head(d, d.x, group.len);
    const toks = try fetchRows(d, group.len);
    for (group, toks) |c, t| {
        c.ctx.held[k] = t;
        c.ctx.held_n = k + 1;
    }
    d.batch_passes += 1;
}

test "passes pack consecutive items up to 16 rows; a longer item runs alone" {
    var out: [8][2]usize = undefined;
    const n = plan(&.{ 3, 4, 2, 8 }, &out);
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 3 }, .{ 3, 4 } }, out[0..n]);
    const m = plan(&.{ 2, 17, 1 }, &out);
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 } }, out[0..m]);
    var ones: [17]usize = @splat(1);
    var wide: [17][2]usize = undefined;
    const w = plan(&ones, &wide);
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 16 }, .{ 16, 17 } }, wide[0..w]);
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(RingRow));
}
