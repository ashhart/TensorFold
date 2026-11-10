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

pub const Branch = struct { ctx: *Ctx, hidden: u64, token: u32, position: u64, parent: i32 = -1 };
pub const LogitsRow = struct { ids: [16]u32, logits: [16]f32 };

/// A stream's rows in a pass: its context, its first packed row, its rows, and their first slot.
const Seg = struct { ctx: *Ctx, first: usize, rows: usize, p0: u64, step: c_int = 1 };

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
    return layerRowsAt(d, segs, rows, d.ids, d.tab, null);
}

fn layerRowsAt(d: *Mtp, segs: []const Seg, rows: usize, ids: u64, tab: u64, ancestors: ?u64) !void {
    const e = d.e;
    const g = e.g;
    const w = &d.w;
    const aw = g.attnWidth();
    const kw = d.kvWidth();
    try e.embed(ids, d.emb, rows, 0);
    try d.norm(d.emb, 0, w.norm_e, d.en, rows);
    try d.norm(d.hin, 0, w.norm_h, d.hn, rows);
    try d.proj(.{ .a4 = w.fc_e }, d.en, d.h, rows);
    try d.proj(.{ .a4 = w.fc_h }, d.hn, d.r, rows);
    try d.norm(d.h, d.r, w.input_norm, d.x, rows);
    try d.projQkv(rows);
    for (segs) |s| { // rope positions run from each segment's first slot
        const p0: c_longlong = @intCast(s.p0);
        try d.normRope(d.qraw + s.first * 2 * aw * 2, w.q_norm, d.q + s.first * aw * 2, g.q_heads, 2 * aw, 2 * head_dim, p0, s.step, s.rows);
        try d.normRope(d.kraw + s.first * kw * 2, w.k_norm, d.kb + s.first * kw * 2, g.kv_heads, kw, head_dim, p0, s.step, s.rows);
    }
    if (ancestors) |anc| {
        for (segs) |s| try e.run(d.fns.attn_tree, .{ s.rows, g.q_heads, 1 }, .{ 32, 1, 1 }, .{
            d.q + s.first * aw * 2,               d.qraw + s.first * 2 * aw * 2,      @as(c_int, @intCast(2 * aw)),
            d.kb + s.first * kw * 2,              d.vb + s.first * kw * 2,            d.lane_k,
            d.lane_v,                             anc + s.first * Mtp.anc_stride * 4, @as(c_int, Mtp.anc_stride),
            d.ringK(s.ctx),                       d.ringV(s.ctx),                     @as(c_int, @intCast(g.q_heads)),
            @as(c_int, @intCast(g.kv_heads)),     @as(c_int, @intCast(d.capacity)),   @as(c_longlong, @intCast(s.ctx.begin)),
            @as(c_longlong, @intCast(s.ctx.end)), d.att + s.first * aw * 2,
        });
    } else try e.run(d.fns.attn_rows, .{ rows, g.q_heads, 1 }, .{ 32, 1, 1 }, .{ d.q, d.qraw, @as(c_int, @intCast(2 * aw)), d.kb, d.vb, tab, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(g.kv_heads)), d.att });
    try d.proj(w.o, d.att, d.r, rows);
    try d.norm(d.h, d.r, w.post_norm, d.x, rows);
    try d.mlp(d.x, d.r, rows);
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

/// Every tentative prefix is a separate query: siblings read the true past and themselves, never one another.
fn branchMetadata(d: *Mtp, queries: []const Branch, segs: []Seg, tab: []RingRow, ids: []u32) !void {
    if (queries.len > most or segs.len < queries.len or tab.len < queries.len or ids.len < queries.len)
        return error.BadMtpBranches;
    for (queries, 0..) |query, i| {
        if (query.token >= d.e.g.vocab or query.hidden == 0 or query.hidden % 16 != 0 or
            query.ctx.begin > query.ctx.end or query.ctx.end > std.math.maxInt(i64) or
            query.position > std.math.maxInt(i64)) return error.BadMtpBranches;
        segs[i] = .{ .ctx = query.ctx, .first = i, .rows = 1, .p0 = query.position };
        tab[i] = ringRow(d, query.ctx, query.ctx.end, i);
        ids[i] = query.token;
    }
}

/// Up to 64 output states: microbatches share MTP weights, then their head projection and shortlist share a pass.
pub fn branchTopk(d: *Mtp, queries: []const Branch, roots: []const *Ctx, out: []LogitsRow) !void {
    const total = queries.len + roots.len;
    if (total == 0 or total > engine.most_window or out.len != total) return error.BadMtpBranches;
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = .proposal;
    const e = d.e;
    const hb = e.g.hidden * 2;
    errdefer cuda.completion.drain(e.d, e.stream, "a failed MTP branch batch did not drain its staging");
    var all_segs: [engine.most_window]Seg = undefined;
    var all_tab: [engine.most_window]RingRow = undefined;
    var all_ids: [engine.most_window]u32 = undefined;
    var prepared: usize = 0;
    while (prepared < queries.len) {
        const n = @min(most, queries.len - prepared);
        try branchMetadata(d, queries[prepared..][0..n], all_segs[prepared..][0..n], all_tab[prepared..][0..n], all_ids[prepared..][0..n]);
        prepared += n;
    }
    const ids_stage = d.staged.bytes[Mtp.ids_at..][0 .. queries.len * 4];
    const tab_stage = d.staged.bytes[Mtp.btab_at..][0 .. queries.len * @sizeOf(RingRow)];
    @memcpy(ids_stage, std.mem.sliceAsBytes(all_ids[0..queries.len]));
    @memcpy(tab_stage, std.mem.sliceAsBytes(all_tab[0..queries.len]));
    if (queries.len > 0) {
        try cuda.copy_source.enqueue(e.d, d.ids, ids_stage, e.stream);
        try cuda.copy_source.enqueue(e.d, d.tab, tab_stage, e.stream);
    }
    var at: usize = 0;
    while (at < queries.len) {
        const n = @min(most, queries.len - at);
        const group = queries[at..][0..n];
        for (group, 0..) |query, i|
            try d.scratch.copyFrom(d.hin - d.scratch.ptr + i * hb, query.hidden, hb, e.stream.handle);
        try layerRowsAt(d, all_segs[at..][0..n], n, d.ids + at * 4, d.tab + at * @sizeOf(RingRow), null);
        try d.scratch.copyFrom(d.lane_state - d.scratch.ptr + at * hb, d.x, n * hb, e.stream.handle);
        at += n;
    }
    return branchHead(d, queries.len, roots, out);
}

fn branchHead(d: *Mtp, n_queries: usize, roots: []const *Ctx, out: []LogitsRow) !void {
    const e = d.e;
    const hb = e.g.hidden * 2;
    const total = n_queries + roots.len;
    for (roots, 0..) |ctx, i|
        try d.scratch.copyFrom(d.lane_state - d.scratch.ptr + (n_queries + i) * hb, ctx.last.ptr, hb, e.stream.handle);
    try d.proj(e.head, d.lane_state, d.logits, total);
    // The head reads lane_state; the earlier hidden banks are now free for the parallel shortlist's partials.
    const part_bytes = @import("cuda").sample.Draws.topkPartsBytes(e.g.vocab, 16, total);
    if (d.qraw <= d.hin or d.qraw - d.hin < part_bytes) return error.BadMtpBranches;
    try e.draws.topkRows(e.stream, d.logits, e.g.vocab, 16, d.hin, d.words, total);
    const lp_words = 43;
    const st = d.staged.bytes[Mtp.words_at..][0 .. total * lp_words * 4];
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr, d.words, st.len, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    const w = std.mem.bytesAsSlice(u32, st);
    for (out, 0..) |*row, i| for (0..16) |j| {
        row.ids[j] = w[i * lp_words + 2 + j];
        row.logits[j] = @bitCast(w[i * lp_words + 22 + j]);
    };
}

pub fn prefixOrder(queries: []const Branch, order: []usize, depths: []usize) !void {
    if (queries.len > engine.most_window or order.len < queries.len or depths.len < queries.len)
        return error.BadMtpBranches;
    for (queries, 0..) |query, i| {
        if (query.parent < -1 or query.parent >= @as(i32, @intCast(i)) or query.ctx.begin > query.ctx.end or
            query.ctx.end > std.math.maxInt(i64) or query.position > std.math.maxInt(i64)) return error.BadMtpBranches;
        const expected = if (query.parent < 0) query.ctx.end else blk: {
            const p: usize = @intCast(query.parent);
            if (query.ctx != queries[p].ctx) return error.BadMtpBranches;
            break :blk try std.math.add(u64, queries[p].position, 1);
        };
        if (query.position != expected) return error.BadMtpBranches;
        depths[i] = if (query.parent < 0) 0 else depths[@intCast(query.parent)] + 1;
        order[i] = i;
    }
    std.mem.sort(usize, order[0..queries.len], depths, struct {
        fn less(d: []const usize, a: usize, b: usize) bool {
            return d[a] < d[b] or (d[a] == d[b] and a < b);
        }
    }.less);
}

/// Known DFlash tokens let every MTP depth run before one shared vocabulary-head pass.
pub fn prefixTopk(d: *Mtp, queries: []const Branch, roots: []const *Ctx, out: []LogitsRow) !void {
    const total = queries.len + roots.len;
    if (total == 0 or total > engine.most_window or out.len != total) return error.BadMtpBranches;
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = .proposal;
    const e = d.e;
    const hb = e.g.hidden * 2;
    const kb = d.kvWidth() * 2;
    const stride = Mtp.anc_stride;
    var order: [engine.most_window]usize = undefined;
    var depth: [engine.most_window]usize = undefined;
    try prefixOrder(queries, &order, &depth);
    var ids: [engine.most_window]u32 = undefined;
    var ancestors: [engine.most_window * stride]i32 = @splat(0);
    for (order[0..queries.len], 0..) |original, packed_row| {
        const query = queries[original];
        if (query.token >= e.g.vocab) return error.BadMtpBranches;
        ids[packed_row] = query.token;
        var parent = query.parent;
        var n: usize = 0;
        while (parent >= 0) : (parent = queries[@intCast(parent)].parent) {
            ancestors[packed_row * stride + 1 + n] = parent;
            n += 1;
        }
        std.mem.reverse(i32, ancestors[packed_row * stride + 1 ..][0..n]);
        ancestors[packed_row * stride] = @intCast(n);
    }
    const ids_stage = d.staged.bytes[Mtp.ids_at..][0 .. queries.len * 4];
    const anc_stage = d.staged.bytes[Mtp.anc_at..][0 .. queries.len * stride * 4];
    @memcpy(ids_stage, std.mem.sliceAsBytes(ids[0..queries.len]));
    @memcpy(anc_stage, std.mem.sliceAsBytes(ancestors[0 .. queries.len * stride]));
    errdefer cuda.completion.drain(e.d, e.stream, "an MTP prefix pass did not drain its staging");
    if (queries.len > 0) {
        try cuda.copy_source.enqueue(e.d, d.ids, ids_stage, e.stream);
        try cuda.copy_source.enqueue(e.d, d.anc, anc_stage, e.stream);
    }
    var at: usize = 0;
    while (at < queries.len) {
        var end = at + 1;
        while (end < queries.len and end - at < most and depth[order[end]] == depth[order[at]]) : (end += 1) {}
        var segs: [most]Seg = undefined;
        for (order[at..end], 0..) |original, local| {
            const query = queries[original];
            const source = if (query.parent < 0) query.ctx.last.ptr else d.lane_state + @as(usize, @intCast(query.parent)) * hb;
            try d.scratch.copyFrom(d.hin - d.scratch.ptr + local * hb, source, hb, e.stream.handle);
            segs[local] = .{ .ctx = query.ctx, .first = local, .rows = 1, .p0 = query.position, .step = 0 };
        }
        try layerRowsAt(d, segs[0 .. end - at], end - at, d.ids + at * 4, 0, d.anc + at * stride * 4);
        for (order[at..end], 0..) |original, local| {
            try d.scratch.copyFrom(d.lane_state - d.scratch.ptr + original * hb, d.x + local * hb, hb, e.stream.handle);
            try d.scratch.copyFrom(d.lane_k - d.scratch.ptr + original * kb, d.kb + local * kb, kb, e.stream.handle);
            try d.scratch.copyFrom(d.lane_v - d.scratch.ptr + original * kb, d.vb + local * kb, kb, e.stream.handle);
        }
        at = end;
    }
    return branchHead(d, queries.len, roots, out);
}

/// The round's absorbs: up to `most` rows a pass, each stream's rows one segment; larger or empty ones as before.
pub fn absorb(d: *Mtp, incoming: []const dr.Absorb) !void {
    const useful = try d.gpa.alloc(dr.Absorb, incoming.len);
    defer d.gpa.free(useful);
    const items = try @import("cuda_mtp_budget.zig").active(d, incoming, useful);
    if (items.len == 0) return;
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
    if (!d.head_enabled) {
        d.batch_passes += 1;
        return e.stream.synchronize();
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

test "MTP branch metadata isolates sibling keys and keeps each prefix's requested position" {
    var e: engine.Engine = undefined;
    e.g.vocab = 248320;
    e.g.kv_heads = 4;
    var d: Mtp = undefined;
    d.e = &e;
    d.capacity = 16384;
    var ctx: Ctx = .{ .ring = .{ .d = undefined, .ptr = 0x1000, .len = 1 << 28 }, .last = undefined, .step = undefined, .begin = 10, .end = 100 };
    var queries = [_]Branch{
        .{ .ctx = &ctx, .hidden = 0x2000, .token = 1, .position = 100 },
        .{ .ctx = &ctx, .hidden = 0x3000, .token = 2, .position = 103 },
    };
    var segs: [2]Seg = undefined;
    var table: [2]RingRow = undefined;
    var ids: [2]u32 = undefined;
    try branchMetadata(&d, &queries, &segs, &table, &ids);
    for (table, segs, 0..) |row, seg, i| {
        try std.testing.expectEqual(@as(i32, @intCast(i)), row.seg);
        try std.testing.expectEqual(@as(i64, 100), row.end);
        try std.testing.expectEqual(@as(usize, 1), seg.rows);
        try std.testing.expectEqual(queries[i].position, seg.p0);
    }
    queries[1].token = 248320;
    try std.testing.expectError(error.BadMtpBranches, branchMetadata(&d, &queries, &segs, &table, &ids));
}

test "MTP prefix order runs parents first, keeps stream boundaries and rejects crossed or shifted ancestry" {
    var a: Ctx = .{ .ring = undefined, .last = undefined, .step = undefined, .end = 100 };
    var b: Ctx = .{ .ring = undefined, .last = undefined, .step = undefined, .end = 200 };
    var queries = [_]Branch{
        .{ .ctx = &a, .hidden = 0, .token = 1, .position = 100 },
        .{ .ctx = &a, .hidden = 0, .token = 2, .position = 101, .parent = 0 },
        .{ .ctx = &a, .hidden = 0, .token = 3, .position = 102, .parent = 1 },
        .{ .ctx = &b, .hidden = 0, .token = 4, .position = 200 },
        .{ .ctx = &b, .hidden = 0, .token = 5, .position = 201, .parent = 3 },
    };
    var order: [5]usize = undefined;
    var depth: [5]usize = undefined;
    try prefixOrder(&queries, &order, &depth);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 1, 4, 2 }, &order);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 0, 1 }, &depth);
    queries[4].parent = 1;
    try std.testing.expectError(error.BadMtpBranches, prefixOrder(&queries, &order, &depth));
    queries[4].parent = 3;
    queries[2].position = 103;
    try std.testing.expectError(error.BadMtpBranches, prefixOrder(&queries, &order, &depth));
}
