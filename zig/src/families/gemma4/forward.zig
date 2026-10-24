//! One stream's decode window through Gemma 4 dense: embed, each layer's glue and lane projections, the head.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const state = @import("state.zig");
const kernels = @import("kernels.zig");
const weights = @import("weights.zig");
const lane = @import("../../core/lane_projection.zig");
const Model = @import("model.zig").Model;

const Kind = kernels.Kind;

/// One stream's rows in a forward: its cache, and how many rows it adds at the cache's length.
pub const Segment = struct { cache: *state.Cache, rows: usize };

/// Which of a layer's stages a call encodes (the bench times each alone); every forward takes all three.
pub const Stages = struct { mm: bool = true, attn: bool = true, glue: bool = true };

/// Rows a call of the attention kernels takes: its partials' scratch is sized for this many.
const attend_rows = state.max_rows;
/// Keys a gemma_decode_attention threadgroup takes (its CD).
const decode_chunk = 256;

/// Each sub-batch's chunk count, by its first row; the kernels' meta sits at that row too.
pub const Chunks = [state.prompt_rows][2]usize;

/// Host-side inputs: ids, each row's position and first key, and each attention sub-batch's meta at its first row.
pub fn prepare(s: *state.Scratch, segments: []const Segment, ids: []const u32, chunks: *Chunks) void {
    @memcpy(s.ids.slice(u32, ids.len), ids);
    var at: usize = 0;
    for (segments) |seg| {
        const p0 = seg.cache.len;
        for (s.pos.slice(i32, ids.len)[at..][0..seg.rows], 0..) |*p, i| p.* = @intCast(p0 + i);
        for (0..2) |kind| {
            const global = kind == 1;
            const ck: usize = if (global) decode_chunk else 128;
            const lo = s.lo[kind].slice(i32, ids.len);
            for (lo[at..][0..seg.rows], 0..) |*l, i| {
                const p = p0 + i;
                l.* = @intCast(if (global or p + 1 <= cfg.window) 0 else p + 1 - cfg.window);
            }
            var sub: usize = 0;
            while (sub < seg.rows) : (sub += attend_rows) {
                const n = @min(attend_rows, seg.rows - sub);
                const first_chunk = @as(usize, @intCast(lo[at + sub])) / ck;
                chunks[at + sub][kind] = (p0 + sub + n - 1) / ck - first_chunk + 1;
                const ring: i32 = if (global) 0 else state.ring;
                // every key the sub-batch reads is in the cache already: no row counts as new (first new key past all)
                s.meta[kind].slice(i32, state.prompt_rows * state.meta_ints)[(at + sub) * state.meta_ints ..][0..5].* = .{ @intCast(first_chunk), @intCast(chunks[at + sub][kind]), ring, @intCast(n), std.math.maxInt(i32) };
            }
        }
        at += seg.rows;
    }
}

/// A projection: the row path on a chip without tensor units; at 4/64 on the group sums its input's kernel wrote.
fn project(m: *Model, s: *state.Scratch, e: mtl.ComputeEncoder, l: weights.Linear, x: mtl.Buffer, y: mtl.Buffer, rows: usize) !void {
    if (l.raw) |raw| return weights.projectRow(&m.projections, m.device, e, raw, l, x, y, rows);
    const p = try m.projections.get(m.gpa, m.device, l);
    const out: lane.Ref = .{ .buf = y, .off = l.col * 2 };
    if (!l.reg()) return l.encodeRows(p, e, .{ .buf = x }, out, rows);
    try p.encodeSums(e, .{ .buf = x }, .{ .buf = l.w }, .{ .buf = l.sb }, .{ .ref = .{ .buf = s.sums }, .stride = @intCast(rows) }, out, @intCast(rows));
}

/// A stack's runs, each writing its columns of y's rows.
fn projectStack(m: *Model, s: *state.Scratch, e: mtl.ComputeEncoder, st: *const weights.Stack, x: mtl.Buffer, y: mtl.Buffer, rows: usize) !void {
    for (st.list()) |l| try project(m, s, e, l, x, y, rows);
}

/// The sums a producing kernel writes: s.sums with the window's rows as stride.
fn sumsOut(e: mtl.ComputeEncoder, s: *state.Scratch, rows: usize, index: usize) void {
    e.setBuffer(s.sums, 0, index);
    e.setValue(@as(i32, @intCast(rows)), index + 1);
}

fn bind(e: mtl.ComputeEncoder, bufs: []const mtl.Buffer, first: usize) void {
    for (bufs, first..) |b, i| e.setBuffer(b, 0, i);
}

fn bindRef(e: mtl.ComputeEncoder, r: weights.Ref, i: usize) void {
    e.setBuffer(r.buf, r.off, i);
}

/// Every segment's rows in one forward: soft-capped logits in s.logits by segment; each cache grows by its rows.
pub fn run(m: *Model, s: *state.Scratch, segments: []const Segment, ids: []const u32) !void {
    const geo = m.config.geo;
    var rows: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.ContextFull;
        rows += seg.rows;
    }
    if (rows == 0 or rows > state.prompt_rows or rows != ids.len) return error.BadWindow;
    var chunks: Chunks = undefined;
    prepare(s, segments, ids, &chunks);
    var at: usize = 0;
    for (segments) |seg| {
        var sub: usize = 0;
        while (sub < seg.rows) : (sub += attend_rows) for (chunks[at + sub]) |n| if (geo.query_heads * attend_rows * n > s.slots) return error.ContextFull;
        at += seg.rows;
    }
    const w = &m.weights;
    const ks = &m.kernels.sliding;
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(ks.embed);
    bind(e, &.{s.ids}, 0);
    bindRef(e, w.embed_w, 1);
    bindRef(e, w.embed_s, 2);
    bindRef(e, w.embed_b, 3);
    bind(e, &.{s.h[0]}, 4);
    e.setValue(embedScale(geo.hidden), 5);
    e.dispatchThreads(.{ .width = geo.hidden / 2, .height = rows }, .{ .width = 256 });
    e.setPipeline(ks.rms);
    bind(e, &.{s.h[0]}, 0);
    bindRef(e, w.layers[0].input_norm, 1);
    bind(e, &.{s.normed[0]}, 2);
    e.setValue(cfg.eps, 3);
    sumsOut(e, s, rows, 4);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
    var cur: usize = 0;
    const fused = rows == 1 and m.fuse;
    for (0..geo.layers) |i| {
        try layer(m, e, s, segments, &chunks, i, cur, rows, .{}, fused);
        cur = 1 - cur;
    }
    if (fused) denseTail(m, e, s, geo.layers - 1, 1 - cur, rows); // the last layer's tail, no projection to carry it
    try project(m, s, e, w.head, s.normed[cur], s.logits, rows);
    e.setPipeline(ks.softcap);
    bind(e, &.{s.logits}, 0);
    e.setValue(cfg.softcap, 1);
    e.dispatchThreads(.{ .width = rows * cfg.vocab }, .{ .width = 256 });
    // each row's greedy token (the softcap keeps the order): tree_round's argmax in 16-row pieces
    var first: usize = 0;
    while (first < rows) : (first += 16) {
        const n = @min(16, rows - first);
        try m.picks.argmax(e, .{ .buf = s.logits, .off = first * cfg.vocab * 2 }, .{ .buf = s.picks, .off = first * 8 }, .{ .rows = @intCast(n), .vocab = cfg.vocab, .stride = cfg.vocab });
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("gemma4: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
    for (segments) |seg| seg.cache.len += seg.rows;
}

/// One stream's window at the cache's length (`p0` must equal it).
pub fn window(m: *Model, s: *state.Scratch, c: *state.Cache, ids: []const u32, p0: usize) !void {
    if (p0 != c.len) return error.BadPosition;
    return run(m, s, &.{.{ .cache = c, .rows = ids.len }}, ids);
}

/// A dense tail's tap output: layer i's rows at its tap's columns and their stride, or `spare` with stride 0.
pub fn tapRef(m: *const Model, taps: ?mtl.Buffer, spare: mtl.Buffer, i: usize) struct { ref: lane.Ref, stride: i32 } {
    const t = m.tap_of[i];
    if (taps == null or t < 0) return .{ .ref = .{ .buf = spare }, .stride = 0 };
    const hidden = m.config.geo.hidden;
    return .{ .ref = .{ .buf = taps.?, .off = @as(usize, @intCast(t)) * hidden * 2 }, .stride = @intCast(m.taps * hidden) };
}

pub fn tapOut(e: mtl.ComputeEncoder, m: *const Model, taps: ?mtl.Buffer, spare: mtl.Buffer, i: usize, index: usize) void {
    const t = tapRef(m, taps, spare, i);
    e.setBuffer(t.ref.buf, t.ref.off, index);
    e.setValue(t.stride, index + 1);
}

/// Layer i's dense tail on the window's rows: the next residual and normed input in s.h and s.normed[1 - cur].
fn denseTail(m: *Model, e: mtl.ComputeEncoder, s: *state.Scratch, i: usize, cur: usize, rows: usize) void {
    const l = &m.weights.layers[i];
    e.setPipeline(m.kernels.sliding.dense_tail);
    bind(e, &.{ s.hn, s.y }, 0);
    bindRef(e, l.post_feedforward, 2);
    bindRef(e, l.scalar, 3);
    bindRef(e, if (i + 1 < m.config.geo.layers) m.weights.layers[i + 1].input_norm else m.weights.final_norm, 4);
    e.setValue(cfg.eps, 5);
    bind(e, &.{ s.h[1 - cur], s.normed[1 - cur] }, 6);
    sumsOut(e, s, rows, 8);
    tapOut(e, m, s.taps, s.hn, i, 10);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
}

/// A fused projection on one row: its prologue's inputs and outputs (gemma4/fused.metal); eps in f.x, a tap in i.x.
fn projectFused(m: *Model, e: mtl.ComputeEncoder, l: weights.Linear, fuse: u8, x: mtl.Buffer, y: mtl.Buffer, a: [6]lane.Ref, o: [2]lane.Ref, tap: i32) !void {
    const p = try m.projections.getFused(m.gpa, m.device, l, fuse);
    try p.encodeFused(e, .{ .buf = x }, .{ .buf = l.w }, .{ .buf = l.sb }, .{ .buf = y }, .{ .a = a, .o = o, .f = .{ cfg.eps, 0, 0, 0 }, .i = .{ tap, 0, 0, 0 } }, 1);
}

/// sqrt(hidden) as MLX multiplies by it: a weak-typed scalar, rounded to bf16 first.
pub fn embedScale(hidden: usize) f32 {
    const x: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
    const b: u32 = @bitCast(x);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

/// Layer i on a window's rows; `fused` (one row): dense tails as the next qkv's prologue, attention tails as gate/up's.
pub fn layer(m: *Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment, chunks: *const Chunks, i: usize, cur: usize, rows: usize, st: Stages, fused: bool) !void {
    const geo = m.config.geo;
    const global = cfg.global(i);
    const kd: *const Kind = m.kernels.kind(global);
    const l = &m.weights.layers[i];
    const nk = geo.kvHeads(i);
    const dh = cfg.headDim(i);
    const kind: usize = @intFromBool(global);
    const width = l.qkv.n;
    if (st.mm and fused and i > 0) {
        const pl = &m.weights.layers[i - 1];
        const tap = tapRef(m, s.taps, s.hn, i - 1);
        try projectFused(m, e, l.qkv.runs[0], 1, s.y, s.qkv, .{ .{ .buf = s.hn }, .{ .buf = s.y }, pl.post_feedforward, pl.scalar, l.input_norm, l.input_norm }, .{ .{ .buf = s.h[cur] }, tap.ref }, tap.stride);
    } else if (st.mm) try projectStack(m, s, e, &l.qkv, s.normed[cur], s.qkv, rows);
    var at: usize = 0;
    for (segments) |seg| {
        const n = seg.rows;
        const c = seg.cache;
        if (!st.attn) {
            at += n;
            continue;
        }
        const kv_off = at * nk * dh * 2; // this stream's keys and values [nk, n, dh] in s.k, s.v
        const q_off = at * geo.query_heads * dh * 2;
        const prep = !(fused and !global); // a fused sliding layer's attention kernel preps its own heads
        if (prep) {
            e.setPipeline(kd.qkv);
            e.setBuffer(s.qkv, at * width * 2, 0);
            bindRef(e, l.q_norm, 1);
            bindRef(e, l.k_norm, 2);
            bind(e, &.{if (global) m.inv_global else m.inv_sliding}, 3);
            e.setBuffer(s.pos, at * 4, 4);
            e.setValue(cfg.eps, 5);
            e.setBuffer(s.q, q_off, 6);
            e.setBuffer(s.k, kv_off, 7);
            e.setBuffer(s.v, kv_off, 8);
            bind(e, &.{ c.k[i], c.v[i] }, 9); // keys and values straight into the cache at each row's slot
            e.setValue([3]i32{ if (global) 0 else state.ring, @intCast(state.slotsOf(i, c.capacity)), 0 }, 11);
            e.dispatchGroups(.{ .width = geo.query_heads + 2 * nk, .height = n }, .{ .width = 32 });
        }
        var sub: usize = 0;
        while (sub < n) : (sub += attend_rows) {
            const r = at + sub;
            const rn = @min(attend_rows, n - sub);
            if (global) { // a key head's eight query heads as one tile: each key read once a token (M5's tensor units)
                const nch = chunks[r][kind]; // from the first key, in 256-key chunks (prepare)
                e.setPipeline(kd.decode_attn);
                e.setBuffer(s.q, r * geo.query_heads * dh * 2, 0);
                bind(e, &.{ c.k[i], c.v[i] }, 1);
                e.setValue([4]i32{ @intCast(rn), @intCast(c.len + sub), @intCast(state.slotsOf(i, c.capacity)), @intCast(nch) }, 3);
                bind(e, &.{ s.pm, s.pl, s.po }, 4);
                e.dispatchGroups(.{ .width = nch, .height = nk, .depth = (rn * kd.heads_per_kv + 15) / 16 }, .{ .width = 32 * dh / 128 });
                e.setPipeline(kd.merge);
                bind(e, &.{ s.pm, s.pl, s.po }, 0);
                e.setBuffer(s.meta[kind], r * state.meta_ints * 4, 3);
                e.setBuffer(s.attn, r * geo.query_heads * dh * 2, 4);
                e.setBuffer(s.sums, r * 4, 5); // this sub-batch's rows of the o projection's sums
                e.setValue(@as(i32, @intCast(rows)), 6);
                e.dispatchGroups(.{ .width = 1, .height = geo.query_heads, .depth = rn }, .{ .width = 32 });
                continue;
            }
            const direct = !prep and chunks[r][kind] == 1; // one chunk: the kernel writes the merged output itself
            e.setPipeline(kd.partial);
            e.setBuffer(s.q, r * geo.query_heads * dh * 2, 0);
            bind(e, &.{ c.k[i], c.v[i], c.k[i], c.v[i] }, 1); // the new rows were written first: every key is a cached one
            e.setBuffer(s.pos, r * 4, 5);
            e.setBuffer(s.lo[kind], r * 4, 6);
            e.setBuffer(s.meta[kind], r * state.meta_ints * 4, 7);
            e.setValue(@as(i32, @intCast(state.slotsOf(i, c.capacity))), 8);
            bind(e, &.{ s.pm, s.pl, s.po }, 9);
            e.setBuffer(s.qkv, r * width * 2, 12);
            bindRef(e, l.q_norm, 13);
            bindRef(e, l.k_norm, 14);
            bind(e, &.{m.inv_sliding}, 15);
            e.setValue(cfg.eps, 16);
            e.setValue([2]i32{ @intFromBool(!prep), @intFromBool(direct) }, 17);
            e.setBuffer(s.attn, r * geo.query_heads * dh * 2, 18);
            e.setBuffer(s.sums, r * 4, 19);
            e.setValue(@as(i32, @intCast(rows)), 20);
            bind(e, &.{ c.k[i], c.v[i] }, 21);
            e.dispatchGroups(.{ .width = chunks[r][kind], .height = nk, .depth = rn }, .{ .width = 32 * kd.heads_per_kv * kd.split });
            if (direct) continue;
            e.setPipeline(kd.merge);
            bind(e, &.{ s.pm, s.pl, s.po }, 0);
            e.setBuffer(s.meta[kind], r * state.meta_ints * 4, 3);
            e.setBuffer(s.attn, r * geo.query_heads * dh * 2, 4);
            e.setBuffer(s.sums, r * 4, 5);
            e.setValue(@as(i32, @intCast(rows)), 6);
            e.dispatchGroups(.{ .width = 1, .height = geo.query_heads, .depth = rn }, .{ .width = 32 });
        }
        at += n;
    }
    if (st.mm) try project(m, s, e, l.o, s.attn, s.o, rows);
    if (!st.glue) {
        if (st.mm) try projectStack(m, s, e, &l.gate_up, s.n_mlp, s.gate_up, rows);
        if (st.mm) try project(m, s, e, l.down, s.act, s.y, rows);
        return;
    }
    if (fused and st.mm) {
        try projectFused(m, e, l.gate_up.runs[0], 2, s.o, s.act, .{ .{ .buf = s.h[cur] }, .{ .buf = s.o }, l.post_attention, l.pre_feedforward, l.pre_feedforward, l.pre_feedforward }, .{ .{ .buf = s.hn }, .{ .buf = s.hn } }, 0);
        try projectFused(m, e, l.down, 3, s.act, s.y, .{ .{ .buf = s.act }, .{ .buf = s.act }, .{ .buf = s.act }, .{ .buf = s.act }, .{ .buf = s.act }, .{ .buf = s.act } }, .{ .{ .buf = s.y }, .{ .buf = s.y } }, 0);
        return;
    }
    e.setPipeline(kd.attn_tail);
    bind(e, &.{ s.h[cur], s.o }, 0);
    bindRef(e, l.post_attention, 2);
    bindRef(e, l.pre_feedforward, 3);
    e.setValue(cfg.eps, 4);
    bind(e, &.{ s.hn, s.n_mlp }, 5);
    sumsOut(e, s, rows, 7);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    if (st.mm) try projectStack(m, s, e, &l.gate_up, s.n_mlp, s.gate_up, rows);
    e.setPipeline(kd.geglu);
    bind(e, &.{ s.gate_up, s.act }, 0);
    e.setValue(@as(u32, @intCast(geo.intermediate)), 2);
    sumsOut(e, s, rows, 3);
    e.setValue(@as(i32, @intFromBool(l.gate_up.interleaved)), 5); // one run: its rows interleaved in 16-row blocks
    e.dispatchGroups(.{ .width = geo.intermediate / 64, .height = rows }, .{ .width = 64 });
    if (st.mm) try project(m, s, e, l.down, s.act, s.y, rows);
    denseTail(m, e, s, i, cur, rows);
}
