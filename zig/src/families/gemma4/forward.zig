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
                const win = seg.cache.geo.sliding_window;
                l.* = @intCast(if (global or p + 1 <= win) 0 else p + 1 - win);
            }
            var sub: usize = 0;
            while (sub < seg.rows) : (sub += attend_rows) {
                const n = @min(attend_rows, seg.rows - sub);
                const first_chunk = @as(usize, @intCast(lo[at + sub])) / ck;
                chunks[at + sub][kind] = (p0 + sub + n - 1) / ck - first_chunk + 1;
                const ring: i32 = if (global) 0 else @intCast(state.ringOf(seg.cache.geo));
                // every key the sub-batch reads is in the cache already: no row counts as new (first new key past all)
                s.meta[kind].slice(i32, state.prompt_rows * state.meta_ints)[(at + sub) * state.meta_ints ..][0..5].* = .{ @intCast(first_chunk), @intCast(chunks[at + sub][kind]), ring, @intCast(n), std.math.maxInt(i32) };
            }
        }
        at += seg.rows;
    }
}

/// A projection: the row path on a chip without tensor units; at 4/64 on the group sums its input's kernel wrote.
pub fn project(m: *Model, s: *state.Scratch, e: mtl.ComputeEncoder, l: weights.Linear, x: mtl.Buffer, y: mtl.Buffer, rows: usize) !void {
    if (l.raw) |raw| return weights.projectRow(&m.projections, m.device, e, raw, l, .{ .buf = x }, y, rows);
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
fn sumsOut(e: mtl.ComputeEncoder, s: anytype, rows: usize, index: usize) void {
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
    return runWindow(m, s, segments, ids, .all);
}

/// Which rows of a window are read; none: it stops before Model.deepFrom's layers and the head.
const Reads = enum { all, last, none };

/// run; `.last`: a prompt pass's window, where only the last row needs logits (on the row path; not with tensor units).
fn runWindow(m: *Model, s: *state.Scratch, segments: []const Segment, ids: []const u32, reads: Reads) !void {
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
    embedRows(m, e, s, rows);
    e.setPipeline(ks.rms);
    bind(e, &.{s.h[0]}, 0);
    bindRef(e, w.layers[0].input_norm, 1);
    bind(e, &.{s.normed[0]}, 2);
    e.setValue(cfg.eps, 3);
    sumsOut(e, s, rows, 4);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
    const used = if (reads == .none) m.deepFrom() else geo.layers;
    if (m.kernels.ple) |*pk| pleInputs(m, e, s, pk, rows, used);
    var cur: usize = 0;
    const fused = rows == 1 and m.fuse;
    for (0..used) |i| {
        try layer(m, e, s, segments, &chunks, i, cur, rows, .{}, fused);
        cur = 1 - cur;
    }
    if (fused and reads != .none) denseTail(m, e, s, geo.layers - 1, 1 - cur, rows); // the last layer's tail, no projection to carry it
    if (reads != .none) {
        // a prompt pass's window needs its last row's logits alone (at that row's place in s.logits)
        const from: usize = if (reads == .last and w.head.raw != null) rows - 1 else 0;
        if (from == 0) try project(m, s, e, w.head, s.normed[cur], s.logits, rows) else try weights.projectRowAt(&m.projections, m.device, e, w.head.raw.?, w.head, .{ .buf = s.normed[cur], .off = from * geo.hidden * 2 }, s.logits, from * cfg.vocab * 2, 1);
        e.setPipeline(ks.softcap);
        e.setBuffer(s.logits, from * cfg.vocab * 2, 0);
        e.setValue(cfg.softcap, 1);
        e.dispatchThreads(.{ .width = (rows - from) * cfg.vocab }, .{ .width = 256 });
        // each row's greedy token (the softcap keeps the order): tree_round's argmax in 16-row pieces
        var first: usize = from;
        while (first < rows) : (first += 16) {
            const n = @min(16, rows - first);
            try m.picks.argmax(e, .{ .buf = s.logits, .off = first * cfg.vocab * 2 }, .{ .buf = s.picks, .off = first * 8 }, .{ .rows = @intCast(n), .vocab = cfg.vocab, .stride = cfg.vocab });
        }
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

/// The window's rows' scaled embeddings in s.h[0].
pub fn embedRows(m: *Model, e: mtl.ComputeEncoder, s: *state.Scratch, rows: usize) void {
    const w = &m.weights;
    e.setPipeline(m.kernels.sliding.embed);
    bind(e, &.{s.ids}, 0);
    bindRef(e, w.embed_w, 1);
    bindRef(e, w.embed_s, 2);
    bindRef(e, w.embed_b, 3);
    bind(e, &.{s.h[0]}, 4);
    e.setValue(embedScale(m.config.geo.hidden), 5);
    e.dispatchThreads(.{ .width = m.config.geo.hidden / 2, .height = rows }, .{ .width = 256 });
}

/// A prompt pass's window: the same rows' arithmetic as `window`, logits for the last row alone.
pub fn promptWindow(m: *Model, s: *state.Scratch, c: *state.Cache, ids: []const u32, p0: usize) !void {
    if (p0 != c.len) return error.BadPosition;
    return runWindow(m, s, &.{.{ .cache = c, .rows = ids.len }}, ids, .last);
}

/// A prompt pass's window nothing reads: the cache fills, the layers from `Model.deepFrom` and the head are left out.
pub fn fillWindow(m: *Model, s: *state.Scratch, c: *state.Cache, ids: []const u32, p0: usize) !void {
    if (p0 != c.len) return error.BadPosition;
    return runWindow(m, s, &.{.{ .cache = c, .rows = ids.len }}, ids, .none);
}

/// x rounded to bf16, to nearest even.
fn bf16Round(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

/// y [rows, d.n] = x [rows, d.k] times a bf16 matrix d, a simdgroup four columns; y's first `cols` columns only.
fn denseBf16(e: mtl.ComputeEncoder, pk: *const kernels.Ple, x: mtl.Buffer, d: weights.Dense, y: mtl.Buffer, rows: usize, cols: usize) void {
    e.setPipeline(pk.dense);
    e.setBuffer(x, 0, 0);
    bindRef(e, d.w, 1);
    e.setBuffer(y, 0, 2);
    e.setValue([3]i32{ @intCast(rows), @intCast(d.n), @intCast(d.k) }, 3);
    e.dispatchGroups(.{ .width = (cols / 4 + 1) / 2 }, .{ .width = 64 });
}

/// The E-series' per-layer inputs of layers [0, used) for the window's rows, into s.ple.pli.
pub fn pleInputs(m: *Model, e: mtl.ComputeEncoder, s: anytype, pk: *const kernels.Ple, rows: usize, used: usize) void {
    const geo = m.config.geo;
    const t = m.weights.ple.?;
    const sp = s.ple.?;
    const wide = geo.layers * geo.layer_input;
    e.setPipeline(pk.embed);
    bind(e, &.{s.ids}, 0);
    bindRef(e, t.table_w, 1);
    bindRef(e, t.table_s, 2);
    bindRef(e, t.table_b, 3);
    bind(e, &.{sp.tok}, 4);
    e.setValue(bf16Round(@sqrt(@as(f32, @floatFromInt(geo.layer_input)))), 5);
    e.dispatchThreads(.{ .width = wide / 2, .height = rows }, .{ .width = 256 });
    // The model projection of the scaled embeddings (s.h[0]) into sp.prj: DeepMind _modules.py encode_per_layer_input.
    denseBf16(e, pk, s.h[0], t.model_projection, sp.prj, rows, used * geo.layer_input);
    // Projection norm, plus the table rows, times rsqrt(2): one group per (layer, row), P threads.
    e.setPipeline(pk.combine);
    bind(e, &.{ sp.prj, sp.tok }, 0);
    bindRef(e, t.projection_norm, 2);
    e.setValue(cfg.eps, 3);
    e.setValue(@as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(geo.hidden))), 4);
    e.setValue(@as(f32, 1.0) / @sqrt(@as(f32, 2.0)), 5);
    bind(e, &.{sp.pli}, 6);
    e.dispatchGroups(.{ .width = used, .height = rows }, .{ .width = geo.layer_input });
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

/// Layer i's dense tail on the window's rows into s.h and s.normed[1 - cur]; E-series: unscaled, `pleTail` follows.
pub fn denseTail(m: *Model, e: mtl.ComputeEncoder, s: anytype, i: usize, cur: usize, rows: usize) void {
    const l = &m.weights.layers[i];
    const next = if (i + 1 < m.config.geo.layers) m.weights.layers[i + 1].input_norm else m.weights.final_norm;
    if (l.ple_gate != null) {
        const sp = s.ple.?;
        tail(m, e, s, s.hn, s.y, l.post_feedforward, .{ .buf = sp.one }, next, cur, rows, null);
        pleTail(m, e, s, i, cur, rows);
        return;
    }
    tail(m, e, s, s.hn, s.y, l.post_feedforward, l.scalar, next, cur, rows, i);
}

/// The E-series' per-layer block on the residual rows in s.h[1 - cur]: in place, then the next layer's normed input.
pub fn pleTail(m: *Model, e: mtl.ComputeEncoder, s: anytype, i: usize, cur: usize, rows: usize) void {
    // The Block's PLE step in DeepMind _modules.py: gate and gelu * per-layer input, projection, post norm, add, scale.
    const geo = m.config.geo;
    const l = &m.weights.layers[i];
    const sp = s.ple.?;
    const pk = &m.kernels.ple.?;
    e.setPipeline(pk.gate);
    e.setBuffer(s.h[1 - cur], 0, 0);
    bindRef(e, l.ple_gate.?.w, 1);
    e.setBuffer(sp.mid, 0, 2);
    e.setValue([3]i32{ @intCast(rows), @intCast(geo.layer_input), @intCast(geo.hidden) }, 3);
    e.setBuffer(sp.pli, 0, 4);
    e.setValue([2]i32{ @intCast(i * geo.layer_input), @intCast(geo.layers * geo.layer_input) }, 5);
    e.dispatchGroups(.{ .width = geo.layer_input / 4, .height = (rows + kernels.gate_rows - 1) / kernels.gate_rows }, .{ .width = 32 * kernels.gate_ksplit });
    denseBf16(e, pk, sp.mid, l.ple_projection.?, s.y, rows, geo.hidden);
    const nw = l.ple_norm.?;
    tail(m, e, s, s.h[1 - cur], s.y, nw, l.scalar, if (i + 1 < geo.layers) m.weights.layers[i + 1].input_norm else m.weights.final_norm, cur, rows, i);
}

/// gemma_dense_tail: (h + RMSNorm(y) w) * scalar into s.h[1 - cur] and the next normed input; `tap`: the tapped layer.
fn tail(m: *Model, e: mtl.ComputeEncoder, s: anytype, h: mtl.Buffer, y: mtl.Buffer, w: weights.Ref, scalar: weights.Ref, next: weights.Ref, cur: usize, rows: usize, tap: ?usize) void {
    e.setPipeline(m.kernels.sliding.dense_tail);
    bind(e, &.{ h, y }, 0);
    bindRef(e, w, 2);
    bindRef(e, scalar, 3);
    bindRef(e, next, 4);
    e.setValue(cfg.eps, 5);
    bind(e, &.{ s.h[1 - cur], s.normed[1 - cur] }, 6);
    sumsOut(e, s, rows, 8);
    if (tap) |i| tapOut(e, m, s.taps, s.hn, i, 10) else tapOut(e, m, null, s.hn, 0, 10);
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
    const kd: *const Kind = m.kernels.of(geo, i);
    const own_kv = state.kvSource(geo, i) == i; // a shared layer preps q alone and reads its source layer's cache
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
            e.setValue([3]i32{ if (global) 0 else @intCast(state.ringOf(c.geo)), @intCast(state.slotsOfGeo(c.geo, i, c.capacity)), 0 }, 11);
            e.dispatchGroups(.{ .width = geo.query_heads + (if (own_kv) 2 * nk else 0), .height = n }, .{ .width = 32 });
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
                e.setValue([4]i32{ @intCast(rn), @intCast(c.len + sub), @intCast(state.slotsOfGeo(c.geo, i, c.capacity)), @intCast(nch) }, 3);
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
            e.setValue(@as(i32, @intCast(state.slotsOfGeo(c.geo, i, c.capacity))), 8);
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
