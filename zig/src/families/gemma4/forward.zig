//! One stream's decode window through Gemma 4 dense: embed, 60 layers of glue and lane projections, the head.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const state = @import("state.zig");
const kernels = @import("kernels.zig");
const weights = @import("weights.zig");
const Model = @import("model.zig").Model;

const Kind = kernels.Kind;

/// One stream's rows in a forward: its cache, and how many rows it adds at the cache's length.
pub const Segment = struct { cache: *state.Cache, rows: usize };

/// Rows a call of the attention kernels takes: its partials' scratch is sized for this many.
const attend_rows = state.max_rows;

/// Each sub-batch's chunk count, by its first row; the kernels' meta sits at that row too.
const Chunks = [state.prompt_rows][2]usize;

/// Host-side inputs: ids, each row's position and first key, and each attention sub-batch's meta at its first row.
fn prepare(s: *state.Scratch, segments: []const Segment, ids: []const u32, chunks: *Chunks) void {
    @memcpy(s.ids.slice(u32, ids.len), ids);
    var at: usize = 0;
    for (segments) |seg| {
        const p0 = seg.cache.len;
        for (s.pos.slice(i32, ids.len)[at..][0..seg.rows], 0..) |*p, i| p.* = @intCast(p0 + i);
        for (0..2) |kind| {
            const global = kind == 1;
            const ck: usize = if (global) 64 else 128;
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

fn project(m: *Model, e: mtl.ComputeEncoder, l: weights.Linear, x: mtl.Buffer, y: mtl.Buffer, rows: usize) !void {
    const p = try m.projections.get(m.gpa, m.device, l.n, l.k);
    try p.encode(e, .{ .buf = x }, .{ .buf = l.w }, .{ .buf = l.sb }, .{ .buf = y }, @intCast(rows));
}

fn bind(e: mtl.ComputeEncoder, bufs: []const mtl.Buffer, first: usize) void {
    for (bufs, first..) |b, i| e.setBuffer(b, 0, i);
}

fn bindRef(e: mtl.ComputeEncoder, r: weights.Ref, i: usize) void {
    e.setBuffer(r.buf, r.off, i);
}

/// Every segment's rows in one forward: soft-capped logits in s.logits by segment; each cache grows by its rows.
pub fn run(m: *Model, s: *state.Scratch, segments: []const Segment, ids: []const u32) !void {
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
        while (sub < seg.rows) : (sub += attend_rows) for (chunks[at + sub]) |n| if (cfg.query_heads * attend_rows * n > s.slots) return error.ContextFull;
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
    e.setValue(embedScale(), 5);
    e.dispatchThreads(.{ .width = cfg.hidden / 2, .height = rows }, .{ .width = 256 });
    e.setPipeline(ks.rms);
    bind(e, &.{s.h[0]}, 0);
    bindRef(e, w.layers[0].input_norm, 1);
    bind(e, &.{s.normed[0]}, 2);
    e.setValue(cfg.eps, 3);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
    var cur: usize = 0;
    for (0..cfg.layers) |i| {
        try layer(m, e, s, segments, &chunks, i, cur, rows);
        cur = 1 - cur;
    }
    try project(m, e, w.head, s.normed[cur], s.logits, rows);
    e.setPipeline(ks.softcap);
    bind(e, &.{s.logits}, 0);
    e.setValue(cfg.softcap, 1);
    e.dispatchThreads(.{ .width = rows * cfg.vocab }, .{ .width = 256 });
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

/// sqrt(hidden) as MLX multiplies by it: a weak-typed scalar, rounded to bf16 first.
pub fn embedScale() f32 {
    const x: f32 = @sqrt(@as(f32, cfg.hidden));
    const b: u32 = @bitCast(x);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

fn layer(m: *Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment, chunks: *const Chunks, i: usize, cur: usize, rows: usize) !void {
    const global = cfg.global(i);
    const kd: *const Kind = m.kernels.kind(global);
    const l = &m.weights.layers[i];
    const nk = cfg.kvHeads(i);
    const dh = cfg.headDim(i);
    const kind: usize = @intFromBool(global);
    const width = l.qkv.n;
    try project(m, e, l.qkv, s.normed[cur], s.qkv, rows);
    var at: usize = 0;
    for (segments) |seg| {
        const n = seg.rows;
        const c = seg.cache;
        const kv_off = at * nk * dh * 2; // this stream's keys and values [nk, n, dh] in s.k, s.v
        const q_off = at * cfg.query_heads * dh * 2;
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
        e.dispatchGroups(.{ .width = cfg.query_heads + 2 * nk, .height = n }, .{ .width = 32 });
        e.setPipeline(kd.kv_write);
        e.setBuffer(s.k, kv_off, 0);
        e.setBuffer(s.v, kv_off, 1);
        bind(e, &.{ c.k[i], c.v[i] }, 2);
        e.setBuffer(s.pos, at * 4, 4);
        e.setValue([3]i32{ @intCast(n), if (global) 0 else state.ring, @intCast(state.slotsOf(i, c.capacity)) }, 5);
        e.dispatchThreads(.{ .width = dh, .height = n, .depth = nk }, .{ .width = @min(dh, 256) });
        var sub: usize = 0;
        while (sub < n) : (sub += attend_rows) {
            const r = at + sub;
            const rn = @min(attend_rows, n - sub);
            e.setPipeline(kd.partial);
            e.setBuffer(s.q, r * cfg.query_heads * dh * 2, 0);
            bind(e, &.{ c.k[i], c.v[i], c.k[i], c.v[i] }, 1); // the new rows were written first: every key is a cached one
            e.setBuffer(s.pos, r * 4, 5);
            e.setBuffer(s.lo[kind], r * 4, 6);
            e.setBuffer(s.meta[kind], r * state.meta_ints * 4, 7);
            e.setValue(@as(i32, @intCast(state.slotsOf(i, c.capacity))), 8);
            bind(e, &.{ s.pm, s.pl, s.po }, 9);
            e.dispatchGroups(.{ .width = chunks[r][kind], .height = nk, .depth = rn }, .{ .width = 32 * kd.heads_per_kv * kd.split });
            e.setPipeline(kd.merge);
            bind(e, &.{ s.pm, s.pl, s.po }, 0);
            e.setBuffer(s.meta[kind], r * state.meta_ints * 4, 3);
            e.setBuffer(s.attn, r * cfg.query_heads * dh * 2, 4);
            e.dispatchGroups(.{ .width = 1, .height = cfg.query_heads, .depth = rn }, .{ .width = 32 });
        }
        at += n;
    }
    try project(m, e, l.o, s.attn, s.o, rows);
    e.setPipeline(kd.attn_tail);
    bind(e, &.{ s.h[cur], s.o }, 0);
    bindRef(e, l.post_attention, 2);
    bindRef(e, l.pre_feedforward, 3);
    e.setValue(cfg.eps, 4);
    bind(e, &.{ s.hn, s.n_mlp }, 5);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    try project(m, e, l.gate_up, s.n_mlp, s.gate_up, rows);
    e.setPipeline(kd.geglu);
    bind(e, &.{ s.gate_up, s.act }, 0);
    e.setValue(@as(u32, cfg.intermediate), 2);
    e.dispatchThreads(.{ .width = cfg.intermediate, .height = rows }, .{ .width = 256 });
    try project(m, e, l.down, s.act, s.y, rows);
    e.setPipeline(kd.dense_tail);
    bind(e, &.{ s.hn, s.y }, 0);
    bindRef(e, l.post_feedforward, 2);
    bindRef(e, l.scalar, 3);
    bindRef(e, if (i + 1 < cfg.layers) m.weights.layers[i + 1].input_norm else m.weights.final_norm, 4);
    e.setValue(cfg.eps, 5);
    bind(e, &.{ s.h[1 - cur], s.normed[1 - cur] }, 6);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
}
