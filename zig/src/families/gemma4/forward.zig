//! One stream's decode window through Gemma 4 dense: embed, 60 layers of glue and lane projections, the head.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const state = @import("state.zig");
const kernels = @import("kernels.zig");
const weights = @import("weights.zig");
const Model = @import("model.zig").Model;

const Kind = kernels.Kind;

/// The window's host-side inputs: token ids, positions, and each layer kind's key ranges and attention chunks.
fn prepare(s: *state.Scratch, ids: []const u32, p0: usize) [2]usize {
    const rows = ids.len;
    @memcpy(s.ids.slice(u32, rows), ids);
    for (s.pos.slice(i32, rows), 0..) |*p, i| p.* = @intCast(p0 + i);
    var chunks: [2]usize = undefined;
    for (0..2) |kind| {
        const global = kind == 1;
        const ck: usize = if (global) 64 else 128;
        var low: usize = std.math.maxInt(usize);
        for (s.lo[kind].slice(i32, rows), 0..) |*l, i| {
            const p = p0 + i;
            const first: usize = if (global or p + 1 <= cfg.window) 0 else p + 1 - cfg.window;
            l.* = @intCast(first);
            low = @min(low, first);
        }
        const first_chunk = low / ck;
        chunks[kind] = (p0 + rows - 1) / ck - first_chunk + 1;
        const ring: i32 = if (global) 0 else state.ring;
        s.meta[kind].slice(i32, 4)[0..4].* = .{ @intCast(first_chunk), @intCast(chunks[kind]), ring, @intCast(rows) };
    }
    return chunks;
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

/// The window's logits [rows, vocab] bf16 (soft-capped) in s.logits; the cache keeps the window's keys and values.
pub fn window(m: *Model, s: *state.Scratch, c: *state.Cache, ids: []const u32, p0: usize) !void {
    const rows = ids.len;
    if (rows == 0 or rows > state.max_rows) return error.BadWindow;
    if (p0 + rows > c.capacity) return error.ContextFull;
    const chunks = prepare(s, ids, p0);
    for (chunks) |n| if (cfg.query_heads * rows * n > s.slots) return error.ContextFull;
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
        try layer(m, e, s, c, i, cur, rows, chunks[@intFromBool(cfg.global(i))]);
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
    c.len = p0 + rows;
}

/// sqrt(hidden) as MLX multiplies by it: a weak-typed scalar, rounded to bf16 first.
pub fn embedScale() f32 {
    const x: f32 = @sqrt(@as(f32, cfg.hidden));
    const b: u32 = @bitCast(x);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

fn layer(m: *Model, e: mtl.ComputeEncoder, s: *state.Scratch, c: *state.Cache, i: usize, cur: usize, rows: usize, chunks: usize) !void {
    const global = cfg.global(i);
    const kd: *const Kind = m.kernels.kind(global);
    const l = &m.weights.layers[i];
    const nk = cfg.kvHeads(i);
    const dh = cfg.headDim(i);
    const kind: usize = @intFromBool(global);
    try project(m, e, l.qkv, s.normed[cur], s.qkv, rows);
    e.setPipeline(kd.qkv);
    bind(e, &.{s.qkv}, 0);
    bindRef(e, l.q_norm, 1);
    bindRef(e, l.k_norm, 2);
    bind(e, &.{ if (global) m.inv_global else m.inv_sliding, s.pos }, 3);
    e.setValue(cfg.eps, 5);
    bind(e, &.{ s.q, s.k, s.v }, 6);
    e.dispatchGroups(.{ .width = cfg.query_heads + 2 * nk, .height = rows }, .{ .width = 32 });
    e.setPipeline(kd.partial);
    bind(e, &.{ s.q, c.k[i], c.v[i], s.k, s.v, s.pos, s.lo[kind], s.meta[kind] }, 0);
    e.setValue(@as(i32, @intCast(state.slotsOf(i, c.capacity))), 8);
    bind(e, &.{ s.pm, s.pl, s.po }, 9);
    e.dispatchGroups(.{ .width = chunks, .height = nk, .depth = rows }, .{ .width = 32 * kd.heads_per_kv * kd.split });
    e.setPipeline(kd.merge);
    bind(e, &.{ s.pm, s.pl, s.po, s.meta[kind], s.attn }, 0);
    e.dispatchGroups(.{ .width = 1, .height = cfg.query_heads, .depth = rows }, .{ .width = 32 });
    e.setPipeline(kd.kv_write);
    bind(e, &.{ s.k, s.v, c.k[i], c.v[i], s.pos }, 0);
    e.setValue([3]i32{ @intCast(rows), if (global) 0 else state.ring, @intCast(state.slotsOf(i, c.capacity)) }, 5);
    e.dispatchThreads(.{ .width = dh, .height = rows, .depth = nk }, .{ .width = @min(dh, 256) });
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
