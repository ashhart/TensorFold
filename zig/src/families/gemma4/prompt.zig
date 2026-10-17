//! A prompt chunk of up to 512 rows of one stream: core/affine_mm projections, the decode glue, the last row's logits.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const state = @import("state.zig");
const kernels = @import("kernels.zig");
const affine_mm = @import("affine_mm.zig");
const lane = @import("../../core/lane_projection.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");

const Ref = lane.Ref;
const R = state.chunk_rows;
const private = mtl.ResourceOptions.private | mtl.ResourceOptions.untracked;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Q = @import("weights.zig").Mlx;

pub const Scratch = struct {
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    /// Each row's last key on sliding layers: its position, or its image's last row (bidirectional vision blocks).
    hi: mtl.Buffer,
    h: [2]mtl.Buffer,
    normed: [2]mtl.Buffer,
    qkv: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    attn: mtl.Buffer,
    o: mtl.Buffer,
    hn: mtl.Buffer,
    n_mlp: mtl.Buffer,
    gate_up: mtl.Buffer,
    act: mtl.Buffer,
    y: mtl.Buffer,
    logits: mtl.Buffer,
    /// The drafter's tapped rows [rows, taps x hidden] bf16, once a drafter asked for them.
    taps: ?mtl.Buffer = null,

    pub fn init(device: mtl.Device, capacity: usize) !Scratch {
        const d = cfg.hidden;
        const qd = cfg.query_heads * cfg.global_dim;
        _ = capacity;
        var s: Scratch = undefined;
        s.taps = null;
        var made: usize = 0;
        errdefer s.release(made);
        inline for (.{ .{ "ids", R * 4, true }, .{ "pos", R * 4, true }, .{ "hi", R * 4, true }, .{ "qkv", R * (qd + 2 * 16 * 256) * 2, false }, .{ "q", R * qd * 2, false }, .{ "k", R * 16 * 256 * 2, false }, .{ "v", R * 16 * 256 * 2, false }, .{ "attn", R * qd * 2, false }, .{ "o", R * d * 2, false }, .{ "hn", R * d * 2, false }, .{ "n_mlp", R * d * 2, false }, .{ "gate_up", R * 2 * cfg.intermediate * 2, false }, .{ "act", R * cfg.intermediate * 2, false }, .{ "y", R * d * 2, false }, .{ "logits", cfg.vocab * 2, true } }) |f| {
            @field(s, f[0]) = try device.buffer(f[1], if (f[2]) shared else private);
            made += 1;
        }
        for ([_][]mtl.Buffer{ &s.h, &s.normed }) |pair| for (pair) |*b| {
            b.* = try device.buffer(R * d * 2, private);
            made += 1;
        };
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        s.release(std.math.maxInt(usize));
        if (s.taps) |b| b.deinit();
    }

    /// Releases the first `made` buffers in init's order, so a failed init frees what it made.
    fn release(s: *Scratch, made: usize) void {
        var i: usize = 0;
        inline for (.{ "ids", "pos", "hi", "qkv", "q", "k", "v", "attn", "o", "hn", "n_mlp", "gate_up", "act", "y", "logits" }) |f| {
            if (i < made) @field(s, f).deinit();
            i += 1;
        }
        for (s.h ++ s.normed) |b| {
            if (i < made) b.deinit();
            i += 1;
        }
    }
};

/// y[:, col..col + q.n] (row pitch `pitch`) = x q^T.
fn mm(m: *Model, e: mtl.ComputeEncoder, q: Q, x: mtl.Buffer, y: mtl.Buffer, col: usize, pitch: usize, rows: u32) void {
    affine_mm.denseBatch(e, m.mm, Ref{ .buf = x }, q, Ref{ .buf = y, .off = col * 2 }, rows, 1, .{ .x_row = @intCast(q.k), .y_row = @intCast(pitch) });
}

/// The glue's optional group sums off (the lane kernels' operand; affine_mm dequantizes its weights instead).
fn noSums(e: mtl.ComputeEncoder, s: *Scratch, index: usize) void {
    e.setBuffer(s.pos, 0, index);
    e.setValue(@as(i32, 0), index + 1);
}

fn bind(e: mtl.ComputeEncoder, bufs: []const mtl.Buffer, first: usize) void {
    for (bufs, first..) |b, i| e.setBuffer(b, 0, i);
}

/// An encoded image's rows in the prompt: positions [at, at + n), its rows from `rows` (5376 bf16 each).
pub const Span = struct { at: usize, n: u32, rows: Ref };

/// `ids` (at most 512) at the cache's length, `images`' rows over theirs (a span wholly in or out); s.logits: the last.
pub fn run(m: *Model, s: *Scratch, c: *state.Cache, ids: []const u32, images: []const Span) !void {
    const rows = ids.len;
    if (rows == 0 or rows > R) return error.BadChunk;
    if (c.len + rows > c.capacity) return error.ContextFull;
    const p0 = c.len;
    @memcpy(s.ids.slice(u32, rows), ids);
    for (s.pos.slice(i32, rows), s.hi.slice(i32, rows), 0..) |*p, *h, i| {
        p.* = @intCast(p0 + i);
        h.* = p.*;
    }
    for (images) |span| {
        if (span.at + span.n <= p0 or span.at >= p0 + rows) continue;
        if (span.at < p0 or span.at + span.n > p0 + rows) return error.ImageAcrossChunks;
        @memset(s.hi.slice(i32, rows)[span.at - p0 ..][0..span.n], @intCast(span.at + span.n - 1));
    }
    const ks = &m.kernels.sliding;
    const w = &m.weights;
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    const r32: u32 = @intCast(rows);
    e.setPipeline(ks.embed);
    bind(e, &.{s.ids}, 0);
    e.setBuffer(w.embed_w.buf, w.embed_w.off, 1);
    e.setBuffer(w.embed_s.buf, w.embed_s.off, 2);
    e.setBuffer(w.embed_b.buf, w.embed_b.off, 3);
    bind(e, &.{s.h[0]}, 4);
    e.setValue(fwd.embedScale(), 5);
    e.dispatchThreads(.{ .width = cfg.hidden / 2, .height = rows }, .{ .width = 256 });
    for (images) |span| if (span.at >= p0 and span.at < p0 + rows) {
        const v = m.vision orelse return error.NoGemmaVision;
        v.inject(e, span.rows, .{ .buf = s.h[0], .off = (span.at - p0) * cfg.hidden * 2 }, span.n);
    };
    e.setPipeline(ks.rms);
    bind(e, &.{s.h[0]}, 0);
    e.setBuffer(w.layers[0].input_norm.buf, w.layers[0].input_norm.off, 1);
    bind(e, &.{s.normed[0]}, 2);
    e.setValue(cfg.eps, 3);
    noSums(e, s, 4);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
    var cur: usize = 0;
    for (0..cfg.layers) |i| {
        try layer(m, e, s, c, i, cur, r32, .{});
        cur = 1 - cur;
    }
    // the head on the last row only (the lane kernel's one-row bits), soft-capped
    const head = try m.projections.get(m.gpa, m.device, w.head.n, w.head.k);
    try head.encode(e, .{ .buf = s.normed[cur], .off = (rows - 1) * cfg.hidden * 2 }, .{ .buf = w.head.w }, .{ .buf = w.head.sb }, .{ .buf = s.logits }, 1);
    e.setPipeline(ks.softcap);
    bind(e, &.{s.logits}, 0);
    e.setValue(cfg.softcap, 1);
    e.dispatchThreads(.{ .width = cfg.vocab }, .{ .width = 256 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("gemma4: prompt chunk failed: {s}", .{msg});
        return error.GpuFailed;
    }
    c.len += rows;
}

/// Which of a layer's dispatches to encode: a chunk takes them all; the bench times the stages apart.
pub const Stages = struct { mm: bool = true, attn: bool = true, glue: bool = true };

/// Layer `i` over `rows` rows of s.normed[cur] (keys at the cache's length) into s.h[1 - cur] and s.normed[1 - cur].
pub fn layer(m: *Model, e: mtl.ComputeEncoder, s: *Scratch, c: *state.Cache, i: usize, cur: usize, rows: u32, st: Stages) !void {
    const global = cfg.global(i);
    const kd: *const kernels.Kind = m.kernels.kind(global);
    const l = &m.weights.layers[i];
    const nk = cfg.kvHeads(i);
    const dh = cfg.headDim(i);
    const qn = cfg.query_heads * dh;
    const width = qn + nk * dh * @as(usize, if (global) 1 else 2);
    if (st.mm) {
        mm(m, e, l.mlx[0], s.normed[cur], s.qkv, 0, width, rows);
        mm(m, e, l.mlx[1], s.normed[cur], s.qkv, qn, width, rows);
        if (!global) mm(m, e, l.mlx[2], s.normed[cur], s.qkv, qn + nk * dh, width, rows);
    }
    if (st.glue) {
        e.setPipeline(kd.qkv);
        bind(e, &.{s.qkv}, 0);
        e.setBuffer(l.q_norm.buf, l.q_norm.off, 1);
        e.setBuffer(l.k_norm.buf, l.k_norm.off, 2);
        bind(e, &.{ if (global) m.inv_global else m.inv_sliding, s.pos }, 3);
        e.setValue(cfg.eps, 5);
        bind(e, &.{ s.q, s.k, s.v, c.k[i], c.v[i] }, 6); // keys and values straight into the cache too
        e.setValue([3]i32{ if (global) 0 else state.ring, @intCast(state.slotsOf(i, c.capacity)), 0 }, 11);
        e.dispatchGroups(.{ .width = cfg.query_heads + 2 * nk, .height = rows }, .{ .width = 32 });
    }
    if (st.attn) {
        e.setPipeline(kd.prompt_attn);
        bind(e, &.{ s.q, c.k[i], c.v[i], s.attn }, 0);
        e.setValue([3]i32{ @intCast(rows), @intCast(c.len), @intCast(state.slotsOf(i, c.capacity)) }, 4);
        bind(e, &.{s.hi}, 5);
        e.dispatchGroups(.{ .width = (rows + 31) / 32, .height = cfg.query_heads }, .{ .width = 32 * dh / 64 });
    }
    if (st.mm) {
        mm(m, e, l.mlx[3], s.attn, s.o, 0, cfg.hidden, rows);
    }
    if (st.glue) {
        e.setPipeline(kd.attn_tail);
        bind(e, &.{ s.h[cur], s.o }, 0);
        e.setBuffer(l.post_attention.buf, l.post_attention.off, 2);
        e.setBuffer(l.pre_feedforward.buf, l.pre_feedforward.off, 3);
        e.setValue(cfg.eps, 4);
        bind(e, &.{ s.hn, s.n_mlp }, 5);
        noSums(e, s, 7);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }
    if (st.mm) {
        mm(m, e, l.mlx[4], s.n_mlp, s.gate_up, 0, 2 * cfg.intermediate, rows);
        mm(m, e, l.mlx[5], s.n_mlp, s.gate_up, cfg.intermediate, 2 * cfg.intermediate, rows);
    }
    if (st.glue) {
        e.setPipeline(kd.geglu);
        bind(e, &.{ s.gate_up, s.act }, 0);
        e.setValue(@as(u32, cfg.intermediate), 2);
        noSums(e, s, 3);
        e.setValue(@as(i32, 0), 5);
        e.dispatchGroups(.{ .width = cfg.intermediate / 64, .height = rows }, .{ .width = 64 });
    }
    if (st.mm) {
        mm(m, e, l.mlx[6], s.act, s.y, 0, cfg.hidden, rows);
    }
    if (!st.glue) return;
    e.setPipeline(kd.dense_tail);
    bind(e, &.{ s.hn, s.y }, 0);
    e.setBuffer(l.post_feedforward.buf, l.post_feedforward.off, 2);
    e.setBuffer(l.scalar.buf, l.scalar.off, 3);
    const nx = if (i + 1 < cfg.layers) m.weights.layers[i + 1].input_norm else m.weights.final_norm;
    e.setBuffer(nx.buf, nx.off, 4);
    e.setValue(cfg.eps, 5);
    bind(e, &.{ s.h[1 - cur], s.normed[1 - cur] }, 6);
    noSums(e, s, 8);
    fwd.tapOut(e, m, s.taps, s.hn, i, 10);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
}
