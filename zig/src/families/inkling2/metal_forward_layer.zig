//! One layer's GPU dispatches: attention to the router, then post-MLP convolution and residual; stateful work per item.
const std = @import("std");
const mtl = @import("metal");
const View = @import("../../core/exl3_projection.zig").View;
const config = @import("config.zig");
const st = @import("metal_state.zig");
const wts = @import("metal_weights.zig");
const ops_mod = @import("metal_ops.zig");
const Model = @import("metal_model.zig").Model;

comptime {
    std.debug.assert(st.hist_ring == ops_mod.conv_ring);
}

/// Items one pass takes at most (streams sharing a forward).
pub const max_items = 64;

/// A stream's rows: `ids` appended at `state.len`.
pub const Item = struct {
    state: *st.State,
    ids: []const u32,
    images: []const ImageRows = &.{},
    /// Trailing rows of this item that get logits (0: none).
    logit_rows: u32 = 1,
    /// Capture the draft taps of every row of this item.
    taps: bool = false,
};

/// Image-tower rows substituted for the embeddings of `ids[row .. row + rows]` (raw projector output, BF16 [rows, H]).
pub const ImageRows = struct { row: u32, rows: u32, buf: mtl.Buffer, off: usize = 0 };

fn v(b: mtl.Buffer, off: usize) View {
    return .{ .buffer = b, .offset = off };
}

fn tv(t: wts.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}

/// Rows [first, first + rows) of a bf16 buffer of `width` columns.
fn rows16(b: mtl.Buffer, first: usize, width: usize) View {
    return v(b, first * width * 2);
}

fn f32Row(t: wts.Tensor, i: usize, width: u32) View {
    return v(t.buffer, t.offset + i * width * 4);
}

/// A pass over `items`: the encoder of the command buffer being built and where each item's rows start.
pub const Run = struct {
    m: *Model,
    items: []const Item,
    base: [max_items]u32 = undefined,
    /// Rows of the pass over every item.
    n: u32 = 0,
    cb: mtl.CommandBuffer = undefined,
    e: mtl.ComputeEncoder = undefined,
    /// A command buffer is being encoded: its encoder is not ended and nothing was committed.
    open: bool = false,

    pub fn init(m: *Model, items: []const Item) Run {
        var r: Run = .{ .m = m, .items = items };
        for (items, 0..) |it, k| {
            r.base[k] = r.n;
            r.n += @intCast(it.ids.len);
        }
        return r;
    }

    pub fn begin(r: *Run) void {
        std.debug.assert(!r.open);
        r.cb = r.m.queue.commandBuffer();
        r.e = r.cb.compute(.serial);
        r.open = true;
    }

    /// End the encoder, run the command buffer and wait for it.
    pub fn flush(r: *Run) !void {
        std.debug.assert(r.open);
        r.e.end();
        r.open = false;
        r.cb.commit();
        r.cb.wait();
        if (r.cb.failure()) |msg| {
            std.log.err("inkling2 forward: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    /// Drop an uncommitted command buffer on encoding failure.
    pub fn abort(r: *Run) void {
        if (!r.open) return;
        r.e.end();
        r.open = false;
    }

    fn hidden(r: *const Run) u32 {
        return r.m.config.text.hidden;
    }

    fn lin(r: *Run, x: View, w: wts.Tensor, y: View, rows: u32) !void {
        try r.m.ops.linear(r.e, .{ .x = x, .w = tv(w), .y = y, .rows = rows, .n = @intCast(w.shape[0]), .k = @intCast(w.shape[1]) });
    }

    /// A tap is the partial entering a logical visit after any restart, or the final normalized state.
    pub fn capture(r: *Run, entry: u32) !void {
        if (r.m.taps == null) return;
        const mtp = &r.m.config.mtp.?;
        const ti = std.mem.indexOfScalar(u32, mtp.taps[0..mtp.taps_n], entry) orelse return;
        const taps = r.m.taps.?;
        const src = if (entry == r.m.plan.n_visits) r.m.scratch.xf else r.m.scratch.hid(entry);
        const h = r.hidden();
        var row: u32 = 0;
        for (r.items, 0..) |it, k| {
            if (!it.taps) continue;
            const n: u32 = @intCast(it.ids.len);
            try r.m.ops.copyRows(r.e, .{
                .src = rows16(src, r.base[k], h),
                .src_ld = h,
                .dst = v(taps, (@as(usize, row) * mtp.taps_n + ti) * h * 2),
                .dst_ld = mtp.taps_n * h,
                .rows = n,
                .cols = h,
            });
            row += n;
        }
    }

    /// Embeddings of every row, image rows substituted.
    pub fn embed(r: *Run) !void {
        const m = r.m;
        const h = r.hidden();
        try m.ops.embed(r.e, .{
            .ids = v(m.scratch.ids, 0),
            .table = tv(m.weights.embed),
            .norm = tv(m.weights.embed_norm),
            .out = v(m.scratch.hid(0), 0),
            .rows = r.n,
            .vocab = m.config.text.vocab,
            .hidden = h,
            .eps = m.config.text.eps,
        });
        for (r.items, 0..) |it, k| for (it.images) |img| {
            try m.ops.copyRows(r.e, .{
                .src = v(img.buf, img.off),
                .src_ld = h,
                .dst = rows16(m.scratch.hid(0), r.base[k] + img.row, h),
                .dst_ld = h,
                .rows = img.rows,
                .cols = h,
            });
        };
    }

    /// Bank the previous pass before resetting the partial: prefix blocks stay; the first core slot holds the entry.
    pub fn enter(r: *Run, k: u32) !void {
        const m = r.m;
        const p = &m.plan;
        if (!p.restarts(k)) return;
        const s = &m.scratch;
        const h = r.hidden();
        const plane = @as(usize, r.n) * h * 2;
        if (p.bank_blocks != 0) try r.copy(v(s.blocks, (p.pre_blocks + 1) * plane), v(s.bank, 0), p.bank_blocks * r.n);
        try r.copy(v(s.hid(k), 0), v(s.bank_h, 0), r.n);
        try r.copy(v(s.blocks, p.pre_blocks * plane), v(s.hid(k), 0), r.n);
    }

    fn copy(r: *Run, src: View, dst: View, rows: u32) !void {
        const h = r.hidden();
        try r.m.ops.copyRows(r.e, .{ .src = src, .dst = dst, .src_ld = h, .dst_ld = h, .rows = rows, .cols = h });
    }

    fn messages(r: *Run, j: u32) !void {
        const m = r.m;
        const s = &m.scratch;
        const h = r.hidden();
        const plane = r.n * h;
        const q = m.weights.cross.?;
        const row = @as(usize, j) * 2;
        try m.loop_ops.mix(r.e, .{
            .blocks = v(s.bank, 0),
            .block_stride = plane,
            .nb = m.plan.bank_blocks,
            .partial = v(s.bank_h, 0),
            .partial_f32 = false,
            .queries = .{ f32Row(q.main, row, h), f32Row(q.kv, row, h), f32Row(q.kv, row + 1, h), f32Row(q.main, row + 1, h) },
            .nq = 4,
            .out = v(s.msg, 0),
            .out_stride = plane,
            .rows = r.n,
            .hidden = h,
        });
    }

    fn gated(r: *Run, nb: u32, partial: View, query: wts.Tensor, nq: u32, controls: []const wts.Controller, norms: [3]View, slot: u32, out: View) !void {
        const m = r.m;
        const s = &m.scratch;
        const t = &m.config.text;
        const h = t.hidden;
        const rank = t.loop.?.rank;
        const plane = r.n * h;
        const q1 = f32Row(query, if (nq == 3) 1 else 0, h);
        const q2 = f32Row(query, if (nq == 3) 2 else 0, h);
        try m.loop_ops.mix(r.e, .{
            .blocks = v(s.blocks, 0),
            .block_stride = plane,
            .nb = nb,
            .partial = partial,
            .partial_f32 = slot == 3,
            .queries = .{ tv(query), q1, q2, tv(query) },
            .nq = nq,
            .out = v(s.mixf, 0),
            .out_stride = plane,
            .rows = r.n,
            .hidden = h,
        });
        const plane_bytes = @as(usize, plane) * 4;
        const states: [3]View = .{ v(s.mixf, 0), v(s.mixf, if (nq == 3) plane_bytes else 0), v(s.mixf, if (nq == 3) 2 * plane_bytes else 0) };
        const gates: u32 = @intCast(controls.len);
        const slots: [3]u32 = if (slot == 3) .{ 3, 3, 3 } else .{ 0, 1, 2 };
        const msg = v(s.msg, 0);
        const down = v(s.down, 0);
        var dw: [3]View = undefined;
        var forget: [3]View = undefined;
        var gain: [3]View = undefined;
        for (0..3) |i| {
            const c = &controls[if (i < gates) i else 0];
            dw[i] = tv(c.down);
            forget[i] = tv(c.forget);
            gain[i] = tv(c.gain);
        }
        try m.loop_ops.gateDown(r.e, .{
            .h = states,
            .msg = msg,
            .msg_stride = plane,
            .slots = slots,
            .gates = gates,
            .down_w = dw,
            .down = down,
            .rows = r.n,
            .hidden = h,
            .rank = rank,
            .eps_unweighted = t.eps,
        });
        try m.loop_ops.gateMix(r.e, .{
            .h = states,
            .msg = msg,
            .msg_stride = plane,
            .slots = slots,
            .gates = gates,
            .down = down,
            .forget = forget,
            .gain = gain,
            .norms = norms,
            .out = out,
            .out_stride = plane,
            .rows = r.n,
            .hidden = h,
            .rank = rank,
            .eps = t.eps,
        });
    }

    /// Layer `i` from its input to the router's picks (scratch.rids / rwts) and the MLP input (scratch.xm).
    pub fn attend(r: *Run, k: u32) !void {
        const m = r.m;
        const t = &m.config.text;
        const s = &m.scratch;
        const visit = m.plan.visit(k);
        const i = visit.layer;
        const spec = m.plan.at(i);
        const L = &m.weights.layers[i];
        const a = &L.attn;
        const e = r.e;
        const h: u32 = t.hidden;
        const n = r.n;
        const hd = spec.heads;
        const global = spec.kind == .global;
        const hin = s.hid(k);
        const blocks_in = spec.blocks_in;
        const qw: usize = @as(usize, hd.q) * hd.dim;
        const kvw: usize = @as(usize, hd.kv) * hd.dim;
        const relw: usize = @as(usize, hd.q) * t.d_rel;
        const taps: usize = @as(usize, hd.kv) * config.conv_taps;
        const nq: u32 = spec.attn_queries;
        const plane: usize = @as(usize, n) * h; // elements of one block / query output over the pass's rows

        // the attention input: layer 0 norms the embedding, later layers mix the finished blocks with the running state
        if (visit.messages) {
            try r.messages(visit.core_index);
            const norms: [3]View = .{ tv(a.norms[0].?), tv(a.norms[1] orelse a.norms[0].?), tv(a.norms[2] orelse a.norms[0].?) };
            try r.gated(blocks_in, v(hin, 0), L.attn_query.?, nq, L.controllers.?[0..3], norms, 0, v(s.xin, 0));
        } else if (i == 0) {
            try m.ops.rmsNorm(e, .{ .x = v(hin, 0), .w = tv(a.norms[0].?), .out = v(s.xin, 0), .rows = n, .hidden = h, .eps = t.eps });
        } else {
            var norms: [3]View = undefined;
            for (&norms, a.norms) |*o, w| o.* = if (w) |tensor| tv(tensor) else tv(a.norms[0].?);
            try m.ops.attnRes(e, .{
                .blocks = v(s.blocks, 0),
                .block_stride = @intCast(plane),
                .nb = blocks_in,
                .partial = v(hin, 0),
                .partial_f32 = false,
                .queries = tv(L.attn_query.?),
                .nq = nq,
                .norms = norms,
                .out = v(s.xin, 0),
                .out_stride = @intCast(plane),
                .rows = n,
                .hidden = h,
                .eps = t.eps,
            });
        }
        const xq = v(s.xin, 0);
        const separate_qkv = global or visit.messages;
        const xk = if (separate_qkv) v(s.xin, plane * 2) else xq;
        const xv = if (separate_qkv) v(s.xin, 2 * plane * 2) else xq;
        try r.lin(xq, a.wq, v(s.q, 0), n);
        try r.lin(xk, a.wk, v(s.kraw, 0), n);
        try r.lin(xv, a.wv, v(s.vraw, 0), n);
        try r.lin(xq, a.wr, v(s.rel, 0), n);
        try r.lin(xq, a.gate, v(s.gate, 0), n);
        try r.lin(xq, a.sink, v(s.sink, 0), n);
        if (global) {
            try r.lin(xk, a.k_taps.?, v(s.dynk, 0), n);
            try r.lin(xv, a.v_taps.?, v(s.dynv, 0), n);
        }

        const hl = st.hist(t, i);
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const pos0 = it.state.len;
            const hist = it.state.hist[k];
            const kv = it.state.kv[k];
            const kb = st.kvBytes(t, i, it.state.geometry);
            const ring = st.ring(t, i, it.state.geometry);
            const log: ?ops_mod.Tau = if (global) .{ .pos0 = pos0, .alpha = t.log_alpha, .floor = t.log_floor } else null;
            inline for (.{ .{ s.kraw, s.dynk, s.kc, a.k_conv, hl.k }, .{ s.vraw, s.dynv, s.vc, a.v_conv, hl.v } }) |side| {
                const raw = rows16(side[0], b, kvw);
                if (global) {
                    try m.ops.convDynamic(e, .{
                        .x = raw,
                        .dyn = rows16(side[1], b, taps),
                        .static_w = tv(side[3]),
                        .hist = v(hist, side[4]),
                        .out = rows16(side[2], b, kvw),
                        .rows = rows,
                        .channels = @intCast(kvw),
                        .kv_heads = hd.kv,
                        .pos0 = pos0,
                    });
                } else {
                    try m.ops.convStatic(e, .{ .x = raw, .hist = v(hist, side[4]), .w = tv(side[3]), .out = rows16(side[2], b, kvw), .rows = rows, .channels = @intCast(kvw), .pos0 = pos0 });
                }
                try m.ops.convStore(e, .{ .x = raw, .hist = v(hist, side[4]), .rows = rows, .channels = @intCast(kvw), .pos0 = pos0 });
            }
            // query and relative states: scaled L2 clip, then the global layers' context scaling (in place)
            const q = rows16(s.q, b, qw);
            const rel = rows16(s.rel, b, relw);
            try m.ops.qPrep(e, .{ .x = q, .scale = tv(a.q_scale), .out = q, .rows = rows, .heads = hd.q, .dim = hd.dim, .threshold = t.qk_clip, .tau = log });
            try m.ops.relPrep(e, .{ .x = rel, .scale = tv(a.rel_scale), .out = rel, .rows = rows, .heads = hd.q, .dim = t.d_rel, .threshold = t.rel_clip, .tau = log });
            try m.ops.kvPrep(e, .{ .k = rows16(s.kc, b, kvw), .v = rows16(s.vc, b, kvw), .kcache = v(kv, 0), .vcache = v(kv, kb), .rows = rows, .kv_heads = hd.kv, .dim = hd.dim, .pos0 = pos0, .ring = ring, .threshold = t.qk_clip, .eps = t.eps });
            try m.ops.attention(e, .{
                .q = q,
                .rel = rel,
                .sink = rows16(s.sink, b, hd.q),
                .gate = rows16(s.gate, b, qw),
                .kcache = v(kv, 0),
                .vcache = v(kv, kb),
                .proj = tv(a.rel_proj),
                .out = rows16(s.att, b, qw),
                .rows = rows,
                .pos0 = pos0,
                .heads = hd.q,
                .kv_heads = hd.kv,
                .dim = hd.dim,
                .d_rel = t.d_rel,
                .extent = spec.extent,
                .window = if (global) 0 else t.window,
                .ring = ring,
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd.dim))),
            });
        }
        try r.lin(v(s.att, 0), a.wo, v(s.o, 0), n);

        // post-attention convolution, then residual: a block boundary keeps only the attention output as partial
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const hist = it.state.hist[k];
            try m.ops.convStatic(e, .{ .x = rows16(s.o, b, h), .hist = v(hist, hl.attn), .w = tv(L.attn_conv), .out = rows16(s.a, b, h), .rows = rows, .channels = h, .pos0 = it.state.len });
            try m.ops.convStore(e, .{ .x = rows16(s.o, b, h), .hist = v(hist, hl.attn), .rows = rows, .channels = h, .pos0 = it.state.len });
        }
        const boundary = spec.boundary;
        try m.ops.residualAttn(e, .{ .conv = v(s.a, 0), .block = if (boundary) null else v(hin, 0), .out = v(s.h32, 0), .rows = n, .cols = h });
        // The repeat's first boundary joins the same saved core entry already in this slot.
        if (boundary and !m.plan.restarts(k)) try r.copy(v(hin, 0), v(s.blocks, @as(usize, i / t.block) * plane * 2), n);

        const mnorm: [3]View = .{ tv(L.mlp_norm), tv(L.mlp_norm), tv(L.mlp_norm) };
        if (visit.messages) {
            try r.gated(blocks_in + @intFromBool(boundary), v(s.h32, 0), L.mlp_query, 1, L.controllers.?[3..4], mnorm, 3, v(s.xm, 0));
        } else try m.ops.attnRes(e, .{
            .blocks = v(s.blocks, 0),
            .block_stride = @intCast(plane),
            .nb = blocks_in + @intFromBool(boundary),
            .partial = v(s.h32, 0),
            .partial_f32 = true,
            .queries = tv(L.mlp_query),
            .nq = 1,
            .norms = mnorm,
            .out = v(s.xm, 0),
            .out_stride = @intCast(plane),
            .rows = n,
            .hidden = h,
            .eps = t.eps,
        });
        try m.ops.route(e, .{
            .x = v(s.xm, 0),
            .w = tv(L.router),
            .bias = tv(L.router_bias),
            .ids = v(s.rids, 0),
            .weights = v(s.rwts, 0),
            .rows = n,
            .hidden = h,
            .experts = t.experts,
            .topk = t.topk,
            .scale = t.route_scale,
        });
    }

    /// Layer `i` after its experts: the post-MLP convolution and the residual into the next layer's hidden stream.
    pub fn finish(r: *Run, k: u32) !void {
        const m = r.m;
        const t = &m.config.text;
        const s = &m.scratch;
        const i = m.plan.visit(k).layer;
        const h: u32 = t.hidden;
        const hl = st.hist(t, i);
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const hist = it.state.hist[k];
            try m.ops.convStatic(r.e, .{ .x = rows16(s.moe, b, h), .hist = v(hist, hl.mlp), .w = tv(m.weights.layers[i].mlp_conv), .out = rows16(s.c, b, h), .rows = rows, .channels = h, .pos0 = it.state.len });
            try m.ops.convStore(r.e, .{ .x = rows16(s.moe, b, h), .hist = v(hist, hl.mlp), .rows = rows, .channels = h, .pos0 = it.state.len });
        }
        try m.ops.residualMlp(r.e, .{ .h = v(s.h32, 0), .conv = v(s.c, 0), .out = v(s.hid(k + 1), 0), .rows = r.n, .cols = h });
    }

    /// The final attention-residual mix and norm over every row, the final-norm taps, then each item's last-row logits.
    pub fn head(r: *Run) !void {
        const m = r.m;
        const t = &m.config.text;
        const s = &m.scratch;
        const h: u32 = t.hidden;
        const n = r.n;
        const e = r.e;
        const blocks = m.plan.final_sources - 1;
        const norm = tv(m.weights.norm);
        try m.ops.attnRes(e, .{
            .blocks = v(s.blocks, 0),
            .block_stride = @intCast(@as(usize, n) * h),
            .nb = blocks,
            .partial = v(s.hid(m.plan.n_visits), 0),
            .partial_f32 = false,
            .queries = tv(m.weights.output_query),
            .nq = 1,
            .norms = .{ norm, norm, norm },
            .out = v(s.xf, 0),
            .out_stride = @intCast(@as(usize, n) * h),
            .rows = n,
            .hidden = h,
            .eps = t.eps,
        });
        try r.capture(m.plan.n_visits);
        var picked: u32 = 0;
        var first: u32 = undefined;
        var end: u32 = 0;
        for (r.items, 0..) |it, k| {
            if (it.logit_rows == 0) continue;
            if (picked == 0) first = r.base[k] + @as(u32, @intCast(it.ids.len)) - it.logit_rows;
            end = r.base[k] + @as(u32, @intCast(it.ids.len));
            picked += it.logit_rows;
        }
        if (picked == 0) return;
        const x = if (end - first == picked) rows16(s.xf, first, h) else gather: {
            var at: u32 = 0;
            for (r.items, 0..) |it, k| {
                if (it.logit_rows == 0) continue;
                const start = r.base[k] + @as(u32, @intCast(it.ids.len)) - it.logit_rows;
                try m.ops.copyRows(e, .{ .src = rows16(s.xf, start, h), .src_ld = h, .dst = rows16(s.xl, at, h), .dst_ld = h, .rows = it.logit_rows, .cols = h });
                at += it.logit_rows;
            }
            break :gather v(s.xl, 0);
        };
        const vocab = t.unpadded_vocab;
        try m.ops.linear(e, .{ .x = x, .w = tv(m.weights.unembed), .y = v(s.ylogit, 0), .rows = picked, .n = vocab, .k = h });
        try m.ops.softcap(e, .{ .y = v(s.ylogit, 0), .out = v(m.logits, 0), .rows = picked, .n = vocab, .ldy = vocab, .cap = t.logit_cap });
    }
};
