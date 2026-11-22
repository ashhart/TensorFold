//! One layer's GPU dispatches: attention to the router, then post-MLP convolution and residual; stateful work per item.
const std = @import("std");
const mtl = @import("metal");
const View = @import("../../core/exl3_projection.zig").View;
const config = @import("config.zig");
const st = @import("metal_state.zig");
const wts = @import("metal_weights.zig");
const ops_mod = @import("metal_ops.zig");
const fused_ops_mod = @import("metal_fused_ops.zig");
const layer_fused = @import("metal_forward_layer_fused.zig");
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

fn fused(w: wts.Tensor, y: View) ops_mod.Fused {
    return .{ .w = tv(w), .y = y, .n = @intCast(w.shape[0]) };
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
    fused: *const fused_ops_mod.FusedOps,
    items: []const Item,
    base: [max_items]u32 = undefined,
    /// Rows of the pass over every item.
    n: u32 = 0,
    cb: mtl.CommandBuffer = undefined,
    e: mtl.ComputeEncoder = undefined,
    /// A command buffer is being encoded: its encoder is not ended and nothing was committed.
    open: bool = false,

    pub fn init(m: *Model, items: []const Item, fused_ops: *const fused_ops_mod.FusedOps) Run {
        var r: Run = .{ .m = m, .fused = fused_ops, .items = items };
        for (items, 0..) |it, k| {
            r.base[k] = r.n;
            r.n += @intCast(it.ids.len);
        }
        return r;
    }

    fn allItemsFusedConv(r: *const Run) bool {
        for (r.items) |it| {
            if (it.ids.len > fused_ops_mod.max_fused_conv_rows) return false;
        }
        return true;
    }

    pub fn begin(r: *Run) void {
        std.debug.assert(!r.open);
        r.cb = r.m.queue.commandBuffer();
        r.e = r.cb.compute(.concurrent);
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
        var has_images = false;
        for (r.items) |it| {
            if (it.images.len > 0) {
                has_images = true;
                break;
            }
        }
        if (has_images) {
            r.e.barrier();
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
        r.e.barrier();
    }

    /// Bank the previous pass before resetting the partial: prefix blocks stay; the first core slot holds the entry.
    pub fn enter(r: *Run, k: u32) !void {
        const m = r.m;
        const p = &m.plan;
        if (!p.restarts(k)) return;
        const s = &m.scratch;
        const h = r.hidden();
        const plane = @as(usize, r.n) * h * 2;
        try r.fused.enter(r.e, .{
            .src_blocks = if (p.bank_blocks != 0) v(s.blocks, (p.pre_blocks + 1) * plane) else v(s.blocks, p.pre_blocks * plane),
            .bank = v(s.bank, 0),
            .entry_block = v(s.blocks, p.pre_blocks * plane),
            .hid = v(s.hid(k), 0),
            .bank_h = v(s.bank_h, 0),
            .hidden = h,
            .rows = r.n,
            .bank_rows = p.bank_blocks * r.n,
        });
        r.e.barrier();
    }

    fn messagesAndGated(r: *Run, j: u32, nb: u32, partial: View, query: wts.Tensor, nq: u32, controls: []const wts.Controller, norms: [3]View, out: View) !void {
        const m = r.m;
        const s = &m.scratch;
        const h = r.hidden();
        const plane = r.n * h;
        const q = m.weights.cross.?;
        const row = @as(usize, j) * 2;
        const q1 = f32Row(query, if (nq == 3) 1 else 0, h);
        const q2 = f32Row(query, if (nq == 3) 2 else 0, h);
        try r.fused.dualLoopMix(r.e, .{
            .blocks0 = v(s.bank, 0),
            .partial0 = v(s.bank_h, 0),
            .queries0 = .{ f32Row(q.main, row, h), f32Row(q.kv, row, h), f32Row(q.kv, row + 1, h), f32Row(q.main, row + 1, h) },
            .nb0 = m.plan.bank_blocks,
            .nq0 = 4,
            .out0 = v(s.msg, 0),
            .blocks1 = v(s.blocks, 0),
            .partial1 = partial,
            .queries1 = .{ tv(query), q1, q2 },
            .nb1 = nb,
            .nq1 = nq,
            .out1 = v(s.mixf, 0),
            .block_stride = plane,
            .out_stride = plane,
            .rows = r.n,
            .hidden = h,
        });
        r.e.barrier();
        try r.gatedTail(nq, controls, norms, 0, out);
    }

    fn gated(r: *Run, nb: u32, partial: View, query: wts.Tensor, nq: u32, controls: []const wts.Controller, norms: [3]View, slot: u32, out: View) !void {
        const m = r.m;
        const s = &m.scratch;
        const t = &m.config.text;
        const h = t.hidden;
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
        r.e.barrier();
        try r.gatedTail(nq, controls, norms, slot, out);
    }

    fn gatedTail(r: *Run, nq: u32, controls: []const wts.Controller, norms: [3]View, slot: u32, out: View) !void {
        const m = r.m;
        const s = &m.scratch;
        const t = &m.config.text;
        const h = t.hidden;
        const rank = t.loop.?.rank;
        const plane = r.n * h;
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
        r.e.barrier();
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
        const relw: usize = @as(usize, hd.q) * t.d_rel;
        const nq: u32 = spec.attn_queries;
        const plane: usize = @as(usize, n) * h; // elements of one block / query output over the pass's rows

        // the attention input: layer 0 norms the embedding, later layers mix the finished blocks with the running state
        if (visit.messages) {
            const norms: [3]View = .{ tv(a.norms[0].?), tv(a.norms[1] orelse a.norms[0].?), tv(a.norms[2] orelse a.norms[0].?) };
            try r.messagesAndGated(visit.core_index, blocks_in, v(hin, 0), L.attn_query.?, nq, L.controllers.?[0..3], norms, v(s.xin, 0));
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
        e.barrier();
        const xq = v(s.xin, 0);
        const separate_qkv = global or visit.messages;
        const xk = if (separate_qkv) v(s.xin, plane * 2) else xq;
        const xv = if (separate_qkv) v(s.xin, 2 * plane * 2) else xq;

        if (n <= 16 and !separate_qkv) {
            try m.ops.linearFusedRows(e, xq, n, h, &.{
                fused(a.wq, v(s.q, 0)),
                fused(a.wk, v(s.kraw, 0)),
                fused(a.wv, v(s.vraw, 0)),
                fused(a.wr, v(s.rel, 0)),
                fused(a.gate, v(s.gate, 0)),
                fused(a.sink, v(s.sink, 0)),
            });
        } else if (n <= 16) {
            try m.ops.linearFusedRows(e, xq, n, h, &.{
                fused(a.wq, v(s.q, 0)),
                fused(a.wr, v(s.rel, 0)),
                fused(a.gate, v(s.gate, 0)),
                fused(a.sink, v(s.sink, 0)),
            });
            if (global) {
                try m.ops.linearFusedRows(e, xk, n, h, &.{ fused(a.wk, v(s.kraw, 0)), fused(a.k_taps.?, v(s.dynk, 0)) });
                try m.ops.linearFusedRows(e, xv, n, h, &.{ fused(a.wv, v(s.vraw, 0)), fused(a.v_taps.?, v(s.dynv, 0)) });
            } else {
                try m.ops.linearFusedRows(e, xk, n, h, &.{fused(a.wk, v(s.kraw, 0))});
                try m.ops.linearFusedRows(e, xv, n, h, &.{fused(a.wv, v(s.vraw, 0))});
            }
        } else {
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
        }
        e.barrier();

        const fused_conv = r.allItemsFusedConv();
        try layer_fused.qkvPrep(r, k, i, global, fused_conv);

        // Wave 3: attention per item; single-chunk items overlap, multi-chunk items serialize on ops.scratch.
        var scratch_busy = false;
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const pos0 = it.state.len;
            const kv = it.state.kv[k];
            const kb = st.kvBytes(t, i, it.state.geometry);
            const ring = st.ring(t, i, it.state.geometry);
            const last: u64 = @as(u64, pos0) + rows;
            const keys: u64 = if (global) last else @min(last, t.window);
            const uses_scratch = keys > ops_mod.attn_chunk;
            if (uses_scratch and scratch_busy) e.barrier();
            try m.ops.attention(e, .{
                .q = rows16(s.q, b, qw),
                .rel = rows16(s.rel, b, relw),
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
            if (uses_scratch) scratch_busy = true;
        }
        e.barrier();

        if (n <= 16) {
            try m.ops.linearFusedRows(e, v(s.att, 0), n, @intCast(qw), &.{fused(a.wo, v(s.o, 0))});
        } else {
            try r.lin(v(s.att, 0), a.wo, v(s.o, 0), n);
        }
        e.barrier();

        // post-attention convolution, then residual: a block boundary keeps only the attention output as partial
        const boundary = spec.boundary;
        try layer_fused.postAttnConvResidual(r, k, i, boundary, fused_conv);

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
        e.barrier();
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
        e.barrier();
    }

    /// Layer `i` after its experts: the post-MLP convolution and the residual into the next layer's hidden stream.
    pub fn finish(r: *Run, k: u32) !void {
        const i = r.m.plan.visit(k).layer;
        try layer_fused.postMlpConvResidual(r, k, i, r.allItemsFusedConv());
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
        e.barrier();
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
            e.barrier();
            break :gather v(s.xl, 0);
        };
        const vocab = t.unpadded_vocab;
        try m.ops.linear(e, .{ .x = x, .w = tv(m.weights.unembed), .y = v(s.ylogit, 0), .rows = picked, .n = vocab, .k = h });
        e.barrier();
        try m.ops.softcap(e, .{ .y = v(s.ylogit, 0), .out = v(m.logits, 0), .rows = picked, .n = vocab, .ldy = vocab, .cap = t.logit_cap });
    }
};
