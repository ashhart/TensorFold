//! Fused and prefill conv/prep/residual helpers for Inkling 2's Metal layer pass.
const mtl = @import("metal");
const View = @import("../../core/exl3_projection.zig").View;
const config = @import("config.zig");
const st = @import("metal_state.zig");
const wts = @import("metal_weights.zig");
const ops_mod = @import("metal_ops.zig");
const layer = @import("metal_forward_layer.zig");

fn v(b: mtl.Buffer, off: usize) View {
    return .{ .buffer = b, .offset = off };
}

fn tv(t: wts.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}

fn rows16(b: mtl.Buffer, first: usize, width: usize) View {
    return v(b, first * width * 2);
}

/// Fused or prefill Q/Rel prep, K/V convolution, history store, and KV cache prep.
pub fn qkvPrep(r: *layer.Run, k: u32, i: u32, global: bool, fused_conv: bool) !void {
    const m = r.m;
    const t = &m.config.text;
    const s = &m.scratch;
    const e = r.e;
    const spec = m.plan.at(i);
    const a = &m.weights.layers[i].attn;
    const hd = spec.heads;
    const qw: usize = @as(usize, hd.q) * hd.dim;
    const kvw: usize = @as(usize, hd.kv) * hd.dim;
    const relw: usize = @as(usize, hd.q) * t.d_rel;
    const taps: usize = @as(usize, hd.kv) * config.conv_taps;
    const hl = st.hist(t, i);

    if (fused_conv) {
        // Fused Wave 1+2: qPrep + relPrep + K/V conv + K/V convStore + kvPrep in 1 dispatch and 1 barrier.
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const pos0 = it.state.len;
            const hist = it.state.hist[k];
            const kv = it.state.kv[k];
            const kb = st.kvBytes(t, i, it.state.geometry);
            const ring = st.ring(t, i, it.state.geometry);
            const log: ?ops_mod.Tau = if (global) .{ .pos0 = pos0, .alpha = t.log_alpha, .floor = t.log_floor } else null;
            if (global) {
                try r.fused.qkvPrepDynamic(e, .{
                    .q = rows16(s.q, b, qw),
                    .q_scale = tv(a.q_scale),
                    .rel = rows16(s.rel, b, relw),
                    .rel_scale = tv(a.rel_scale),
                    .kraw = rows16(s.kraw, b, kvw),
                    .vraw = rows16(s.vraw, b, kvw),
                    .khist = v(hist, hl.k),
                    .vhist = v(hist, hl.v),
                    .kw = tv(a.k_conv),
                    .vw = tv(a.v_conv),
                    .kcache = v(kv, 0),
                    .vcache = v(kv, kb),
                    .dynk = rows16(s.dynk, b, taps),
                    .dynv = rows16(s.dynv, b, taps),
                    .rows = rows,
                    .q_heads = hd.q,
                    .kv_heads = hd.kv,
                    .dim = hd.dim,
                    .rel_dim = t.d_rel,
                    .pos0 = pos0,
                    .kv_ring = ring,
                    .qk_threshold = t.qk_clip,
                    .rel_threshold = t.rel_clip,
                    .eps = t.eps,
                    .tau = log,
                });
            } else {
                try r.fused.qkvPrepStatic(e, .{
                    .q = rows16(s.q, b, qw),
                    .q_scale = tv(a.q_scale),
                    .rel = rows16(s.rel, b, relw),
                    .rel_scale = tv(a.rel_scale),
                    .kraw = rows16(s.kraw, b, kvw),
                    .vraw = rows16(s.vraw, b, kvw),
                    .khist = v(hist, hl.k),
                    .vhist = v(hist, hl.v),
                    .kw = tv(a.k_conv),
                    .vw = tv(a.v_conv),
                    .kcache = v(kv, 0),
                    .vcache = v(kv, kb),
                    .rows = rows,
                    .q_heads = hd.q,
                    .kv_heads = hd.kv,
                    .dim = hd.dim,
                    .rel_dim = t.d_rel,
                    .pos0 = pos0,
                    .kv_ring = ring,
                    .qk_threshold = t.qk_clip,
                    .rel_threshold = t.rel_clip,
                    .eps = t.eps,
                    .tau = log,
                });
            }
        }
        e.barrier();
        return;
    }

    // Prefill path (rows > max_fused_conv_rows): K/V convs and fused q/rel prep, then convStore and kvPrep.
    for (r.items, 0..) |it, idx| {
        const b: usize = r.base[idx];
        const rows: u32 = @intCast(it.ids.len);
        const pos0 = it.state.len;
        const hist = it.state.hist[k];
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
        }
        try r.fused.qRelPrep(e, .{
            .q = rows16(s.q, b, qw),
            .q_scale = tv(a.q_scale),
            .rel = rows16(s.rel, b, relw),
            .rel_scale = tv(a.rel_scale),
            .rows = rows,
            .q_heads = hd.q,
            .dim = hd.dim,
            .rel_dim = t.d_rel,
            .qk_threshold = t.qk_clip,
            .rel_threshold = t.rel_clip,
            .tau = log,
        });
    }
    e.barrier();

    for (r.items, 0..) |it, idx| {
        const b: usize = r.base[idx];
        const rows: u32 = @intCast(it.ids.len);
        const pos0 = it.state.len;
        const hist = it.state.hist[k];
        const kv = it.state.kv[k];
        const kb = st.kvBytes(t, i, it.state.geometry);
        const ring = st.ring(t, i, it.state.geometry);
        inline for (.{ .{ s.kraw, hl.k }, .{ s.vraw, hl.v } }) |side| {
            try m.ops.convStore(e, .{ .x = rows16(side[0], b, kvw), .hist = v(hist, side[1]), .rows = rows, .channels = @intCast(kvw), .pos0 = pos0 });
        }
        try m.ops.kvPrep(e, .{ .k = rows16(s.kc, b, kvw), .v = rows16(s.vc, b, kvw), .kcache = v(kv, 0), .vcache = v(kv, kb), .rows = rows, .kv_heads = hd.kv, .dim = hd.dim, .pos0 = pos0, .ring = ring, .threshold = t.qk_clip, .eps = t.eps });
    }
    e.barrier();
}

/// Post-attention convolution, history store, block snapshot, and residual into `scratch.h32`.
pub fn postAttnConvResidual(r: *layer.Run, k: u32, i: u32, boundary: bool, fused_conv: bool) !void {
    const m = r.m;
    const t = &m.config.text;
    const s = &m.scratch;
    const e = r.e;
    const h: u32 = t.hidden;
    const n = r.n;
    const plane: usize = @as(usize, n) * h;
    const hin = s.hid(k);
    const L = &m.weights.layers[i];
    const hl = st.hist(t, i);

    if (fused_conv) {
        const save_block = boundary and !m.plan.restarts(k);
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const hist = it.state.hist[k];
            try r.fused.convStoreResidualAttn(e, .{
                .x = rows16(s.o, b, h),
                .hist = v(hist, hl.attn),
                .w = tv(L.attn_conv),
                .hin = rows16(hin, b, h),
                .block_dst = if (save_block) rows16(s.blocks, @as(usize, i / t.block) * n + b, h) else null,
                .out = v(s.h32, b * h * 4),
                .rows = rows,
                .channels = h,
                .pos0 = it.state.len,
                .has_block = !boundary,
            });
        }
        e.barrier();
        return;
    }

    for (r.items, 0..) |it, idx| {
        const b: usize = r.base[idx];
        const rows: u32 = @intCast(it.ids.len);
        const hist = it.state.hist[k];
        try m.ops.convStatic(e, .{ .x = rows16(s.o, b, h), .hist = v(hist, hl.attn), .w = tv(L.attn_conv), .out = rows16(s.a, b, h), .rows = rows, .channels = h, .pos0 = it.state.len });
    }
    // The repeat's first boundary joins the same saved core entry already in this slot.
    if (boundary and !m.plan.restarts(k)) {
        try m.ops.copyRows(e, .{ .src = v(hin, 0), .dst = v(s.blocks, @as(usize, i / t.block) * plane * 2), .src_ld = h, .dst_ld = h, .rows = n, .cols = h });
    }
    e.barrier();
    for (r.items, 0..) |it, idx| {
        const b: usize = r.base[idx];
        const rows: u32 = @intCast(it.ids.len);
        const hist = it.state.hist[k];
        try m.ops.convStore(e, .{ .x = rows16(s.o, b, h), .hist = v(hist, hl.attn), .rows = rows, .channels = h, .pos0 = it.state.len });
    }
    try m.ops.residualAttn(e, .{ .conv = v(s.a, 0), .block = if (boundary) null else v(hin, 0), .out = v(s.h32, 0), .rows = n, .cols = h });
    e.barrier();
}

/// Post-MLP convolution, history store, and residual into the next visit's hidden buffer.
pub fn postMlpConvResidual(r: *layer.Run, k: u32, i: u32, fused_conv: bool) !void {
    const m = r.m;
    const t = &m.config.text;
    const s = &m.scratch;
    const h: u32 = t.hidden;
    const hl = st.hist(t, i);

    if (fused_conv) {
        for (r.items, 0..) |it, idx| {
            const b: usize = r.base[idx];
            const rows: u32 = @intCast(it.ids.len);
            const hist = it.state.hist[k];
            try r.fused.convStoreResidualMlp(r.e, .{
                .x = rows16(s.moe, b, h),
                .hist = v(hist, hl.mlp),
                .w = tv(m.weights.layers[i].mlp_conv),
                .h32 = v(s.h32, b * h * 4),
                .out = rows16(s.hid(k + 1), b, h),
                .rows = rows,
                .channels = h,
                .pos0 = it.state.len,
            });
        }
        r.e.barrier();
        return;
    }

    for (r.items, 0..) |it, idx| {
        const b: usize = r.base[idx];
        const rows: u32 = @intCast(it.ids.len);
        const hist = it.state.hist[k];
        try m.ops.convStatic(r.e, .{ .x = rows16(s.moe, b, h), .hist = v(hist, hl.mlp), .w = tv(m.weights.layers[i].mlp_conv), .out = rows16(s.c, b, h), .rows = rows, .channels = h, .pos0 = it.state.len });
    }
    r.e.barrier();
    for (r.items, 0..) |it, idx| {
        const b: usize = r.base[idx];
        const rows: u32 = @intCast(it.ids.len);
        const hist = it.state.hist[k];
        try m.ops.convStore(r.e, .{ .x = rows16(s.moe, b, h), .hist = v(hist, hl.mlp), .rows = rows, .channels = h, .pos0 = it.state.len });
    }
    try m.ops.residualMlp(r.e, .{ .h = v(s.h32, 0), .conv = v(s.c, 0), .out = v(s.hid(k + 1), 0), .rows = r.n, .cols = h });
    r.e.barrier();
}
