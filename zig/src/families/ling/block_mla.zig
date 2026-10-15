//! Ling's MLA mixer on CUDA, absorbed: projections in their loaded formats, roped latent cache, head gate, `dense`.
const std = @import("std");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const block_kda = @import("block_kda.zig");

const glue = graph.glue;
const latent = graph.latent;
const Ops = graph.Ops;
const Config = @import("config.zig").Config;
const kw = state.cache_width;
const lw = 512;

/// The block on b.normed into b.y: q (lora pair or q_proj), latent and roped key into the cache, attention, gate.
pub fn block(o: Ops, b: *const state.Buffers, a: weights.Mla, ab: graph.Absorb, g: *const graph.Graph, mi: usize, segs: []const block_kda.Seg, rows: usize) !void {
    const c = &g.w.config;
    const D: usize = c.hidden;
    const H: usize = c.mla_heads;
    const qk: usize = c.qkDim();
    const nope: usize = c.nope;
    const dv: usize = c.v_dim;
    if (c.kv_lora != lw or nope != 128 or dv != 128 or c.rope_dim != latent.rope_width) return error.LatentShape;
    if (a.q_a) |qa| {
        const ql: usize = c.q_lora;
        try o.mm(qa, b.normed, D, rows, b.qa_in);
        try glue.rmsnorm(o.set(), o.s, b.qa_in, ql, a.q_a_norm, b.qr, ql, null, c.eps, rows, ql);
        try o.mm(a.q_b.?, b.qr, ql, rows, b.q);
    } else try o.mm(a.q.?, b.normed, D, rows, b.q);
    try o.mm(a.kv_a, b.normed, D, rows, b.kv);
    try glue.rmsnorm(o.set(), o.s, b.kv, kw, a.kv_a_norm, b.lat, kw, null, c.eps, rows, lw);
    if (a.gate) |gp| try o.mm(gp, b.normed, D, rows, b.g);
    try latent.absorbQStrided(o.set(), o.s, b.q, qk, ab.wk, b.qa, kw, rows, H, nope, lw);
    const scale = c.smScale();
    for (segs) |sg| {
        const st = sg.st;
        const r = sg.row0;
        if (st.pos + sg.rows > st.capacity) return error.CacheFull;
        try latent.rope(o.set(), o.s, .{ .x = b.q + (r * H * qk + nope) * 2, .x_row = H * qk, .x_head = qk, .out = b.qa + (r * H * kw + lw) * 2, .o_row = H * kw, .o_head = kw }, sg.rows, H, st.pos_dev, g.freq);
        try latent.rope(o.set(), o.s, .{ .x = b.kv + (r * kw + lw) * 2, .x_row = kw, .x_head = 0, .out = b.lat + (r * kw + lw) * 2, .o_row = kw, .o_head = 0 }, sg.rows, 1, st.pos_dev, g.freq);
        try latent.latentWrite(o.set(), o.s, b.lat + r * kw * 2, kw, st.cacheAt(mi), st.pos_dev, sg.rows, kw);
        var r0: usize = 0;
        while (r0 < sg.rows) : (r0 += state.att_rows) {
            const m = @min(state.att_rows, sg.rows - r0);
            var at = st.pos_dev;
            if (r0 > 0) {
                try o.addI32(st.pos_dev, r0, b.at);
                at = b.at;
            }
            const nch = latent.chunksFor(st.pos + r0 + m);
            if (nch > b.nch) return error.CacheFull;
            const row = r + r0;
            try latent.attentionRoped(o.set(), o.s, b.qa + row * H * kw * 2, st.cacheAt(mi), at, b.partials, b.ol + row * H * lw * 2, b.dummy, m, H, lw, nch, scale);
        }
    }
    try latent.expandV(o.set(), o.s, b.ol, ab.wv, b.vn, rows, H, dv, lw);
    if (a.gate != null) try o.gate(b.vn, b.g, H, rows, H, dv);
    try o.mm(a.o, b.vn, H * dv, rows, b.y);
}
