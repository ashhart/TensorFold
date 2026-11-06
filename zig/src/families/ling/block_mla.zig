//! Ling's MLA mixer on CUDA, absorbed: projections in their loaded formats, roped latent cache, head gate, `dense`.
const std = @import("std");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const block_kda = @import("block_kda.zig");
const expanded = @import("block_mla_prompt.zig");

const glue = graph.glue;
const latent = graph.latent;
const ops = graph.ops;
const Ops = graph.Ops;
const Config = @import("config.zig").Config;
const kw = state.cache_width;
const lw = 512;

/// Room for a segment's rows: the sequence's cache and the scratch's attention partials (512-key chunks).
pub fn room(b: *const state.Buffers, st: *const state.State, rows: usize) !void {
    if (st.pos + rows > st.capacity or latent.chunksFor(st.pos + rows) > b.nch) return error.CacheFull;
}

/// The block on b.normed into b.y: q (lora pair or q_proj), latent and roped key into the cache, attention, gate.
pub fn block(o: Ops, b: *const state.Buffers, a: weights.Mla, ab: graph.Absorb, g: *const graph.Graph, mi: usize, segs: []const block_kda.Seg, rows: usize) !void {
    const c = &g.w.config;
    const D: usize = c.hidden;
    const H: usize = c.mla_heads;
    const qk: usize = c.qkDim();
    const nope: usize = c.nope;
    const dv: usize = c.v_dim;
    if (c.kv_lora != lw or nope != 128 or dv != 128 or c.rope_dim != ops.rope_width) return error.LatentShape;
    if (a.q_a) |qa| {
        const ql: usize = c.q_lora;
        try o.mm(qa, b.normed, D, rows, b.qa_in);
        try glue.rmsnorm(o.set(), o.s, b.qa_in, ql, a.q_a_norm, b.qr, ql, null, c.eps, rows, ql);
        try o.mm(a.q_b.?, b.qr, ql, rows, b.q);
    } else try o.mm(a.q.?, b.normed, D, rows, b.q);
    const kvs = a.kv_a.outputs(); // kv_a's rows: 576, or an EXL3 pack's 640 (its padding read past)
    try o.mm(a.kv_a, b.normed, D, rows, b.kv);
    try glue.rmsnorm(o.set(), o.s, b.kv, kvs, a.kv_a_norm, b.lat, kw, null, c.eps, rows, lw);
    if (a.gate) |gp| try o.mm(gp, b.normed, D, rows, b.g);
    const split = state.splitOf(g.w);
    const wide = expanded.takes(o, b, split != null); // prompt rows in the expanded form: no absorb, no expand_v
    if (!wide) try ops.absorbQ(&o.m.ling, o.s, b.q, qk, ab.wk, b.qa, kw, rows, H, nope, lw);
    const scale = c.smScale();
    for (segs) |sg| try room(b, sg.st, sg.rows); // every segment before the first cache write
    for (segs) |sg| {
        const st = sg.st;
        const r = sg.row0;
        var rp = st.pos_dev; // the rope's first position: the cache's, or past it by a check's rope_base
        if (st.rope_base != 0) {
            try o.addI32(st.pos_dev, st.rope_base, b.at + 4);
            rp = b.at + 4;
        }
        const qr: ops.RopeRows = if (wide) .{ .x = b.q + (r * H * qk + nope) * 2, .x_row = H * qk, .x_head = qk, .out = b.q + (r * H * qk + nope) * 2, .o_row = H * qk, .o_head = qk } else .{ .x = b.q + (r * H * qk + nope) * 2, .x_row = H * qk, .x_head = qk, .out = b.qa + (r * H * kw + lw) * 2, .o_row = H * kw, .o_head = kw };
        try ops.rope(&o.m.ling, o.s, qr, sg.rows, H, rp, g.freq, g.rope_scale);
        try ops.rope(&o.m.ling, o.s, .{ .x = b.kv + (r * kvs + lw) * 2, .x_row = kvs, .x_head = 0, .out = b.lat + (r * kw + lw) * 2, .o_row = kw, .o_head = 0 }, sg.rows, 1, rp, g.freq, g.rope_scale);
        // a split cache is written by pages: this rank's chunks to theirs, other ranks' rows to the spare page
        const t: latent.Table = if (split != null) .{ .pt = g.table, .ps = latent.chunk * kw * 2 } else .{};
        try latent.latentWrite(o.set(), o.s, b.lat + r * kw * 2, kw, st.cacheAt(mi), t, st.pos_dev, sg.rows, kw);
        if (wide) { // a split cache gathers pages through b.lat: after every segment's write (below)
            if (split == null) try expanded.attend(o, b, a, g, mi, sg, scale);
            continue;
        }
        if (split != null) continue; // every segment's queries roped before they are gathered
        var r0: usize = 0;
        while (r0 < sg.rows) : (r0 += b.ar) {
            const m = @min(b.ar, sg.rows - r0);
            var at = st.pos_dev;
            if (r0 > 0) {
                try o.addI32(st.pos_dev, r0, b.at);
                at = b.at;
            }
            const nch = @max(latent.chunksFor(st.pos + r0 + m), o.nch_fix);
            const row = r + r0;
            try ops.attention(&o.m.ling, o.set(), o.s, b.qa + row * H * kw * 2, st.cacheAt(mi), at, b.partials, b.ol + row * H * lw * 2, b.dummy, m, H, lw, nch, scale);
        }
    }
    if (split) |s| if (wide) {
        for (segs) |sg| try expanded.attend(o, b, a, g, mi, sg, scale);
    } else try splitAttention(o, b, g, mi, segs, rows, scale, s);
    if (!wide) try ops.expandV(&o.m.ling, o.s, b.ol, ab.wv, b.vn, rows, H, dv, lw);
    if (a.gate != null) try o.gate(b.vn, b.g, H, rows, H, dv);
    try o.mm(a.o, b.vn, H * dv, rows, b.y);
}

/// Split-cache attention (two ranks): all heads over this rank's chunks, merged, swapped, then merged across the ranks.
fn splitAttention(o: Ops, b: *const state.Buffers, g: *const graph.Graph, mi: usize, segs: []const block_kda.Seg, rows: usize, scale: f32, s: state.Split) !void {
    const tp = g.tp orelse return error.BadWorld;
    if (s.world != 2 or tp.world != 2) return error.BadWorld;
    const H: usize = g.w.config.mla_heads;
    const q_rows = rows * H * kw * 2; // one rank's absorbed queries for the forward's rows
    try tp.allGather(o.s, b.qa, b.qa_all, q_rows);
    const peer = 1 - s.rank;
    for (segs) |sg| {
        const st = sg.st;
        var r0: usize = 0;
        while (r0 < sg.rows) : (r0 += b.ar) {
            const m = @min(b.ar, sg.rows - r0);
            var at = st.pos_dev;
            if (r0 > 0) {
                try o.addI32(st.pos_dev, r0, b.at);
                at = b.at;
            }
            const nloc = s.owned(latent.chunksFor(st.pos + r0 + m));
            const row = sg.row0 + r0;
            const po = m * H * lw * 4; // a rank's merged o for these rows; then its m and l, m * H * 4 each
            const ml = m * H * 4;
            try ops.chunksSplit(&o.m.ling, o.s, b.qa_all + peer * q_rows + row * H * kw * 2, st.cacheAt(mi), at, b.partials, m, H, nloc, scale, s.world, s.rank);
            try ops.mergePart(&o.m.ling, o.s, b.partials, b.sent, m, H, nloc);
            try tp.allGather(o.s, b.sent.po, b.got.po, po);
            try tp.allGather(o.s, b.sent.pm, b.got.pm, ml);
            try tp.allGather(o.s, b.sent.pl, b.got.pl, ml);
            try ops.chunksSplit(&o.m.ling, o.s, b.qa_all + s.rank * q_rows + row * H * kw * 2, st.cacheAt(mi), at, b.partials, m, H, nloc, scale, s.world, s.rank);
            const own: latent.Partials = .{ .po = b.got.po + s.rank * po, .pm = b.got.pm + s.rank * ml, .pl = b.got.pl + s.rank * ml };
            try ops.mergePart(&o.m.ling, o.s, b.partials, own, m, H, nloc); // over the copy of `sent` the gather left here
            try latent.merge(o.set(), o.s, b.got, b.ol + row * H * lw * 2, b.dummy, m, H, lw, s.world, false);
        }
    }
}
