//! A DeepSeek-V4.1 block over a forward's rows: engram, attention (each stream's on its own rows), then the MoE.
const std = @import("std");
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const idx = @import("index.zig");
const ffn = @import("moe.zig");
const Buffer = @import("metal").Buffer;
const fwd = @import("forward.zig");
const Enc = fwd.Enc;

const EngramArgs = extern struct { eps: f32 };

/// Publish global decoder latents and index keys even when this chunk feeds no local window.
pub fn decoderKeys(x: Enc, i: usize, l: *const wts.Layer) !void {
    comptime std.debug.assert(c.kv_sources[c.kv_sources.len - 1] == @import("decoder_prefill.zig").global_kv_layer);
    const s = x.s;
    const source = c.kv_sources.len - 1;
    const boundary = x.boundaryForLayer(l.hc_attn, s.pre[0], s.pre[1], l.attn_norm, false);
    x.completeBoundary(boundary);
    try idx.compress(x, source, c.ratios[i], l.attn.compressor.?, l.attn.indexer.?, x.m.freq(i));
}

/// One block: the engram lookup (layers 1 and 14), attention and the MoE, each between its boundaries.
pub fn block(input: Enc, i: usize, l: *const wts.Layer) !void {
    var x = input;
    const s = x.s;
    if (l.engram) |eg| {
        x.flush();
        x.linear(eg.wkv, s.engram_rows[s.engram_set][c.engramIndex(i).?], 0, s.engram_kv, 0, x.rows);
        x.pipe("ds_engram");
        x.buf(s.x, 0, 0);
        x.buf(s.engram_kv, 0, 1);
        x.tensor(eg.q_weight, 2);
        x.tensor(eg.k_weight, 3);
        x.args(EngramArgs{ .eps = c.norm_eps }, 4);
        x.buf(s.img, 0, 5);
        x.groups(.{ c.hc, x.rows, 1 }, .{ 256, 1, 1 });
    }
    const attn_boundary = x.boundaryForLayer(l.hc_attn, s.pre[0], s.pre[1], l.attn_norm, true);
    x.expand_skip = 0;
    try attention(x, i, l, attn_boundary);
    x.expand();
    const ffn_boundary = x.boundaryForLayer(l.hc_ffn, s.pre[1], s.pre[0], l.ffn_norm, true);
    try ffn.moe(x, &l.moe, c.topk, c.experts, ffn_boundary);
    x.expand();
}

fn attention(x: Enc, layer: usize, l: *const wts.Layer, boundary: ?fwd.HcJob) !void {
    const s = x.s;
    const m = x.m;
    const rows = x.rows;
    const a = &l.attn;
    const freq = m.freq(layer);
    // q = wq_b(q_norm(wq_a(x))) and the window key kv_norm(wkv(x)) into the ring, the two chains side by side
    x.linear(a.wq_a, s.normed, 0, s.qr, 0, rows);
    x.completeBoundary(boundary);
    x.with();
    x.linear(a.wkv, s.normed, 0, s.kv, 0, rows);
    x.rms(s.qr, 0, a.q_norm, s.qr, 0, rows, c.q_rank);
    x.with();
    for (x.segs) |g| x.of(g).kvFinish(s.kv, a.kv_norm, freq, g.cache.windows[layer]);
    const ratio: usize = c.ratios[layer];
    const overlap = m.projection_overlap and ratio != 0 and (a.indexer != null or a.compressor != null) and
        x.segs.len == 1 and rows <= 1 + c.draft_block;
    const direct = x.short_index and ratio != 0 and (x.base + rows) / ratio <= st.picks;
    const needs_query = !direct or m.short_index_audit;
    const audit = overlap and m.projection_audit;
    if (audit) {
        x.linear(a.wq_b, s.qr, 0, s.query_ref, 0, rows);
        if (a.indexer) |ix| idx.projectQuery(x, ix, s.index_query_ref, s.index_weight_ref);
    }
    x.linear(a.wq_b, s.qr, 0, s.q, 0, rows);
    if (overlap and needs_query) if (a.indexer) |ix| {
        x.with();
        idx.projectQuery(x, ix, s.iq, s.iw);
    };

    const si: ?usize = if (ratio == 0) null else std.mem.indexOfScalar(usize, &c.kv_sources, c.kvSource(layer).?).?;
    if (si) |k| {
        if (a.compressor) |cp| {
            if (overlap) x.with();
            try idx.compress(x, k, ratio, cp, a.indexer.?, freq);
        }
        if (audit) {
            auditProjection(x, s.q, s.query_ref, rows * c.heads * c.head_dim);
            if (a.indexer != null) {
                auditProjection(x, s.iq, s.index_query_ref, rows * c.index_heads * c.index_dim);
                auditProjection(x, s.iw, s.index_weight_ref, rows * c.index_heads);
            }
        }
        if (a.indexer) |ix| {
            if (needs_query) try idx.index(x, k, layer, ratio, ix, freq, overlap);
            if (direct) @import("short_index.zig").emit(x, k, ratio, layer);
        }
    }
    // sparse attention over each stream's window and picks, then the output's RoPE taken back
    for (x.segs) |g| {
        const comp = if (si) |k| g.cache.sources[k].latents else g.cache.windows[layer]; // unread without picks
        x.of(g).attend(s.q, g.cache.windows[layer], comp, s.picks, if (si != null) st.picks else 0, a.sink, s.o, -1, freq, ratio);
    }
    // the grouped low-rank output: block-diagonal wo_a, wo_b
    x.woA(a.wo_a, rows);
    x.linear(a.wo_b, s.oa, 0, s.branch, 0, rows);
}

fn auditProjection(x: Enc, actual: Buffer, expected: Buffer, len: usize) void {
    x.pipe("tf_attention_projection_audit");
    x.buf(actual, 0, 0);
    x.buf(expected, 0, 1);
    x.buf(x.s.projection_error, 0, 2);
    x.args(@as(u32, @intCast(len)), 3);
    x.threads(.{ len, 1, 1 }, .{ 256, 1, 1 });
    x.segs[0].cache.projection_audited += len;
}
