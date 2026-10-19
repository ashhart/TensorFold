//! A training pass on the batched pass: sequences side by side, each from position 0 and a fresh state.
const std = @import("std");
const cuda = @import("cuda");
const batch = @import("cuda_batch.zig");
const gdn = @import("cuda_gdn.zig");
const wt = @import("cuda_prompt_wt.zig");

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const out_k = wt.out_k;
const nch = wt.nch;
const out_sk = wt.out_sk;
const state_n = wt.state_n;
const ple_layer = wt.ple_layer;
const ple_tail_n = wt.ple_tail_n;
const pss_n = batch.pss_n;

/// Each list[i]'s rows of `ids` (`at` even) side by side: row-wise launches take all, a mixer one sequence at a time.
pub fn spans(comptime Tri: type, p: *batch.Pass(Tri), ids: []const u32, list: []const batch.Span, eos: i64) !void {
    const base = p.s;
    const n = ids.len;
    if (n == 0 or n > base.cap) return error.UnexpectedTensor;
    for (list) |sp| if (sp.at % 2 != 0 or sp.rows == 0 or sp.at + sp.rows > n or sp.rows > nch * 512 or sp.rows > p.seq.cap) return error.UnexpectedTensor;
    defer p.s = base;
    p.rows = n;
    p.pos0 = 0;
    p.pending = null;
    p.pending_f32 = false;
    // each sequence's n-gram rows from its own start, staged in the scratch's pinned rows
    try base.ready();
    const emb = base.hostEmb(n);
    const xs = base.hostXs(n);
    @memset(emb, 0);
    @memset(xs, 0);
    for (list) |sp| {
        var hist = [2]i64{ eos, eos };
        try p.m.ngram.rows(&hist, ids[sp.at..][0..sp.rows], emb[sp.at * dims ..][0 .. sp.rows * dims], xs[sp.at * (dims / 32) ..][0 .. sp.rows * (dims / 32)]);
    }
    try p.stage(ids);
    try p.tri.embed(base.ids, p.m.table_w, p.m.table_s, p.m.table_b, base.h, dims, streams, n);
    const zero = struct {
        fn f(pp: *batch.Pass(Tri), ptr: u64, words: usize) !void {
            const dr = pp.stream.d;
            try dr.check(dr.api.cuMemsetD32Async(ptr, 0, words, pp.stream.handle), "cuMemsetD32Async");
        }
    }.f;
    for (p.m.layers, 0..) |*lw, layer| {
        const keep: batch.LayerKeep = if (p.hooks.keep) |kp| kp.layers[layer] else .{};
        if (layer == ple_layer) {
            try p.upload(base.ple_emb, std.mem.sliceAsBytes(emb));
            try p.upload(base.ple_xs, std.mem.sliceAsBytes(xs));
            try base.uploaded();
            if (p.pending != null) try p.writeback(0, n);
            for (list) |sp| {
                try zero(p, p.seq.tail.ptr, ple_tail_n * wide / 2);
                var view = base.*;
                view.h += sp.at * wide * 2;
                view.ple_emb += sp.at * dims * 2;
                view.ple_xs += sp.at * (dims / 32) * 4;
                p.s = &view;
                p.rows = sp.rows;
                try p.pleBlock(lw, ids[sp.at..][0..sp.rows]);
                p.s = base;
                p.rows = n;
            }
        }
        try p.writeback(0, n);
        try p.readoutKeep(lw.attn_hc, 0, n, keep.attn_hc);
        for (list) |sp| {
            var view = base.*;
            view.mixed += sp.at * dims * 2;
            view.xsm += sp.at * (dims / 32) * 4;
            var k = keep;
            k.hist = 0;
            k.state_out = 0;
            if (k.proj != 0) k.proj += sp.at * gdn.proj_width * 2;
            if (k.pq != 0) k.pq += sp.at * q_heads * 2 * head_dim * 2;
            if (k.pk != 0) k.pk += sp.at * kv_heads * head_dim * 2;
            if (k.pv != 0) k.pv += sp.at * kv_heads * head_dim * 2;
            if (k.gated != 0) k.gated += sp.at * out_k * 2;
            p.s = &view;
            p.rows = sp.rows;
            if (lw.linear) {
                const li = wt.linearIndex(layer);
                try zero(p, p.seq.conv[li].ptr, (gdn.taps - 1) * gdn.conv_dim / 2);
                try zero(p, p.seq.state[li].ptr, state_n);
                try p.gdnBlock(lw, layer, li, k);
            } else {
                try p.attnBlockKeep(lw, wt.attnIndex(layer), k);
            }
            if (keep.mix_branch != 0) try p.sumSlices(base.part, keep.mix_branch + sp.at * dims * 2, true, sp.rows * dims, out_sk);
            try p.tri.hcWriteback(base.h + sp.at * wide * 2, base.h + sp.at * wide * 2, base.pss + sp.at * pss_n * 4, base.part, "*fp32", base.inj + sp.at * streams * 2, base.dummy, "*bf16", base.dummy, "*bf16", sp.rows * dims, sp.rows, dims, streams, 4, 1, 1, out_sk);
            p.s = base;
            p.rows = n;
        }
        try p.readoutKeep(lw.mlp_hc, 0, n, keep.mlp_hc);
        try p.moeBlockKeep(lw, keep);
        try p.moeAdds(layer, keep);
        p.pending = layer;
    }
    try p.head(null);
}
