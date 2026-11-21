//! DeepSeek-V4.1-Flash's compressors and indexers: the KV sources' compressed latents and each row's picks.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Enc = fwd.Enc;
const Buffer = mtl.Buffer;

const PoolArgs = extern struct { first: i32, ring: i32 };
const ScoreArgs = extern struct { rows: i32, groups: i32, ratio: i32, first_pos: i32, stride: i32 };
const KeptArgs = extern struct { rows: i32, groups: i32, ratio: i32, first_pos: i32, block: i32, kept: i32 };
const BlockArgs = extern struct { rows: i32, groups: i32, block: i32, first_pos: i32, ratio: i32, stride: i32 };
const TopkArgs = extern struct { n: i32, k: i32, stride: i32 };

/// A KV source's compressor: completed groups pooled or projected, normed, index keys, RoPE'd, fp4 into each cache.
pub fn compress(x: Enc, si: usize, ratio: usize, cp: wts.Compressor, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const rows = x.rows;
    if (ratio > 1) {
        x.gemv(true, cp.wkv, s.normed, 0, c.hidden, s.cproj, 0, rows, c.head_dim, c.hidden, c.head_dim);
        x.with(); // the two projections of the same rows, side by side
        x.gemv(true, cp.wgate.?, s.normed, 0, c.hidden, s.cscore, 0, rows, c.head_dim, c.hidden, c.head_dim);
    } else {
        x.gemv(false, cp.wkv, s.normed, 0, c.hidden, s.latent, 0, rows, c.head_dim, c.hidden, c.head_dim);
    }
    for (x.segs) |g| try compressOne(x.of(g), g.cache, si, ratio, cp, ix, freq);
}

fn compressOne(x: Enc, cache: *st.Cache, si: usize, ratio: usize, cp: wts.Compressor, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const src = &cache.sources[si];
    const span = fwd.completed(x.base, x.rows, ratio);
    const n = span[1] - span[0];
    if (span[1] > src.groups) return error.DeepseekCacheFull;
    const hd = c.head_dim * 2;
    if (ratio > 1) {
        x.putF32(s.cproj, s.pos, cache.proj_kv[si].?, x.rows, c.head_dim, st.ring);
        x.with();
        x.putF32(s.cscore, s.pos, cache.proj_score[si].?, x.rows, c.head_dim, st.ring);
        if (n == 0) return;
        x.pipe("ds_pool2");
        x.buf(cache.proj_kv[si].?, 0, 0);
        x.buf(cache.proj_score[si].?, 0, 1);
        x.at(s.latent, hd, 2);
        x.args(PoolArgs{ .first = @intCast(span[0]), .ring = st.ring }, 3);
        x.threads(.{ c.head_dim, n, 1 }, .{ 256, 1, 1 });
    }
    x.rms(s.latent, x.r0 * hd, cp.norm, s.latent, x.r0 * hd, n, c.head_dim);
    for (s.gpos[si].slice(i32, x.r0 + n)[x.r0..], s.gidx[si].slice(i32, x.r0 + n)[x.r0..], 0..) |*p, *g, j| {
        g.* = @intCast(span[0] + j);
        p.* = @intCast((span[0] + j) * ratio);
    }
    // index keys: k_norm(wk(latent)), RoPE'd at the group's first position, fp4 with e8m0 scales a 32; the latents

    // RoPE'd at the same position, fp4 with e4m3 scales a 16: the two tails side by side once wk has read the latents
    const kd = c.index_dim * 2;
    x.gemv(false, ix.wk.?, s.latent, x.r0 * hd, c.head_dim, s.ik, x.r0 * kd, n, c.index_dim, c.head_dim, c.index_dim);
    x.rms(s.ik, x.r0 * kd, ix.k_norm.?, s.ik, x.r0 * kd, n, c.index_dim);
    x.rope(s.ik, n, 1, c.index_dim, s.gpos[si], freq, false);
    x.with();
    x.rope(s.latent, n, 1, c.head_dim, s.gpos[si], freq, false);
    x.fp4(s.ik, n, c.index_dim, false);
    x.with();
    x.fp4(s.latent, n, c.head_dim, true);
    x.put(s.ik, s.gidx[si], src.keys, n, c.index_dim, 0);
    x.with();
    x.put(s.latent, s.gidx[si], src.latents, n, c.head_dim, 0);
}

pub fn projectQuery(x: Enc, ix: wts.Indexer, query: Buffer, weights: Buffer) void {
    x.linear(ix.wq_b, x.s.qr, 0, query, 0, x.rows);
    x.with();
    x.gemv(false, ix.weights_proj, x.s.normed, 0, c.hidden, weights, 0, x.rows, c.index_heads, c.hidden, c.index_heads);
}

/// An index source's picks: scores over visible keys, each row's 512 best in key order, rows in slices that fit.
pub fn index(x: Enc, si: usize, layer: usize, ratio: usize, ix: wts.Indexer, freq: Buffer, projected: bool) !void {
    const s = x.s;
    const rows = x.rows;
    if (!projected) projectQuery(x, ix, s.iq, s.iw);
    x.rope(s.iq, rows, c.index_heads, c.index_dim, s.pos, freq, false);
    x.with();
    x.pipe("ds_scale");
    x.buf(s.iw, 0, 0);
    x.args(@as(f32, 1.0 / @sqrt(@as(f32, c.index_dim)) * (1.0 / @sqrt(@as(f32, c.index_heads)))), 1);
    x.threads(.{ rows * c.index_heads, 1, 1 }, .{ 256, 1, 1 });
    x.fp4(s.iq, rows, c.index_heads * c.index_dim, false);
    for (x.segs) |g| indexOne(x.of(g), &g.cache.sources[si], layer, ratio);
}

fn indexOne(x: Enc, src: *const st.Source, layer: usize, ratio: usize) void {
    const s = x.s;
    const rows = x.rows;
    const groups = (x.base + rows) / ratio;
    if (groups == 0) {
        x.pipe("ds_topk_bf");
        x.buf(s.scores, 0, 0);
        x.at(s.picks, st.picks * 4, 1);
        x.args(TopkArgs{ .n = 0, .k = st.picks, .stride = 1 }, 2);
        x.groups(.{ rows, 1, 1 }, .{ 1024, 1, 1 });
        return;
    }
    if (layer > c.candidate_source) return indexKept(x, src, groups, ratio);
    std.debug.assert(s.score_bytes > 0);
    const slice = @max(1, @min(rows, s.score_bytes / 2 / groups));
    var r0: usize = 0;
    while (r0 < rows) : (r0 += slice) {
        const n = @min(slice, rows - r0);
        const r = x.r0 + r0;
        const original_row = r + x.token_skip;
        const tile: [2]usize = if (n == 1) .{ 1, 8 } else .{ 2, 16 };
        // a lone stream's prompt chunk: 64-row by 64-key tiles on the tensor units (the dot's fp32 order is the op's)
        const nax: ?mtl.Pipeline = if (fwd.index_nax and x.units and n >= fwd.nax_rows) x.m.kernels.index_nax else null;
        if (nax) |p| {
            x.ep.*.setPipeline(p);
        } else if (n == 1) x.pipe("ds_index_scores_1x8") else x.pipe("ds_index_scores_2x16");
        x.buf(s.iq, r * c.index_heads * c.index_dim * 2, 0);
        x.buf(s.iw, r * c.index_heads * 2, 1);
        x.buf(src.keys, 0, 2);
        x.buf(s.scores, 0, 3);
        x.args(ScoreArgs{ .rows = @intCast(n), .groups = @intCast(groups), .ratio = @intCast(ratio), .first_pos = @intCast(x.base + r0), .stride = @intCast(groups) }, 4);
        if (nax != null) {
            x.groups(.{ (groups + 63) / 64, (n + 63) / 64, 1 }, .{ 128, 1, 1 });
        } else {
            const row_tiles = (n + tile[0] - 1) / tile[0];
            x.threads(.{ row_tiles * 32, (groups + tile[1] - 1) / tile[1], 1 }, .{ @as(usize, @min(row_tiles, 8)) * 32, 1, 1 });
        }
        const nb = (groups + c.candidate_block - 1) / c.candidate_block;
        if (layer == c.candidate_source) {
            x.pipe("ds_block_max");
            x.buf(s.scores, 0, 0);
            x.buf(s.blocks, 0, 1);
            x.args(BlockArgs{ .rows = @intCast(n), .groups = @intCast(groups), .block = c.candidate_block, .first_pos = @intCast(x.base + r0), .ratio = @intCast(ratio), .stride = @intCast(groups) }, 2);
            x.threads(.{ nb, n, 1 }, .{ @min(nb, 256), 1, 1 });
            x.pipe("ds_topk");
            x.buf(s.blocks, 0, 0);
            x.buf(s.keep, original_row * c.candidate_blocks * 4, 1);
            x.args(TopkArgs{ .n = @intCast(nb), .k = c.candidate_blocks, .stride = @intCast(nb) }, 2);
            x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
        }
        x.pipe("ds_topk_bf");
        x.buf(s.scores, 0, 0);
        x.buf(s.picks, original_row * st.picks * 4, 1);
        x.args(TopkArgs{ .n = @intCast(groups), .k = st.picks, .stride = @intCast(groups) }, 2);
        x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
    }
}

/// Indexers after layer 20 score only its candidate blocks, in key order: the top-k equals the masked scan's.
fn indexKept(x: Enc, src: *const st.Source, groups: usize, ratio: usize) void {
    comptime std.debug.assert(c.candidate_block == 8); // ds_index_scores_kept's block
    const s = x.s;
    const slots = c.candidate_blocks * c.candidate_block;
    std.debug.assert(s.score_bytes > 0);
    const slice = @max(1, @min(x.rows, s.score_bytes / 2 / slots));
    var r0: usize = 0;
    while (r0 < x.rows) : (r0 += slice) {
        const n = @min(slice, x.rows - r0);
        const r = x.r0 + r0;
        const original_row = r + x.token_skip;
        const args = KeptArgs{ .rows = @intCast(n), .groups = @intCast(groups), .ratio = @intCast(ratio), .first_pos = @intCast(x.base + r0), .block = c.candidate_block, .kept = c.candidate_blocks };
        x.pipe("ds_index_scores_kept");
        x.buf(s.iq, r * c.index_heads * c.index_dim * 2, 0);
        x.buf(s.iw, r * c.index_heads * 2, 1);
        x.buf(src.keys, 0, 2);
        x.buf(s.scores, 0, 3);
        x.buf(s.keep, original_row * c.candidate_blocks * 4, 4);
        x.args(args, 5);
        x.threads(.{ c.candidate_blocks * 32, n, 1 }, .{ 256, 1, 1 }); // a simdgroup a candidate block (8 keys)
        x.pipe("ds_topk_bf");
        x.buf(s.scores, 0, 0);
        x.buf(s.picks, original_row * st.picks * 4, 1);
        x.args(TopkArgs{ .n = @intCast(slots), .k = st.picks, .stride = @intCast(slots) }, 2);
        x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
        x.pipe("ds_kept_keys");
        x.buf(s.picks, original_row * st.picks * 4, 0);
        x.buf(s.keep, original_row * c.candidate_blocks * 4, 1);
        x.args(args, 2);
        x.threads(.{ st.picks, n, 1 }, .{ 256, 1, 1 });
    }
}
