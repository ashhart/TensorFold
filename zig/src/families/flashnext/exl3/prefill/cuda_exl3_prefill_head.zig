//! The MTP head run expert-major over deep chunks (the hook cuda_mtp_ops.zig calls).

const std = @import("std");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const state = @import("cuda_exl3_prefill_state.zig");
const chunks = @import("cuda_exl3_prefill_chunk.zig");
const experts = @import("cuda_exl3_prefill_moe.zig");
const tiles = @import("cuda_exl3_prefill_tiles.zig");

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const eps = wt.eps;
const out_sk = wt.out_sk;
const att_rows = batch.att_rows;

const max_sub = state.max_sub;
const view = chunks.view;
const moe = experts.moe;
const warmTiles = tiles.warmTiles;

/// The hook in cuda_mtp_ops.zig `forwardHead` (a lane's prompt chunk through the MTP head, after the main pass kept
pub fn head(comptime Tri: type, o: anytype, d: *@import("../../cuda_mtp.zig").Draft, next: []const u32, rows: u64, scores: bool) !bool {
    const op = o.e.exl3_prefill.opts();
    const n = next.len;
    // A paired window hands the head up to two deep chunks of rows: it runs them as two slices (headSplit).
    if (!scores and o.e.exl3_prefill.deep != null and n > o.e.exl3_prefill.deep.?.rows) {
        const first = headSplit(n, o.e.exl3_prefill.deep.?.rows, op.min_rows) orelse return false;
        if (!try headSlice(Tri, o, d, next[0..first], rows, false)) return false;
        if (!try headSlice(Tri, o, d, next[first..], rows + first * wide * 2, false)) return error.UnexpectedTensor;
        return true;
    }
    return headSlice(Tri, o, d, next, rows, scores);
}

/// Where a head call of `n` rows over a paired window splits into two deep-sized slices, or null when it cannot.
fn headSplit(n: usize, deep_rows: usize, min_rows: usize) ?usize {
    if (n <= deep_rows or n > 2 * deep_rows) return null;
    const first = ((n + 1) / 2 + att_rows - 1) / att_rows * att_rows;
    if (first >= n or first > deep_rows or n - first < min_rows) return null;
    return first;
}

test "exl3: a paired window's head splits into two deep slices that cannot decline after the first runs" {
    try std.testing.expectEqual(@as(?usize, 4096), headSplit(8192, 4096, 512));
    try std.testing.expectEqual(@as(?usize, null), headSplit(4096, 4096, 512));
    try std.testing.expectEqual(@as(?usize, null), headSplit(8193, 4096, 512));
    var n: usize = 4097;
    while (n <= 8192) : (n += 61) {
        const first = headSplit(n, 4096, 512) orelse continue;
        try std.testing.expect(first <= 4096 and first % att_rows == 0 and n - first <= first and n - first >= 512);
    }
}

fn headSlice(comptime Tri: type, o: anytype, d: *@import("../../cuda_mtp.zig").Draft, next: []const u32, rows: u64, scores: bool) !bool {
    const op = o.e.exl3_prefill.opts();
    const n = next.len;
    if (!op.enabled or !headEnabled() or scores or n < op.min_rows or d.media != null) return false;
    const e = o.e;
    const dp = o.e.exl3_prefill.deep orelse return false;
    if (dp.owner != @as(*const anyopaque, e) or dp.rows < n) return false;
    const h = o.h;
    const x = h.layer.exl3 orelse return false;
    if (x.tp != null or x.pool == null or (x.layer.act_mode != 1 and x.layer.act_mode != 0)) return false;
    if (d.len + n > d.cap) return error.PromptTooLong;
    try @import("../dplan/cuda_exl3_dplan.zig").toHost(&e.exl3_dplan); // the head's pool may be device-planned (cuda_exl3_chain.zig)
    const mtp = @import("../../cuda_mtp.zig");
    const tri = o.tri;
    const want = @max(@as(usize, 32), 2 * @min(n, mtp.piece_rows));
    const s = try e.reserve(want);
    const piece = @max(@as(usize, 1), @min(mtp.piece_rows, s.cap / 2)) / att_rows * att_rows;
    if (piece == 0) return false;
    const n_pieces = (n + piece - 1) / piece;
    if (n_pieces > max_sub) return false;
    const score_rows = try e.reserveScores(d.cap, piece);
    var p: batch.Pass(Tri) = .{
        .gpa = e.gpa,
        .io = e.io,
        .stream = e.stream.*,
        .tri = tri,
        .k = &e.batch_k,
        .s = s,
        .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram, .dplan = &e.exl3_dplan },
        .seq = d.headSeq(null, score_rows),
        .hooks = .{ .logits = .none },
    };
    var views: [max_sub]batch.Scratch = undefined;
    for (0..n_pieces) |b| views[b] = view(s, dp, b * piece);
    defer p.s = s;
    if (op.tiles and !op.slots_kernel) {
        try warmTiles(e);
        dp.runtime.tiles_on = true;
    }
    defer dp.runtime.tiles_on = false;
    const lw = &h.layer;
    for (0..n_pieces) |b| {
        const at = b * piece;
        const m = @min(piece, n - at);
        p.s = &views[b];
        const v = p.s;
        p.rows = m;
        p.pos0 = d.len + at;
        p.pending = null;
        try p.stage(next[at..][0..m]);
        // forwardHead's piece up to the router (the embedding branch, the hidden branch).
        if (e.table.dense) {
            try tri.embedDense(v.ids, e.table.w.ptr, v.ple_keys, dims, streams, m);
        } else {
            try tri.embed(v.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, v.ple_keys, dims, streams, m);
        }
        try tri.rmsnorm(v.ple_keys, h.norm_e.ptr, v.mixed, v.xsm, eps, wide, m, dims, dims);
        _ = try p.lane(v.mixed, v.xsm, h.fc_e, v.dummy, v.part, m, mtp.fc_sk);
        try p.sumSlices(v.part, v.branch, true, m * dims, mtp.fc_sk);
        try tri.rmsnorm(rows + at * wide * 2, h.norm_h.ptr, v.norm, v.gq, eps, wide, m, wide, wide);
        _ = try p.lane(v.norm, v.gq, h.fc_h, v.dummy, v.part, streams * m, mtp.fc_sk);
        try p.sumSlices(v.part, v.ple_gated, true, streams * m * dims, mtp.fc_sk);
        try tri.addStreams(v.branch, v.ple_gated, v.h, m, dims, streams);
        try p.writeback(0, m);
        try p.readout(lw.attn_hc, 0, m);
        try p.attnBlock(lw, 0);
        try tri.hcWriteback(v.h, v.h, v.pss, v.part, "*fp32", v.inj, v.dummy, "*bf16", v.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, out_sk);
        try p.readout(lw.mlp_hc, 0, m);
        try tri.routerMany(v.mixed, lw.router.ptr, v.router_l, dims, m);
        try tri.topkMany(v.router_l, v.picks, v.wts, m);
    }
    // the head's experts once over the chunk (its own pool, passes of half its lines)
    try moe(Tri, dp, &p, lw, 0, n, op);
    for (0..n_pieces) |b| {
        const m = @min(piece, n - b * piece);
        p.s = &views[b];
        p.rows = m;
        p.pending = 0;
        try p.writeback(0, m);
    }
    const last = &views[n_pieces - 1];
    try p.copy(d.out.ptr, last.h + (n - 1 - (n_pieces - 1) * piece) * wide * 2, wide * 2);
    d.len += n;
    dp.runtime.stats.head_rows += n;
    if (!head_logged) std.log.info("flashnext exl3 prefill: the MTP head runs expert-major over deep chunks ({d} rows first)", .{n});
    head_logged = true;
    return true;
}
var head_logged = false;

fn headEnabled() bool {
    return true;
}
