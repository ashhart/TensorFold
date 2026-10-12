//! A decode window's indexer: index scores (whole, row-blocked), the selections, attn_cuda's top-k.

const std = @import("std");
const calls = @import("block_calls.zig");
const ft = @import("forward_types.zig");
const block = @import("block.zig");
const Arg = calls.Arg;
const Emitter = block.Emitter;
const Norm = block.Emitter.Norm;

/// attn_cuda's MAX_EPT: past 8 CTAs x 512 threads x 31 entries a selection runs the Triton dtopk instead.
const topk_max_ept: i64 = 31;

/// topk.plan: (cluster CTAs, entries a thread) for `nk` entries.
fn topkPlan(nk: i64, cl_in: ?i64) [2]i64 {
    const cl = cl_in orelse @max(1, @min(8, std.math.divCeil(i64, nk, 4096) catch 1));
    var ept = @max(1, std.math.divCeil(i64, nk, cl * 512) catch 1);
    ept += 1 - @mod(ept, 2);
    return .{ cl, ept };
}

/// The window's index selection of layer `src` (reuse layers read the source's).
pub fn selection(e: *Emitter, src: u32) ![2]Arg {
    const k: i64 = e.cfg.index_topk;
    return .{
        try e.view(.{ .buf = try e.fmt("w.sel{d}", .{src}) }, .i32, &.{ e.n, k }, &.{ k, 1 }, e.row0 * k * 4),
        // the counts' 4-byte rows: a run's segment from a row a multiple of 4 (cnt0), so the slice is 16-aligned
        try e.view(.{ .buf = try e.fmt("w.cnt{d}", .{src}) }, .i32, &.{e.n}, &.{1}, e.cnt0 * 4),
    };
}

/// attn_cuda's ticket words: a (row, head tile)'s ranks, 4 with the split kernel (`r1` or `r1_sig`)
pub fn ticketLen(e: *const Emitter) i64 {
    return if (e.o.r1 or e.o.r1_sig) 1024 else 256;
}

/// attn_cuda.split: CTAs a chunk with `r1` (ATTN_SPLIT 4 up to ATTN_SPLIT_ROWS = 2 rows, 1 above)
pub fn attnSplit(n: i64) i64 {
    return if (n <= 2) 4 else 1;
}

/// The indexer: query projection and rope, scores over the keys, top-k (and a candidate source's blocks).
pub fn index(e: *Emitter, L: u32, qn: Arg, iw: Arg, qnorm: ?Norm) !void {
    const n = e.n;
    const IH: i64 = e.cfg.index_heads;
    const ID: i64 = e.cfg.index_dim;
    const qi = try e.buf("L.qi", .{}, .bf16, &.{ n, IH * ID });
    try e.fusedNorm("qi", &.{.{ .w = try e.fmt("L{d}.attn.ix_wq_b", .{L}), .x = qn, .y = qi, .k = e.cfg.q_lora, .n = @intCast(IH * ID) }}, qnorm);
    try e.rope(qi, IH, ID, ID - 64);
    const reindex = e.cfg.mode(L) == .reindex;
    const src = if (reindex) e.cfg.kvSource(L).? else L;
    const ratio: i64 = e.cfg.compressRatio(src);
    const cbs: i64 = e.cfg.candidate_block_size;
    const nk: i64 = if (reindex) @as(i64, e.cfg.candidate_blocks) * cbs else @divFloor(e.end(), ratio);
    const cand = try e.buf("w.cand", .{}, .i32, &.{ n, e.cfg.candidate_blocks });
    // the materialised scores + sort keys within the index budget, else row blocks through `_keys`
    if (reindex) {
        if (n * nk * 16 > e.o.index_budget) return error.Unsupported; // index.reindex in row blocks: > 256 rows at 64 MiB
    } else {
        const nvis = @max(nk, 1);
        if (!e.o.rows and n > 16 and nvis >= ft.stream_min and !e.cfg.isCandidateSource(L) and @popCount(e.cfg.index_topk) == 1) return error.Unsupported;
        if (n * nvis * 12 > e.o.index_budget) return e.blocked(L, qi, iw, src, ratio, nvis);
    }
    const sc = try e.buf("L.scores", .{}, .f32, &.{ n, nk });
    try e.scoreRows(qi, iw, src, ratio, 0, n, try e.pos(), nk, sc, if (reindex) cand else null);
    try e.topkOf(L, sc, cand, nk, ratio, e.o.index_bound);
}

/// index.scores over rows [a, a + rows) into `out` [rows, nk]: dense, or over candidate blocks `cand` (Reindex).
pub fn scoreRows(e: *Emitter, qi: Arg, iw: Arg, src: u32, ratio: i64, a: i64, rows: i64, p: Arg, nk: i64, out: Arg, cand: ?Arg) !void {
    const IH: i64 = e.cfg.index_heads;
    const ID: i64 = e.cfg.index_dim;
    const cbs: i64 = e.cfg.candidate_block_size;
    const spg = try e.paging(ratio, false);
    try e.triton(if (e.o.index_bound and cand == null) "_scores_b" else "_scores", .{ rows, std.math.divCeil(i64, nk, 64) catch unreachable, 1 }, &.{
        .{ .name = "QI", .arg = try e.view(qi.t.role, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, a * IH * ID * 2) },
        .{ .name = "W", .arg = try e.view(iw.t.role, .bf16, &.{ rows, IH }, &.{ IH, 1 }, a * IH * 2) },
        .{ .name = "w_stride", .arg = .{ .i = IH } },
        .{ .name = "IK", .arg = try e.indexKeys(src) },
        .{ .name = "OUT", .arg = out },
        .{ .name = "POS", .arg = p },
        .{ .name = "KEYS", .arg = cand orelse out },
        .{ .name = "k_stride", .arg = .{ .i = if (cand != null) e.cfg.candidate_blocks else nk } },
        .{ .name = "NK", .arg = .{ .i = nk } },
        .{ .name = "o_stride", .arg = .{ .i = nk } },
        .{ .name = "RATIO", .arg = .{ .i = ratio } },
        .{ .name = "H", .arg = .{ .i = IH } },
        .{ .name = "D", .arg = .{ .i = ID } },
        .{ .name = "BP", .arg = .{ .i = 64 } },
        .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
        .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
        .{ .name = "GATHER", .arg = .{ .b = cand != null } },
        .{ .name = "PT", .arg = spg.pt },
        .{ .name = "PSH", .arg = .{ .i = spg.psh } },
        .{ .name = "KFP8", .arg = .{ .b = true } },
        .{ .name = "SL", .arg = .none },
        .{ .name = "PTS", .arg = .{ .i = 0 } },
        .{ .name = "ROWS", .arg = .{ .b = false } },
        .{ .name = "CBS", .arg = .{ .i = if (cand != null) cbs else 0 } },
    });
}

/// Row-blocked selection past the index budget: per block `_scores`, `_keys`, top positions; then counts.
pub fn blocked(e: *Emitter, L: u32, qi: Arg, iw: Arg, src: u32, ratio: i64, nvis: i64) !void {
    const n = e.n;
    const k: i64 = e.cfg.index_topk;
    const cb: i64 = e.cfg.candidate_blocks;
    const cbs: i64 = e.cfg.candidate_block_size;
    const sel = try e.selection(L);
    const cand_src = e.cfg.isCandidateSource(L);
    const cand = try e.buf("w.cand", .{}, .i32, &.{ n, cb });
    const st: i64 = if (n > 16) @max(16, @divFloor(e.o.index_budget, nvis * 12)) else n;
    var a: i64 = 0;
    var j: usize = 0;
    while (a < n) : ({
        a += st;
        j += 1;
    }) {
        const rows = @min(st, n - a);
        const p = if (st >= n) try e.pos() else try e.blockPos(j, a, rows);
        const nk = if (st >= n or e.o.rows) nvis else @max(@divFloor(e.start + a + rows, ratio), 1);
        const sc = try e.buf("L.scores", .{}, .f32, &.{ rows, nk });
        try e.scoreRows(qi, iw, src, ratio, a, rows, p, nk, sc, null);
        const keys = try e.buf("L.ix.keys", .{}, .i64, &.{ rows, nk });
        try e.triton("_keys", .{ rows, std.math.divCeil(i64, nk, 1024) catch unreachable, 1 }, &.{
            .{ .name = "S", .arg = sc },                   .{ .name = "s_stride", .arg = .{ .i = nk } },
            .{ .name = "K", .arg = keys },                 .{ .name = "k_stride", .arg = .{ .i = nk } },
            .{ .name = "POSN", .arg = sc },                .{ .name = "p_stride", .arg = .{ .i = nk } },
            .{ .name = "NK", .arg = .{ .i = nk } },        .{ .name = "BLOCK", .arg = .{ .i = 1024 } },
            .{ .name = "GATHER", .arg = .{ .b = false } },
        });
        try e.glue("topk", &.{ keys, try e.view(sel[0].t.role, .i32, &.{ rows, k }, &.{ k, 1 }, a * k * 4), .{ .i = k } });
        if (!cand_src) continue;
        const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
        const bk = try e.buf("L.ix.bkeys", .{}, .i64, &.{ rows, nb });
        try e.triton("_block_keys", .{ rows, std.math.divCeil(i64, nb, 256) catch unreachable, 1 }, &.{
            .{ .name = "S", .arg = sc },                  .{ .name = "s_stride", .arg = .{ .i = nk } },
            .{ .name = "K", .arg = bk },                  .{ .name = "k_stride", .arg = .{ .i = nb } },
            .{ .name = "POS", .arg = p },                 .{ .name = "NB", .arg = .{ .i = nb } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } }, .{ .name = "BS", .arg = .{ .i = cbs } },
            .{ .name = "TB", .arg = .{ .i = 256 } },      .{ .name = "ROWS", .arg = .{ .b = false } },
        });
        try e.glue("topk", &.{ bk, try e.view(cand.t.role, .i32, &.{ rows, cb }, &.{ cb, 1 }, a * cb * 4), .{ .i = cb } });
    }
    try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
}

/// Win.sub's first position (int32 [1]) of row block [a, a + rows); row mode reads the block's table rows instead.
pub fn blockPos(e: *Emitter, j: usize, a: i64, rows: i64) !Arg {
    const p = try e.buf("L.ix.pos{d}", .{j}, .i32, &.{1});
    try e.glue("block_pos", &.{ p, .{ .i = e.start + a }, .{ .i = a }, .{ .i = rows } });
    return p;
}

/// attn_cuda topk over `scores` [n, nk] of index layer L (up to two jobs a launch); candidate blocks into `cand`.
pub fn topk(e: *Emitter, L: u32, scores: Arg, cand: Arg, nk: i64, ratio: i64) !void {
    return e.topkOf(L, scores, cand, nk, ratio, false);
}

/// `topk`; `bound` (TF_DSV41_INDEX_BOUND): dense select and candidate blocks as the row-bounded twins, same bytes.
pub fn topkOf(e: *Emitter, L: u32, scores: Arg, cand: Arg, nk: i64, ratio: i64, bound: bool) !void {
    const n = e.n;
    const dtopk = if (bound) "_dtopk_b" else "_dtopk";
    const topk_ext = if (bound and e.cfg.mode(L) != .reindex) "tf_dsv41_topk_b_v1.topk" else "tf_dsv41_attn_cuda_v1.topk";
    const reindex = e.cfg.mode(L) == .reindex;
    const cbs: i64 = e.cfg.candidate_block_size;
    const sel = try e.selection(L);
    const k: i64 = e.cfg.index_topk;
    const pl = topkPlan(nk, null);
    if (pl[1] > topk_max_ept) {
        // past attn_cuda's plan limit (8 x 512 x 31 entries): Triton _dtopk with counts, then candidate blocks
        if (reindex) return error.Unsupported;
        try e.triton(dtopk, .{ n, 1, 1 }, &.{
            .{ .name = "S", .arg = scores },             .{ .name = "s_stride", .arg = .{ .i = scores.t.stride[0] } },
            .{ .name = "P", .arg = scores },             .{ .name = "p_stride", .arg = .{ .i = 0 } },
            .{ .name = "POS", .arg = try e.pos() },      .{ .name = "NK", .arg = .{ .i = nk } },
            .{ .name = "OUT", .arg = sel[0] },           .{ .name = "o_stride", .arg = .{ .i = k } },
            .{ .name = "CNT", .arg = sel[1] },           .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "K", .arg = .{ .i = k } },        .{ .name = "MODE", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } }, .{ .name = "T", .arg = .{ .i = 4096 } },
            .{ .name = "BS", .arg = .{ .i = 8 } },       .{ .name = "RB", .arg = .{ .i = 8 } },
            .{ .name = "NPASS", .arg = .{ .i = 8 } },    .{ .name = "SORT", .arg = .{ .b = false } },
            .{ .name = "COUNT", .arg = .{ .b = true } }, .{ .name = "KP", .arg = .{ .i = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(k)) catch unreachable) } },
        });
        if (!e.cfg.isCandidateSource(L)) return;
        const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
        const bp = topkPlan(nb, null);
        if (bp[1] > topk_max_ept) {
            // past 8 x 512 x 31 blocks too (positions past ~1,015,808): dtopk.blocks (_dtopk mode 2, no counts)
            const cb: i64 = e.cfg.candidate_blocks;
            return e.triton(dtopk, .{ n, 1, 1 }, &.{
                .{ .name = "S", .arg = scores },              .{ .name = "s_stride", .arg = .{ .i = scores.t.stride[0] } },
                .{ .name = "P", .arg = scores },              .{ .name = "p_stride", .arg = .{ .i = 0 } },
                .{ .name = "POS", .arg = try e.pos() },       .{ .name = "NK", .arg = .{ .i = nb } },
                .{ .name = "OUT", .arg = cand },              .{ .name = "o_stride", .arg = .{ .i = cand.t.stride[0] } },
                .{ .name = "CNT", .arg = cand },              .{ .name = "RATIO", .arg = .{ .i = ratio } },
                .{ .name = "K", .arg = .{ .i = cb } },        .{ .name = "MODE", .arg = .{ .i = 2 } },
                .{ .name = "ROWS", .arg = .{ .b = false } },  .{ .name = "T", .arg = .{ .i = 4096 } },
                .{ .name = "BS", .arg = .{ .i = cbs } },      .{ .name = "RB", .arg = .{ .i = 8 } },
                .{ .name = "NPASS", .arg = .{ .i = 8 } },     .{ .name = "SORT", .arg = .{ .b = false } },
                .{ .name = "COUNT", .arg = .{ .b = false } }, .{ .name = "KP", .arg = .{ .i = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(cb)) catch unreachable) } },
            });
        }
        const job: [8]Arg = .{ scores, try e.empty(.i32), cand, try e.view(cand.t.role, .i32, &.{0}, &.{e.cfg.candidate_blocks}, 0), .{ .i = nb }, .{ .i = e.cfg.candidate_blocks }, .{ .i = 2 }, .{ .i = bp[1] } };
        var one: std.ArrayList(Arg) = .empty;
        try one.appendSlice(e.a, &job);
        try one.appendSlice(e.a, &job);
        try one.appendSlice(e.a, &.{ .{ .i = 1 }, try e.pos(), .{ .i = ratio }, .{ .i = cbs }, .{ .i = n }, .{ .i = bp[0] } });
        return e.ext(topk_ext, one.items);
    }
    var jobs: [2][8]Arg = undefined;
    jobs[0] = .{ scores, if (reindex) cand else try e.empty(.i32), sel[0], sel[1], .{ .i = nk }, .{ .i = k }, .{ .i = if (reindex) 3 else 0 }, .{ .i = pl[1] } };
    const blocks = e.cfg.isCandidateSource(L);
    jobs[1] = jobs[0];
    if (blocks) {
        const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
        jobs[1] = .{ scores, try e.empty(.i32), cand, try e.view(cand.t.role, .i32, &.{0}, &.{e.cfg.candidate_blocks}, 0), .{ .i = nb }, .{ .i = e.cfg.candidate_blocks }, .{ .i = 2 }, .{ .i = topkPlan(nb, pl[0])[1] } };
    }
    var args: std.ArrayList(Arg) = .empty;
    try args.appendSlice(e.a, &jobs[0]);
    try args.appendSlice(e.a, &jobs[1]);
    try args.appendSlice(e.a, &.{ .{ .i = if (blocks) 2 else 1 }, try e.pos(), .{ .i = ratio }, .{ .i = if (blocks or reindex) cbs else 1 }, .{ .i = n }, .{ .i = pl[0] } });
    try e.ext(topk_ext, args.items);
}
