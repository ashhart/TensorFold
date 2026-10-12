//! A prefill segment's index selection: indexer scores, candidates, top-k (whole or streamed).

const calls = @import("block_calls.zig");
const longpf = @import("longpf.zig");
const prefill = @import("block_prefill.zig");
const E = prefill.E;
const Arg = calls.Arg;
const Dt = calls.Dt;
const stream_min = prefill.stream_min;
const dtopk_rows = prefill.dtopk_rows;
const size = prefill.size;
const cdiv = prefill.cdiv;
const Seg = prefill.Seg;
const pos = prefill.Seg.pos;
const indexKeys = prefill.Seg.indexKeys;

/// The segment's selection of index layer `src` (reuse layers read the latest source's).
pub fn selection(e: *E, src: u32) ![2]Arg {
    const k: i64 = e.cfg.index_topk;
    return .{
        try e.view(.{ .buf = try e.fmt("w.sel{d}", .{src}) }, .i32, &.{ e.n, k }, &.{ k, 1 }, e.row0 * k * 4),
        // the counts' 4-byte rows: a run's segment from a row a multiple of 4 (cnt0), so the slice is 16-aligned
        try e.view(.{ .buf = try e.fmt("w.cnt{d}", .{src}) }, .i32, &.{e.n}, &.{1}, e.cnt0 * 4),
    };
}

/// The segment's rows of the run's per-row candidates (layer 20's, read by the reindex layers).
fn cands(e: *E) !Arg {
    const cb: i64 = e.cfg.candidate_blocks;
    return e.view(.{ .buf = "w.cand" }, .i32, &.{ e.n, cb }, &.{ cb, 1 }, e.row0 * cb * 4);
}

/// Rows [a, a + rows) of a row-major [n, cols] role.
pub fn rowsOf(e: *E, x: Arg, dt: Dt, a: i64, rows: i64, cols: i64) !Arg {
    return e.view(x.t.role, dt, &.{ rows, cols }, &.{ cols, 1 }, x.t.offset + a * cols * size(dt));
}

/// Index scores over rows [a, a + rows): dense, or over `keys`' positions (`cbs` > 0: candidate blocks).
pub fn scores(s: *Seg, keys_of: u32, ratio: i64, a: i64, rows: i64, p: Arg, nk: i64, keys: ?Arg, cbs: i64) !Arg {
    const e = s.e;
    const IH: i64 = e.cfg.index_heads;
    const ID: i64 = e.cfg.index_dim;
    const out = try e.buf("L.ix.scores", .{}, .f32, &.{ rows, nk });
    const spg = try e.paging(ratio, false);
    try e.triton("_scores", .{ rows, cdiv(nk, 64), 1 }, &.{
        .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, (e.row0 + a) * IH * ID * 2) },
        .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, (e.row0 + a) * IH * 4) },
        .{ .name = "w_stride", .arg = .{ .i = IH } },
        .{ .name = "IK", .arg = try indexKeys(e, keys_of) },
        .{ .name = "OUT", .arg = out },
        .{ .name = "POS", .arg = p },
        .{ .name = "KEYS", .arg = keys orelse out },
        .{ .name = "k_stride", .arg = .{ .i = if (cbs > 0) keys.?.t.shape[1] else nk } },
        .{ .name = "NK", .arg = .{ .i = nk } },
        .{ .name = "o_stride", .arg = .{ .i = nk } },
        .{ .name = "RATIO", .arg = .{ .i = ratio } },
        .{ .name = "H", .arg = .{ .i = IH } },
        .{ .name = "D", .arg = .{ .i = ID } },
        .{ .name = "BP", .arg = .{ .i = 64 } },
        .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
        .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
        .{ .name = "GATHER", .arg = .{ .b = keys != null } },
        .{ .name = "PT", .arg = spg.pt },
        .{ .name = "PSH", .arg = .{ .i = spg.psh } },
        .{ .name = "KFP8", .arg = .{ .b = true } },
        .{ .name = "SL", .arg = .none },
        .{ .name = "PTS", .arg = .{ .i = 0 } },
        .{ .name = "ROWS", .arg = .{ .b = false } },
        .{ .name = "CBS", .arg = .{ .i = cbs } },
    });
    return out;
}

/// index.sort_keys: a unique int64 key a score (`posn`: the gathered keys' positions).
pub fn sortKeys(s: *Seg, sc: Arg, rows: i64, nk: i64, posn: ?Arg) !Arg {
    const e = s.e;
    const k = try e.buf("L.ix.keys", .{}, .i64, &.{ rows, nk });
    try e.triton("_keys", .{ rows, cdiv(nk, 1024), 1 }, &.{
        .{ .name = "S", .arg = sc },                          .{ .name = "s_stride", .arg = .{ .i = nk } },
        .{ .name = "K", .arg = k },                           .{ .name = "k_stride", .arg = .{ .i = nk } },
        .{ .name = "POSN", .arg = posn orelse sc },           .{ .name = "p_stride", .arg = .{ .i = nk } },
        .{ .name = "NK", .arg = .{ .i = nk } },               .{ .name = "BLOCK", .arg = .{ .i = 1024 } },
        .{ .name = "GATHER", .arg = .{ .b = posn != null } },
    });
    return k;
}

/// The candidate blocks' keys from the scores, then the top-k (glue) into rows [a, ..) of the candidates.
pub fn candidates(s: *Seg, sc: Arg, a: i64, rows: i64, nk: i64, p: Arg, ratio: i64) !void {
    const e = s.e;
    const cbs: i64 = e.cfg.candidate_block_size;
    const cb: i64 = e.cfg.candidate_blocks;
    const nb = cdiv(nk, cbs);
    const k = try e.buf("L.ix.bkeys", .{}, .i64, &.{ rows, nb });
    try e.triton("_block_keys", .{ rows, cdiv(nb, 256), 1 }, &.{
        .{ .name = "S", .arg = sc },                  .{ .name = "s_stride", .arg = .{ .i = nk } },
        .{ .name = "K", .arg = k },                   .{ .name = "k_stride", .arg = .{ .i = nb } },
        .{ .name = "POS", .arg = p },                 .{ .name = "NB", .arg = .{ .i = nb } },
        .{ .name = "RATIO", .arg = .{ .i = ratio } }, .{ .name = "BS", .arg = .{ .i = cbs } },
        .{ .name = "TB", .arg = .{ .i = 256 } },      .{ .name = "ROWS", .arg = .{ .b = false } },
    });
    try e.glue("topk", &.{ k, try rowsOf(e, try cands(e), .i32, a, rows, cb), .{ .i = cb } });
}

/// Row blocks of the materialised indexer (backend._row_blocks): `per_row` bytes a row under the index budget.
fn step(e: *E, per_row: i64) i64 {
    return if (e.n > 16) @max(16, @divFloor(e.o.index_budget, @max(1, per_row))) else e.n;
}

/// An index layer's selection (blocks._select over backend.select / select_cand / reindex).
pub fn select(s: *Seg, L: u32) !void {
    const e = s.e;
    const n = e.n;
    const md = e.cfg.mode(L);
    const src = e.cfg.kvSource(L).?;
    const ratio: i64 = e.cfg.compressRatio(L);
    const topk: i64 = e.cfg.index_topk;
    const cbs: i64 = e.cfg.candidate_block_size;
    const cb: i64 = e.cfg.candidate_blocks;
    const sel = try selection(e, L);
    // <= 64 rows with scores within the index budget: attn_cuda's top-k; past the stream threshold: stream top-k
    const reindexing = md == .reindex and e.cfg.usesCandidates(L) and s.cand;
    const nvis = @max(@divFloor(e.end(), ratio), 1);
    const streamed = !e.cfg.isCandidateSource(L) and !reindexing and n > 16 and nvis >= stream_min and @popCount(topk) == 1;
    const fits = if (reindexing) n * cb * cbs * 16 <= e.o.index_budget else n * nvis * 12 <= e.o.index_budget;
    if (n <= dtopk_rows and fits and !streamed) {
        // the whole segment's scores, then attn_cuda's top-k (over a source's candidate blocks) and visible counts
        const cand = try cands(e);
        if (reindexing) {
            const sc = try s.scores(src, ratio, 0, n, try pos(e), cb * cbs, cand, cbs);
            return e.topk(L, sc, cand, cb * cbs, ratio);
        }
        const sc = try s.scores(src, ratio, 0, n, try pos(e), nvis, null, 0);
        try e.topk(L, sc, cand, nvis, ratio);
        if (e.cfg.isCandidateSource(L)) s.cand = true;
        return;
    }
    if (md == .reindex and e.cfg.usesCandidates(L) and s.cand) {
        // backend.reindex: the row's own scores over its candidates' positions, in row blocks under the budget
        const NK = cb * cbs;
        const whole = n * NK * 16 <= e.o.index_budget;
        const st = if (whole) n else step(e, NK * 16);
        var a: i64 = 0;
        var j: usize = 0;
        while (a < n) : ({
            a += st;
            j += 1;
        }) {
            const rows = @min(st, n - a);
            const p = if (whole) try pos(e) else try s.blockPos(j, a);
            const ck = try e.buf("L.ix.ckeys", .{}, .i32, &.{ rows, NK });
            try e.glue("cand_keys", &.{ try rowsOf(e, try cands(e), .i32, a, rows, cb), ck, .{ .i = cbs } });
            const sc = try s.scores(src, ratio, a, rows, p, NK, ck, 0);
            try e.glue("topk", &.{ try s.sortKeys(sc, rows, NK, ck), try rowsOf(e, sel[0], .i32, a, rows, topk), .{ .i = topk } });
        }
    } else {
        const cand = e.cfg.isCandidateSource(L);
        // backend.select's stream path (full-mode layers without the candidates' scores): stream_topk.select
        if (streamed) {
            try s.streamSelect(src, ratio, topk, sel[0]);
            try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
            return;
        }
        const whole = n * nvis * 12 <= e.o.index_budget;
        const st = if (whole) n else step(e, nvis * 12);
        var a: i64 = 0;
        var j: usize = 0;
        while (a < n) : ({
            a += st;
            j += 1;
        }) {
            const rows = @min(st, n - a);
            const p = if (whole) try pos(e) else try s.blockPos(j, a);
            const nk = if (whole) nvis else @max(@divFloor(e.start + a + rows, ratio), 1);
            const sc = try s.scores(src, ratio, a, rows, p, nk, null, 0);
            try e.glue("topk", &.{ try s.sortKeys(sc, rows, nk, null), try rowsOf(e, sel[0], .i32, a, rows, topk), .{ .i = topk } });
            if (cand) try s.candidates(sc, a, rows, nk, p, ratio);
        }
        if (cand) s.cand = true;
    }
    try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
}

/// Stream top-k in row blocks within its scratch budget: each block's `_stream`, then the merge glue.
pub fn streamSelect(s: *Seg, src: u32, ratio: i64, topk: i64, out: Arg) !void {
    const e = s.e;
    const n = e.n;
    const IH: i64 = e.cfg.index_heads;
    const ID: i64 = e.cfg.index_dim;
    const st = longpf.plan(n, @max(@divFloor(e.end(), ratio), 1), topk);
    const spg = try e.paging(ratio, false);
    var a: i64 = 0;
    var j: usize = 0;
    while (a < n) : ({
        a += st;
        j += 1;
    }) {
        const rows = @min(st, n - a);
        const p = if (st >= n) try pos(e) else try s.blockPos(j, a);
        const nk = @max(@divFloor(e.start + a + rows, ratio), 1);
        const ns = longpf.splits(nk);
        const buf = try e.buf("L.ix.stream", .{}, .i64, &.{ rows, ns, 2 * topk });
        if (e.o.stream_rb == 1 or e.o.stream_rb == 2 or e.o.stream_rb == 4) {
            // TF_DSV41_STREAM_RB: `_stream_pf` (1) or `_stream_rb<N>`, N rows a program (same buffers and merge)
            const rb = e.o.stream_rb;
            try e.triton(switch (rb) {
                1 => "_stream_pf",
                2 => "_stream_rb2",
                else => "_stream_rb4",
            }, .{ cdiv(rows, rb), ns, 1 }, &.{
                .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, (e.row0 + a) * IH * ID * 2) },
                .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, (e.row0 + a) * IH * 4) },
                .{ .name = "w_stride", .arg = .{ .i = IH } },
                .{ .name = "IK", .arg = try indexKeys(e, src) },
                .{ .name = "POS", .arg = p },
                .{ .name = "R", .arg = .{ .i = rows } },
                .{ .name = "BUF", .arg = buf },
                .{ .name = "nsplit", .arg = .{ .i = ns } },
                .{ .name = "RATIO", .arg = .{ .i = ratio } },
                .{ .name = "H", .arg = .{ .i = IH } },
                .{ .name = "D", .arg = .{ .i = ID } },
                .{ .name = "BP", .arg = .{ .i = 64 } },
                .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
                .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
                .{ .name = "SPLIT", .arg = .{ .i = longpf.split_keys } },
                .{ .name = "K", .arg = .{ .i = topk } },
                .{ .name = "CAP", .arg = .{ .i = 2 * topk } },
                .{ .name = "PT", .arg = spg.pt },
                .{ .name = "PSH", .arg = .{ .i = spg.psh } },
                .{ .name = "KFP8", .arg = .{ .b = true } },
            });
            try e.glue("stream_merge", &.{ buf, try rowsOf(e, out, .i32, a, rows, topk), .{ .i = topk }, try e.buf("L.ix.keys", .{}, .i64, &.{ rows, ns * topk }) });
            continue;
        }
        try e.triton("_stream", .{ rows, ns, 1 }, &.{
            .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, (e.row0 + a) * IH * ID * 2) },
            .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, (e.row0 + a) * IH * 4) },
            .{ .name = "w_stride", .arg = .{ .i = IH } },
            .{ .name = "IK", .arg = try indexKeys(e, src) },
            .{ .name = "POS", .arg = p },
            .{ .name = "KEYS", .arg = buf },
            .{ .name = "k_stride", .arg = .{ .i = 0 } },
            .{ .name = "NK", .arg = .{ .i = nk } },
            .{ .name = "BUF", .arg = buf },
            .{ .name = "nsplit", .arg = .{ .i = ns } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "H", .arg = .{ .i = IH } },
            .{ .name = "D", .arg = .{ .i = ID } },
            .{ .name = "BP", .arg = .{ .i = 64 } },
            .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
            .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
            .{ .name = "MODE", .arg = .{ .i = 0 } },
            .{ .name = "SPLIT", .arg = .{ .i = longpf.split_keys } },
            .{ .name = "K", .arg = .{ .i = topk } },
            .{ .name = "CAP", .arg = .{ .i = 2 * topk } },
            .{ .name = "BS", .arg = .{ .i = 8 } }, // index.BLOCK
            .{ .name = "PT", .arg = spg.pt },
            .{ .name = "PSH", .arg = .{ .i = spg.psh } },
            .{ .name = "KFP8", .arg = .{ .b = true } },
        });
        // the merge's gathered keys [rows, ns x topk] share the materialised path's sort keys' storage
        try e.glue("stream_merge", &.{ buf, try rowsOf(e, out, .i32, a, rows, topk), .{ .i = topk }, try e.buf("L.ix.keys", .{}, .i64, &.{ rows, ns * topk }) });
    }
}
