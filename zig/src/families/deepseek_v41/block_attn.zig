//! A decode window's attention: rope, the KV stores and pools, the compressed exchange, SWA and CSA2 launches.

const calls = @import("block_calls.zig");
const block = @import("block.zig");
const Arg = calls.Arg;
const Pool = block.Pool;
const Emitter = block.Emitter;
const dec = block.Emitter.dec;
const Proj = block.Emitter.Proj;
const Norm = block.Emitter.Norm;
const Paging = block.Emitter.Paging;
const Received = block.Emitter.Received;
const attnSplit = block.Emitter.attnSplit;

// attention ----------------------------------------------------------------------------------------------------

pub fn rope(e: *Emitter, x: Arg, heads: i64, dim: i64, nope: i64) !void {
    const n = e.n;
    const xv = try e.view(x.t.role, .bf16, &.{ n, heads, dim }, &.{ heads * dim, dim, 1 }, 0);
    try e.triton("_rope", .{ n, @divExact(heads, 4), 1 }, &.{
        .{ .name = "X", .arg = xv },                                       .{ .name = "x_rs", .arg = .{ .i = heads * dim } },
        .{ .name = "x_hs", .arg = .{ .i = dim } },                         .{ .name = "OUT", .arg = xv },
        .{ .name = "o_rs", .arg = .{ .i = heads * dim } },                 .{ .name = "o_hs", .arg = .{ .i = dim } },
        .{ .name = "CS", .arg = try e.ropeTable() },                       .{ .name = "cs_rs", .arg = .{ .i = 64 } },
        .{ .name = "POS", .arg = try e.buf("w.pos64", .{}, .i64, &.{n}) }, .{ .name = "H", .arg = .{ .i = heads } },
        .{ .name = "NOPE", .arg = .{ .i = nope } },                        .{ .name = "NB", .arg = .{ .i = 64 } },
        .{ .name = "HALF", .arg = .{ .i = 32 } },                          .{ .name = "BH", .arg = .{ .i = 4 } },
        .{ .name = "INV", .arg = .{ .b = false } },                        .{ .name = "COPY", .arg = .{ .b = false } },
        .{ .name = "PDL", .arg = .{ .b = false } },
    });
}

/// The rope table (cos / sin) a layer uses: the compressed layers' (compress_theta) or the SWA layers'.
pub fn ropeTable(e: *Emitter) !Arg {
    return e.buf("s.rope.{s}", .{if (e.layer_ratio > 0) "comp" else "main"}, .f32, &.{ e.o.rope_rows, 64 });
}

pub fn pos(e: *Emitter) !Arg {
    return e.buf("w.pos", .{}, .i32, &.{1});
}

/// The KV store (_kv_store): the SWA ring or a compressed family (split KV: only the rows this rank owns).
pub fn kvStore(e: *Emitter, lat: Arg, src: u32, ratio: i64) !void {
    const rows = try e.pool(src, ratio);
    const pg = try e.paging(ratio, true);
    try e.triton("_kv_store", .{ e.n, 1, 1 }, &.{
        .{ .name = "LAT", .arg = lat },                                                          .{ .name = "l_stride", .arg = .{ .i = e.cfg.head_dim } },
        .{ .name = "CS", .arg = try e.ropeTable() },                                             .{ .name = "cs_stride", .arg = .{ .i = 64 } },
        .{ .name = "V", .arg = rows[0] },                                                        .{ .name = "S", .arg = rows[1] },
        .{ .name = "v_stride", .arg = .{ .i = pg.vs } },                                         .{ .name = "s_stride", .arg = .{ .i = pg.ss } },
        .{ .name = "POS", .arg = try e.pos() },                                                  .{ .name = "RATIO", .arg = .{ .i = ratio } },
        .{ .name = "RING", .arg = .{ .i = if (ratio == 0) 2 * @as(i64, e.cfg.window) else 1 } }, .{ .name = "PT", .arg = pg.pt },
        .{ .name = "PSH", .arg = .{ .i = pg.psh } },                                             .{ .name = "SL", .arg = .none },
        .{ .name = "PTS", .arg = .{ .i = 0 } },                                                  .{ .name = "ROWS", .arg = .{ .b = false } },
        .{ .name = "PDL", .arg = .{ .b = false } },
    });
}

pub fn paging(e: *Emitter, ratio: i64, store: bool) !Paging {
    const p = e.o.pool orelse return .{ .pt = .none, .psh = 0, .vs = 576, .ss = 8 };
    if (ratio == 0) return .{ .pt = .none, .psh = 0, .vs = 576, .ss = 8 };
    const tab: []const u8 = if (store and p.split) "s.kv.ct" else "s.kv.pt";
    return .{ .pt = try e.buf("{s}", .{tab}, .i32, &.{p.pts}), .psh = Pool.shift(ratio), .vs = Pool.row_bytes, .ss = Pool.row_bytes };
}

/// A layer's KV rows (576 B) and scales (8 B): SWA ring, contiguous compressed rows, or the pool's 584 B records.
pub fn pool(e: *Emitter, L: u32, ratio: i64) ![2]Arg {
    if (ratio > 0) if (e.o.pool) |p| {
        const rows = p.comp_pages * @divExact(Pool.page, ratio);
        const role: calls.Role = .{ .buf = try e.fmt("s.kv.comp.L{d}", .{L}) };
        return .{ try e.view(role, .u8, &.{ rows, 576 }, &.{ Pool.row_bytes, 1 }, 0), try e.view(role, .u8, &.{ rows, 8 }, &.{ Pool.row_bytes, 1 }, 576) };
    };
    const rows: i64 = if (ratio == 0) 2 * @as(i64, e.cfg.window) else @divFloor(e.o.limit, ratio) + 1;
    const kind = if (ratio == 0) "swa" else "comp";
    return .{ try e.buf("s.L{d}.{s}.v", .{ L, kind }, .u8, &.{ rows, 576 }), try e.buf("s.L{d}.{s}.s", .{ L, kind }, .u8, &.{ rows, 8 }) };
}

pub fn indexKeys(e: *Emitter, L: u32) !Arg {
    const r: i64 = e.cfg.compressRatio(L);
    if (e.o.pool) |p| return e.buf("s.kv.ik.L{d}", .{L}, .u8, &.{ p.ik_pages * @divExact(Pool.page, r), Pool.ik_bytes });
    return e.buf("s.L{d}.ik", .{L}, .u8, &.{ @divFloor(e.o.limit, r) + 1, @as(i64, e.cfg.index_dim) + 4 });
}

/// The rows and tokens of the last exchange (a reuse layer: the same selection and kv source).
pub fn received(e: *Emitter, src: u32, ratio: i64) !Received {
    _ = src;
    _ = ratio;
    const p = e.o.pool.?;
    const n = e.n;
    const K: i64 = e.cfg.index_topk;
    const W: i64 = e.o.world;
    const sent: i64 = if (n <= 16) n * K else @divFloor(p.union_cap, (W + 1) * Pool.row_bytes);
    const recv: calls.Role = .{ .buf = "w.kx.recv" };
    return .{
        .cv = try e.view(recv, .u8, &.{ W * sent, 576 }, &.{ Pool.row_bytes, 1 }, 0),
        .cs = try e.view(recv, .u8, &.{ W * sent, 8 }, &.{ Pool.row_bytes, 1 }, 576),
        .tok = try e.buf("w.kx.tok", .{}, .i32, &.{ n, K }),
    };
}

pub fn exchange(e: *Emitter, src: u32, ratio: i64, sel: Arg) !Received {
    const p = e.o.pool.?;
    const n = e.n;
    const K: i64 = e.cfg.index_topk;
    const W: i64 = e.o.world;
    // row windows exchange dense at any rows, as the reference does
    const small = n <= 16 or e.o.rows;
    // dense: every rank sends R x K rows; union: (W + 1) x M rows under the cap
    const sent: i64 = if (small) n * K else @divFloor(p.union_cap, (W + 1) * Pool.row_bytes);
    const recv = try e.buf("w.kx.recv", .{}, .u8, &.{ W * sent, Pool.row_bytes });
    const tok = try e.buf("w.kx.tok", .{}, .i32, &.{ n, K });
    const comp = try e.buf("s.kv.comp.L{d}", .{src}, .u8, &.{ p.comp_pages * @divExact(Pool.page, ratio), Pool.row_bytes });
    try e.glue(if (small) "kx_dense" else "kx_union", &.{
        sel,                                        comp,
        try e.buf("s.kv.ct", .{}, .i32, &.{p.pts}), try e.buf("w.kx.send", .{}, .u8, &.{ sent, Pool.row_bytes }),
        recv,                                       tok,
        .{ .i = Pool.shift(ratio) },
    });
    return .{
        .cv = try e.view(recv.t.role, .u8, &.{ W * sent, 576 }, &.{ Pool.row_bytes, 1 }, 0),
        .cs = try e.view(recv.t.role, .u8, &.{ W * sent, 8 }, &.{ Pool.row_bytes, 1 }, 576),
        .tok = tok,
    };
}

/// Triton `_rms` (rmsnorm.rms: a weight, bf16 in and out): `r1`'s kv_norm beside the SWA store.
pub fn rms(e: *Emitter, x: Arg, w: Arg, out: Arg, k: i64) !void {
    try e.triton("_rms", .{ e.n, 1, 1 }, &.{
        .{ .name = "X", .arg = x },                                              .{ .name = "x_rs", .arg = .{ .i = k } },
        .{ .name = "W", .arg = w },                                              .{ .name = "OUT", .arg = out },
        .{ .name = "o_rs", .arg = .{ .i = k } },                                 .{ .name = "K", .arg = .{ .i = k } },
        .{ .name = "inv_k", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(k)) } }, .{ .name = "eps", .arg = .{ .f = dec(e.cfg.eps) } },
        .{ .name = "BK", .arg = .{ .i = 512 } },                                 .{ .name = "HAS_W", .arg = .{ .b = true } },
        .{ .name = "NARROW", .arg = .{ .b = true } },                            .{ .name = "PDL", .arg = .{ .b = false } },
    });
}

/// Only the 512-wide KV producer belongs to this fusion; other RMS norms keep their calls.
pub fn kvNormStoreOn(e: *const Emitter) bool {
    return e.o.kv_norm_store and e.cfg.head_dim == 512 and e.n >= 1 and e.n <= 64;
}

pub fn normStore(e: *Emitter, x: Arg, w: Arg, v: Arg, s: Arg, p: Arg, sl: Arg, ring: i64, rows: bool) !void {
    try e.ext("tf_dsv41_kv_glue_v1.norm_store", &.{
        x,              w,            try e.ropeTable(), v,           s,                     p,                        sl,
        .{ .i = 512 },  .{ .i = 64 }, .{ .i = 576 },     .{ .i = 8 }, .{ .f = 1.0 / 512.0 }, .{ .f = dec(e.cfg.eps) }, .{ .i = ring },
        .{ .b = rows },
    });
}

/// The SWA store of the window's kv rows (`r1`: kv_norm runs here, RMS_FOLD's `Normed.tensor()`).
pub fn swaStore(e: *Emitter, L: u32, kv: Arg, kvn: Arg) !void {
    if (e.o.r1 and e.kvNormStoreOn()) {
        const rows = try e.pool(L, 0);
        try e.normStore(kv, try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{e.cfg.head_dim}), rows[0], rows[1], try e.pos(), .none, 2 * @as(i64, e.cfg.window), false);
    } else {
        if (e.o.r1) try e.rms(kv, try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{e.cfg.head_dim}), kvn, e.cfg.head_dim);
        try e.kvStore(kvn, L, 0);
    }
}

/// branches.plan's marks (`fork`, `onMain`, `join`); nothing unless TF_DSV41_BRANCHES branches this window
pub fn branching(e: *const Emitter) bool {
    return e.o.branches and e.n <= e.o.branch_rows;
}

pub fn fork(e: *Emitter) void {
    if (!e.branching()) return;
    e.br_fork = true;
    e.br_side = true;
}

pub fn onMain(e: *Emitter) void {
    e.br_side = false;
}

pub fn join(e: *Emitter) void {
    if (!e.branching()) return;
    e.br_join = true;
    e.br_side = false;
}

pub fn attention(e: *Emitter, L: u32) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const md = e.cfg.mode(L);
    const ratio: i64 = e.cfg.compressRatio(L);
    const kv_src = e.cfg.kvSource(L);
    const H: i64 = e.cfg.heads / e.o.world;
    const hd: i64 = e.cfg.head_dim;
    const IH: i64 = e.cfg.index_heads;
    const ID: i64 = e.cfg.index_dim;
    const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
    const indexer = md == .full or md == .reindex;
    const own_kv = md == .full;
    // the indexer's head weights (a bf16 GEMV over the normed input)
    const iw = try e.buf("L.ix_w", .{}, .bf16, &.{ n, IH });
    // index layer: fork, side indexer chain, main q group, join, attn; other layers: side SWA store, main q group
    if (indexer) e.fork();
    if (indexer) try e.triton("_plain", .{ n, IH, 1 }, &.{
        .{ .name = "X", .arg = x },                                                        .{ .name = "x_rs", .arg = .{ .i = D } },
        .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_wp", .{L}, .f32, &.{ IH, D }) }, .{ .name = "w_rs", .arg = .{ .i = D } },
        .{ .name = "OUT", .arg = iw },                                                     .{ .name = "K", .arg = .{ .i = D } },
        .{ .name = "N", .arg = .{ .i = IH } },                                             .{ .name = "BK", .arg = .{ .i = 256 } },
        .{ .name = "PDL", .arg = .{ .b = false } },                                        .{ .name = "BF16", .arg = .{ .b = true } },
    });
    e.onMain();
    // x group: wq_a, wkv (+ the compressor's on a KV source)
    const xg = try e.xGroup(L);
    const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, e.cfg.q_lora });
    const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
    const cw: i64 = if (ratio == 2) 2 * hd else hd;
    // a KV source's compressor projection outlives its layer: keep() carries the accepted row (ratio 2)
    const comp = if (ratio == 2) try e.buf("w.L{d}.comp", .{L}, .bf16, &.{ n, cw }) else try e.buf("L.comp", .{}, .bf16, &.{ n, cw });
    var ps: [4]Proj = undefined;
    for (xg.ws, 0..) |w, i| ps[i] = .{ .w = w, .x = x, .k = @intCast(D), .n = @intCast(xg.ns[i]), .y = switch (i) {
        0 => qa,
        1 => kv,
        else => comp,
    }, .col = if (i == 3) @intCast(hd) else 0 };
    try e.fused("x", ps[0..xg.ws.len]);
    const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, e.cfg.q_lora });
    const kvn = try e.buf("L.kvn", .{}, .bf16, &.{ n, hd });
    // `r1` (RMS_FOLD): no launch here; q_norm runs in its consumers' rot_in, kv_norm beside the SWA store
    const qnorm: ?Norm = if (e.o.r1) .{ .w = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}), .eps = dec(e.cfg.eps) } else null;
    if (!e.o.r1) try e.triton("_rms2", .{ n, 2, 1 }, &.{
        .{ .name = "XA", .arg = qa },                                                             .{ .name = "xa_rs", .arg = .{ .i = e.cfg.q_lora } },
        .{ .name = "WA", .arg = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}) }, .{ .name = "OA", .arg = qn },
        .{ .name = "oa_rs", .arg = .{ .i = e.cfg.q_lora } },                                      .{ .name = "KA", .arg = .{ .i = e.cfg.q_lora } },
        .{ .name = "inv_ka", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(e.cfg.q_lora)) } },      .{ .name = "XB", .arg = kv },
        .{ .name = "xb_rs", .arg = .{ .i = hd } },                                                .{ .name = "WB", .arg = try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}) },
        .{ .name = "OB", .arg = kvn },                                                            .{ .name = "ob_rs", .arg = .{ .i = hd } },
        .{ .name = "KB", .arg = .{ .i = hd } },                                                   .{ .name = "inv_kb", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(hd)) } },
        .{ .name = "eps", .arg = .{ .f = dec(e.cfg.eps) } },                                      .{ .name = "BK", .arg = .{ .i = 512 } },
        .{ .name = "NARROW", .arg = .{ .b = true } },                                             .{ .name = "PDL", .arg = .{ .b = false } },
    });
    if (!indexer) {
        e.fork();
        try e.swaStore(L, kv, kvn); // `r1` (RMS_FOLD): its kv_norm with it, on the side
        e.onMain();
    }
    if (indexer) e.fork();
    if (own_kv) {
        // the compressor: pooled rows, normed, into the pool; the indexer's keys from the same rows
        const lat = try e.buf("L.lat", .{}, .bf16, &.{ n, hd });
        try e.triton("_pool_norm", .{ n, 1, 1 }, &.{
            .{ .name = "BUF", .arg = comp },                                                  .{ .name = "b_stride", .arg = .{ .i = cw } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.comp_norm", .{L}, .f32, &.{hd}) }, .{ .name = "LAT", .arg = lat },
            .{ .name = "POS", .arg = try e.pos() },                                           .{ .name = "n", .arg = .{ .i = n } },
            .{ .name = "EPS", .arg = .{ .f = dec(e.cfg.eps) } },                              .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "SL", .arg = .none },                                                  .{ .name = "PREV", .arg = .none },
            .{ .name = "OFF", .arg = .{ .i = if (ratio == 2) 1 else 0 } },                    .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },                                       .{ .name = "CARRY", .arg = if (ratio == 2) try e.buf("s.L{d}.carry", .{L}, .f32, &.{ 1, cw }) else .none },
            .{ .name = "c_stride", .arg = .{ .i = if (ratio == 2) cw else 0 } },              .{ .name = "SPLIT", .arg = .{ .b = ratio == 2 } },
        });
        try e.kvStore(lat, L, ratio);
        const kp = try e.buf("L.ix_k", .{}, .bf16, &.{ n, ID });
        try e.upstream(try e.fmt("L{d}.attn.ix_wk", .{L}), lat, hd, ID, kp);
        const kpg = try e.paging(ratio, false);
        try e.triton("_index_k", .{ n, 1, 1 }, &.{
            .{ .name = "KP", .arg = kp },                                                    .{ .name = "k_stride", .arg = .{ .i = ID } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_knorm", .{L}, .f32, &.{ID}) }, .{ .name = "CS", .arg = try e.ropeTable() },
            .{ .name = "cs_stride", .arg = .{ .i = 64 } },                                   .{ .name = "IK", .arg = try e.indexKeys(L) },
            .{ .name = "POS", .arg = try e.pos() },                                          .{ .name = "EPS", .arg = .{ .f = dec(e.cfg.eps) } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },                                    .{ .name = "PT", .arg = kpg.pt },
            .{ .name = "PSH", .arg = .{ .i = kpg.psh } },                                    .{ .name = "KFP8", .arg = .{ .b = true } },
            .{ .name = "SL", .arg = .none },                                                 .{ .name = "PTS", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }
    if (indexer) try e.index(L, if (e.o.r1) qa else qn, iw, qnorm);
    const early: ?Received = if (indexer and e.o.pool != null and e.o.pool.?.split and e.o.pool.?.early and (n <= 16 or e.o.rows))
        try e.exchange(kv_src.?, e.cfg.compressRatio(kv_src.?), (try e.selection(L))[0])
    else
        null;
    e.onMain();
    // q group: wq_b
    const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
    try e.fusedNorm("q", &.{.{ .w = try e.fmt("L{d}.attn.wq_b", .{L}), .x = if (e.o.r1) qa else qn, .y = q, .k = e.cfg.q_lora, .n = @intCast(H * hd) }}, qnorm);
    try e.prefetchO(L);
    if (!e.o.r1) try e.rope(q, H, hd, hd - 64); // `r1` (ATTN_ROPE): the attention kernel rotates q as it loads it
    if (indexer) try e.swaStore(L, kv, kvn);
    e.join(); // the attention core reads both streams' values
    // attention over the SWA ring and (a compressed layer) the source's pool at the source's selection
    const o = try e.buf("L.o", .{}, .bf16, &.{ n, H, hd });
    const swa = try e.pool(L, 0);
    var cv = try e.empty(.u8);
    var cs = try e.empty(.u8);
    var tok = try e.empty(.i32);
    var cnt = try e.empty(.i32);
    var apg: Paging = .{ .pt = try e.empty(.i32), .psh = 0, .vs = 0, .ss = 0 };
    if (ratio > 0) {
        const src_ratio = e.cfg.compressRatio(kv_src.?);
        const sel = try e.selection(e.cfg.indexSource(L).?);
        cnt = sel[1];
        if (e.o.pool != null and e.o.pool.?.split) {
            // split: an index layer's selection brings its rows over the exchange; its reuse layers read them too
            const got = early orelse if (indexer) try e.exchange(kv_src.?, src_ratio, sel[0]) else try e.received(kv_src.?, src_ratio);
            cv = got.cv;
            cs = got.cs;
            tok = got.tok;
        } else {
            const p = try e.pool(kv_src.?, src_ratio);
            cv = p[0];
            cs = p[1];
            tok = sel[0];
            if (e.o.pool != null) {
                const pg = try e.paging(src_ratio, false);
                apg = .{ .pt = pg.pt, .psh = pg.psh, .vs = 0, .ss = 0 };
            }
        }
    }
    const rows_scr = n * 5 * H;
    try e.ext("tf_dsv41_attn_cuda_v1.attn", &.{
        try e.view(q.t.role, .bf16, &.{ n, H, hd }, &.{ H * hd, hd, 1 }, 0), cv,                                              cs,                                              tok,               cnt,                                                      swa[0],                                            swa[1],
        try e.buf("w.lo", .{}, .i32, &.{n}),                                 try e.empty(.i32),                               try e.pos(),                                     try e.empty(.i64), apg.pt,                                                   try e.weight("L{d}.attn.sink", .{L}, .f32, &.{H}), try e.ropeTable(),
        try e.buf("s.attn.po", .{}, .f32, &.{rows_scr * 512}),               try e.buf("s.attn.pm", .{}, .f32, &.{rows_scr}), try e.buf("s.attn.pl", .{}, .f32, &.{rows_scr}), o,                 try e.buf("s.attn.ticket", .{}, .i32, &.{e.ticketLen()}), .{ .i = 2 * @as(i64, e.cfg.window) },              .{ .i = apg.psh },
        .{ .i = 0 },                                                         .{ .i = 4 },
    });
    if (e.o.r1) {
        // ATTN_SPLIT 4 (a cluster of 4 a chunk up to ATTN_SPLIT_ROWS = 2 rows, 1 above) and ATTN_ROPE
        const last = &e.out.items[e.out.items.len - 1];
        const more = try e.a.alloc(calls.Named, last.args.len + 2);
        @memcpy(more[0..last.args.len], last.args);
        more[last.args.len] = .{ .arg = .{ .i = attnSplit(n) } };
        more[last.args.len + 1] = .{ .arg = .{ .i = 1 } };
        last.args = more;
    } else if (e.o.r1_sig) try e.appendLast(&.{ .{ .i = 0 }, .{ .i = 0 } });
    // o: wo_a's groups over the heads' columns, then wo_b (this rank's partial)
    const g: usize = e.cfg.o_groups / e.o.world;
    const gk: i64 = @divExact(H * hd, @as(i64, @intCast(g)));
    const oa = try e.buf("L.oa", .{}, .bf16, &.{ n, @as(i64, @intCast(g)) * e.cfg.o_lora });
    var po: [8]Proj = undefined;
    for (po[0..g], 0..) |*p, i| p.* = .{
        .w = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i }),
        .x = try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ H * hd, 1 }, @as(i64, @intCast(i)) * gk * 2),
        .y = oa,
        .col = i * e.cfg.o_lora,
        .k = @intCast(gk),
        .n = e.cfg.o_lora,
    };
    try e.fused("o", po[0..g]);
    try e.fused("wo_b", &.{.{ .w = try e.fmt("L{d}.attn.wo_b", .{L}), .x = oa, .y = try e.buf("L.part", .{}, .bf16, &.{ n, D }), .k = g * e.cfg.o_lora, .n = @intCast(D) }});
}
