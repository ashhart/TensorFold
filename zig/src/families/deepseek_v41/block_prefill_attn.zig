//! A prefill segment's attention: rope, the KV stores, the compressor, CSA2's core over the selected blocks.

const calls = @import("block_calls.zig");
const prefill = @import("block_prefill.zig");
const E = prefill.E;
const Arg = calls.Arg;
const bmq = prefill.bmq;
const kt = prefill.kt;
const Out = prefill.Out;
const Seg = prefill.Seg;
const pieceAt = prefill.Seg.pieceAt;
const selection = prefill.Seg.selection;
const rowsOf = prefill.Seg.rowsOf;
const stashArgs = prefill.stashArgs;
const pf_heads = @import("prefill_heads.zig");

// -- attention -----------------------------------------------------------------------------------------------

pub fn pos(e: *E) !Arg {
    return e.buf("w.pos", .{}, .i32, &.{1});
}

fn rope(e: *E, x: Arg, heads: i64, dim: i64) !void {
    return ropeAt(e, x, heads, dim, "w.pos64");
}

/// `rope` reading each row's position from `positions` (a multi-segment run: every segment's, "w.mpos64").
fn ropeAt(e: *E, x: Arg, heads: i64, dim: i64, positions: []const u8) !void {
    const n = e.n;
    const xv = try e.view(x.t.role, .bf16, &.{ n, heads, dim }, &.{ heads * dim, dim, 1 }, 0);
    try e.triton("_rope", .{ n, @divExact(heads, 4), 1 }, &.{
        .{ .name = "X", .arg = xv },                                            .{ .name = "x_rs", .arg = .{ .i = heads * dim } },
        .{ .name = "x_hs", .arg = .{ .i = dim } },                              .{ .name = "OUT", .arg = xv },
        .{ .name = "o_rs", .arg = .{ .i = heads * dim } },                      .{ .name = "o_hs", .arg = .{ .i = dim } },
        .{ .name = "CS", .arg = try e.ropeTable() },                            .{ .name = "cs_rs", .arg = .{ .i = 64 } },
        .{ .name = "POS", .arg = try e.t(.{ .buf = positions }, .i64, &.{n}) }, .{ .name = "H", .arg = .{ .i = heads } },
        .{ .name = "NOPE", .arg = .{ .i = dim - 64 } },                         .{ .name = "NB", .arg = .{ .i = 64 } },
        .{ .name = "HALF", .arg = .{ .i = 32 } },                               .{ .name = "BH", .arg = .{ .i = 4 } },
        .{ .name = "INV", .arg = .{ .b = false } },                             .{ .name = "COPY", .arg = .{ .b = false } },
        .{ .name = "PDL", .arg = .{ .b = false } },
    });
}

/// The SWA ring this segment's layer L stores into (values, scales): the staging ring or the slot's.
pub fn swa(s: *Seg, L: u32) ![2]Arg {
    const e = s.e;
    if (s.staged) return .{ try e.buf("s.stage.v", .{}, .u8, &.{ s.ring, 576 }), try e.buf("s.stage.s", .{}, .u8, &.{ s.ring, 8 }) };
    return slotRing(e, L);
}

fn slotRing(e: *E, L: u32) ![2]Arg {
    const r = 2 * @as(i64, e.cfg.window);
    return .{ try e.buf("s.L{d}.swa.v", .{L}, .u8, &.{ r, 576 }), try e.buf("s.L{d}.swa.s", .{L}, .u8, &.{ r, 8 }) };
}

/// A KV source's compressed rows (values, scales): the slot's contiguous ones, or the pool's (block.zig).
fn pool(e: *E, L: u32) ![2]Arg {
    return e.pool(L, e.cfg.compressRatio(L));
}

pub fn indexKeys(e: *E, L: u32) !Arg {
    return e.indexKeys(L);
}

/// Rows into the SWA ring (ratio 0) or a compressed family (pool mode: through the slot's table or split table).
pub fn kvStore(s: *Seg, lat: Arg, rows: [2]Arg, ratio: i64) !void {
    const e = s.e;
    const pg = try e.paging(ratio, true);
    try e.triton("_kv_store", .{ e.n, 1, 1 }, &.{
        .{ .name = "LAT", .arg = lat },                                      .{ .name = "l_stride", .arg = .{ .i = e.cfg.head_dim } },
        .{ .name = "CS", .arg = try e.ropeTable() },                         .{ .name = "cs_stride", .arg = .{ .i = 64 } },
        .{ .name = "V", .arg = rows[0] },                                    .{ .name = "S", .arg = rows[1] },
        .{ .name = "v_stride", .arg = .{ .i = pg.vs } },                     .{ .name = "s_stride", .arg = .{ .i = pg.ss } },
        .{ .name = "POS", .arg = try pos(e) },                               .{ .name = "RATIO", .arg = .{ .i = ratio } },
        .{ .name = "RING", .arg = .{ .i = if (ratio == 0) s.ring else 1 } }, .{ .name = "PT", .arg = pg.pt },
        .{ .name = "PSH", .arg = .{ .i = pg.psh } },                         .{ .name = "SL", .arg = .none },
        .{ .name = "PTS", .arg = .{ .i = 0 } },                              .{ .name = "ROWS", .arg = .{ .b = false } },
        .{ .name = "PDL", .arg = .{ .b = false } },
    });
}

/// A KV source's compressor (blocks._compress): pooled rows normed into its pool, the index keys from them.
pub fn compress(s: *Seg, L: u32, projection: Arg) !void {
    const e = s.e;
    const n = e.n;
    const hd: i64 = e.cfg.head_dim;
    const ID: i64 = e.cfg.index_dim;
    const ratio: i64 = e.cfg.compressRatio(L);
    // ratio 2: [the slot's carry (position start - 1) | the rows]
    const buf = if (ratio == 2) try e.buf("L.cbuf", .{}, .f32, &.{ n + 1, 2 * hd }) else projection;
    if (ratio == 2) try e.glue("proj_carry", &.{ try e.buf("s.L{d}.carry", .{L}, .f32, &.{ 1, 2 * hd }), projection, buf });
    const lat = try e.buf("L.lat", .{}, .bf16, &.{ n, hd });
    try e.glue("zeros", &.{lat});
    try e.triton("_pool_norm", .{ n, 1, 1 }, &.{
        .{ .name = "BUF", .arg = buf },                                                   .{ .name = "b_stride", .arg = .{ .i = ratio * hd } },
        .{ .name = "W", .arg = try e.weight("L{d}.attn.comp_norm", .{L}, .f32, &.{hd}) }, .{ .name = "LAT", .arg = lat },
        .{ .name = "POS", .arg = try pos(e) },                                            .{ .name = "n", .arg = .{ .i = n } },
        .{ .name = "EPS", .arg = .{ .f = E.dec(e.cfg.eps) } },                            .{ .name = "RATIO", .arg = .{ .i = ratio } },
        .{ .name = "SL", .arg = .none },                                                  .{ .name = "PREV", .arg = .none },
        .{ .name = "OFF", .arg = .{ .i = 0 } },                                           .{ .name = "ROWS", .arg = .{ .b = false } },
        .{ .name = "PDL", .arg = .{ .b = false } },                                       .{ .name = "CARRY", .arg = .none },
        .{ .name = "c_stride", .arg = .{ .i = 0 } },                                      .{ .name = "SPLIT", .arg = .{ .b = false } },
    });
    try s.kvStore(lat, try pool(e, L), ratio);
    const kp = try e.buf("L.ix_k", .{}, .bf16, &.{ n, ID });
    try s.proj(try e.fmt("L{d}.attn.ix_wk", .{L}), .strips, lat, hd, ID, kp);
    const kpg = try e.paging(ratio, false);
    try e.triton("_index_k", .{ n, 1, 1 }, &.{
        .{ .name = "KP", .arg = kp },                                                    .{ .name = "k_stride", .arg = .{ .i = ID } },
        .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_knorm", .{L}, .f32, &.{ID}) }, .{ .name = "CS", .arg = try e.ropeTable() },
        .{ .name = "cs_stride", .arg = .{ .i = 64 } },                                   .{ .name = "IK", .arg = try indexKeys(e, L) },
        .{ .name = "POS", .arg = try pos(e) },                                           .{ .name = "EPS", .arg = .{ .f = E.dec(e.cfg.eps) } },
        .{ .name = "RATIO", .arg = .{ .i = ratio } },                                    .{ .name = "PT", .arg = kpg.pt },
        .{ .name = "PSH", .arg = .{ .i = kpg.psh } },                                    .{ .name = "KFP8", .arg = .{ .b = true } },
        .{ .name = "SL", .arg = .none },                                                 .{ .name = "PTS", .arg = .{ .i = 0 } },
        .{ .name = "ROWS", .arg = .{ .b = false } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
    });
}

/// Win.sub's first position (int32 [1]) of the row block starting at row `a`.
pub fn blockPos(s: *Seg, j: usize, a: i64) !Arg {
    const e = s.e;
    const p = try e.buf("L.ix.pos{d}", .{j}, .i32, &.{1});
    try e.glue("block_pos", &.{ p, .{ .i = e.start + a } });
    return p;
}

/// A kv source's compressor projection of "w.out" as fp32 rows (ratio 2: [kv | gate], kept for the carry).
pub fn compProj(s: *Seg, L: u32) !Arg {
    const e = s.e;
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const hd: i64 = e.cfg.head_dim;
    const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
    const cw = try e.buf("L.comp", .{}, .bf16, &.{ n, hd });
    try s.proj(try e.fmt("L{d}.attn.comp_wkv", .{L}), .lanes, x, D, hd, cw);
    if (e.cfg.compressRatio(L) == 2) {
        const cg = try e.buf("L.comp_gate", .{}, .bf16, &.{ n, hd });
        try s.proj(try e.fmt("L{d}.attn.comp_wgate", .{L}), .lanes, x, D, hd, cg);
        const p = try e.buf("w.L{d}.proj", .{L}, .f32, &.{ n, 2 * hd });
        try e.glue("proj", &.{ cw, cg, p });
        return p;
    }
    const p = try e.buf("L.proj", .{}, .f32, &.{ n, hd });
    try e.glue("proj", &.{ cw, p });
    return p;
}

/// CED's encoder end at the decoder's first layer: its compressor alone, then the stash of the entering rows.
pub fn encoderEnd(s: *Seg, L: u32) !void {
    const e = s.e;
    if (e.cfg.mode(L) != .full) return error.Unsupported;
    const p = try s.compProj(L);
    const segs = s.multi orelse {
        try s.compress(L, p);
        return e.glue("ced_stash", &try stashArgs(e));
    };
    // blocks.compress_rows: the projection over every row, then each segment's compressor and stash (replay.encode)
    const total = e.n;
    for (segs, 0..) |sp, j| {
        try s.enter(sp, j);
        try s.compress(L, try rowsOf(e, p, .f32, e.row0, e.n, p.t.stride[0]));
        try e.glue("ced_stash", &try stashArgs(e));
    }
    try s.leave(total);
}

pub fn attention(s: *Seg, L: u32) !void {
    const e = s.e;
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const md = e.cfg.mode(L);
    const H: i64 = e.cfg.heads / e.o.world;
    const hd: i64 = e.cfg.head_dim;
    const IH: i64 = e.cfg.index_heads;
    const ID: i64 = e.cfg.index_dim;
    const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
    const indexer = md == .full or md == .reindex;
    // the decoder replay skips the kv source's compressor (blocks.Context.comp_done: the encoder pass stored it)
    const own_kv = md == .full and s.part != .decoder;
    // the x projections (wq_a, wkv, a KV source's compressor), each its own matmul
    const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, e.cfg.q_lora });
    const kva = try e.buf("L.kva", .{}, .bf16, &.{ n, hd });
    try s.proj(try e.fmt("L{d}.attn.wq_a", .{L}), .lanes, x, D, e.cfg.q_lora, qa);
    try s.proj(try e.fmt("L{d}.attn.wkv", .{L}), .lanes, x, D, hd, kva);
    const cproj: Arg = if (own_kv) try s.compProj(L) else undefined;
    const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, e.cfg.q_lora });
    const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
    try e.triton("_rms2", .{ n, 2, 1 }, &.{
        .{ .name = "XA", .arg = qa },                                                             .{ .name = "xa_rs", .arg = .{ .i = e.cfg.q_lora } },
        .{ .name = "WA", .arg = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}) }, .{ .name = "OA", .arg = qn },
        .{ .name = "oa_rs", .arg = .{ .i = e.cfg.q_lora } },                                      .{ .name = "KA", .arg = .{ .i = e.cfg.q_lora } },
        .{ .name = "inv_ka", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(e.cfg.q_lora)) } },      .{ .name = "XB", .arg = kva },
        .{ .name = "xb_rs", .arg = .{ .i = hd } },                                                .{ .name = "WB", .arg = try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}) },
        .{ .name = "OB", .arg = kv },                                                             .{ .name = "ob_rs", .arg = .{ .i = hd } },
        .{ .name = "KB", .arg = .{ .i = hd } },                                                   .{ .name = "inv_kb", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(hd)) } },
        .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },                                    .{ .name = "BK", .arg = .{ .i = 512 } },
        .{ .name = "NARROW", .arg = .{ .b = true } },                                             .{ .name = "PDL", .arg = .{ .b = false } },
    });
    const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
    try s.proj(try e.fmt("L{d}.attn.wq_b", .{L}), .lanes, qn, e.cfg.q_lora, H * hd, q);
    const qi = try e.buf("L.qi", .{}, .bf16, &.{ n, IH * ID });
    if (indexer) try s.proj(try e.fmt("L{d}.attn.ix_wq_b", .{L}), .lanes, qn, e.cfg.q_lora, IH * ID, qi);
    if (s.short) try e.prefetchO(L);
    const positions = if (s.multi != null) "w.mpos64" else "w.pos64";
    try ropeAt(e, q, H, hd, positions);
    if (indexer) {
        try ropeAt(e, qi, IH, ID, positions);
        // the head weights: `_plain`'s fp64 sums, cast to bf16 then fp32 by torch
        const w64 = try e.buf("L.ix_w64", .{}, .f64, &.{ n, IH });
        try e.triton("_plain", .{ n, IH, 1 }, &.{
            .{ .name = "X", .arg = x },                                                        .{ .name = "x_rs", .arg = .{ .i = D } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_wp", .{L}, .f32, &.{ IH, D }) }, .{ .name = "w_rs", .arg = .{ .i = D } },
            .{ .name = "OUT", .arg = w64 },                                                    .{ .name = "K", .arg = .{ .i = D } },
            .{ .name = "N", .arg = .{ .i = IH } },                                             .{ .name = "BK", .arg = .{ .i = 256 } },
            .{ .name = "PDL", .arg = .{ .b = false } },                                        .{ .name = "BF16", .arg = .{ .b = false } },
        });
        try e.glue("ix_w", &.{ w64, try e.buf("L.ix_w", .{}, .f32, &.{ n, IH }) });
    }
    // the CSA2 steps a segment at a time, into the group-major core output
    const groups: i64 = e.cfg.o_groups / e.o.world;
    if ((H != 16 and @mod(H, bmq) != 0) or @mod(H, groups) != 0) return error.Unsupported;
    const core_q = if (H == 16) try pf_heads.query(e, q, hd) else q;
    const hg = @divExact(H, groups);
    const o = try e.buf("L.o", .{}, .bf16, &.{ groups, n, hg * hd });
    if (s.multi) |segs| {
        for (segs, 0..) |sp, j| {
            try s.enter(sp, j);
            try s.core(L, core_q, kv, cproj, own_kv, indexer, o, n);
        }
        try s.leave(n);
    } else try s.core(L, core_q, kv, cproj, own_kv, indexer, o, n);
    // _out_pf: each wo_a group's GEMM over its contiguous slice into its columns of z, then wo_b (bf16 partial)
    const ol: i64 = e.cfg.o_lora;
    const gk = hg * hd;
    const z: Out = .{ .role = .{ .buf = "L.z" }, .dt = .bf16, .stride = groups * ol };
    const k = s.pieces();
    if (k == 1) {
        var g: i64 = 0;
        while (g < groups) : (g += 1) {
            var zg = z;
            zg.offset = g * ol * 2;
            try s.matmul(try e.fmt("L{d}.attn.wo_a.{d}", .{ L, g }), .lanes, try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ gk, 1 }, g * n * gk * 2), gk, ol, zg);
        }
        return s.proj(try e.fmt("L{d}.attn.wo_b", .{L}), .lanes, try e.buf("L.z", .{}, .bf16, &.{ n, groups * ol }), groups * ol, D, try e.buf("L.part", .{}, .bf16, &.{ n, D }));
    }
    // TF_DSV41_PF_OVERLAP: wo_a / wo_b a row piece at a time (row-independent), each partial sent behind the next
    const part = try e.buf("L.part", .{}, .bf16, &.{ n, D });
    const zall = try e.buf("L.z", .{}, .bf16, &.{ n, groups * ol });
    const gath = try s.recv("attn");
    defer e.n = n;
    var p: i64 = 0;
    while (p < k) : (p += 1) {
        const r = pieceAt(n, k, p);
        const m = r[1] - r[0];
        e.n = m;
        var g: i64 = 0;
        while (g < groups) : (g += 1) {
            var zg = z;
            zg.offset = g * ol * 2 + r[0] * groups * ol * 2;
            try s.matmul(try e.fmt("L{d}.attn.wo_a.{d}", .{ L, g }), .lanes, try e.view(o.t.role, .bf16, &.{ m, gk }, &.{ gk, 1 }, (g * n + r[0]) * gk * 2), gk, ol, zg);
        }
        try s.matmul(try e.fmt("L{d}.attn.wo_b", .{L}), .lanes, try rowsOf(e, zall, .bf16, r[0], m, groups * ol), groups * ol, D, .{ .role = part.t.role, .dt = .bf16, .stride = D, .offset = r[0] * D * 2 });
        e.n = n;
        try s.sendPiece(part, gath, r[0], m, p, k);
    }
    s.sent_attn = true;
}

/// The CSA2 steps of the current segment into rows [row0, row0 + n) of the run's group-major output `o`.
pub fn core(s: *Seg, L: u32, q: Arg, kv: Arg, cproj: Arg, own_kv: bool, indexer: bool, o: Arg, total: i64) !void {
    const e = s.e;
    const n = e.n;
    const ratio: i64 = e.cfg.compressRatio(L);
    const real_heads: i64 = e.cfg.heads / e.o.world;
    const H: i64 = if (real_heads == 16) 32 else real_heads;
    const hd: i64 = e.cfg.head_dim;
    const groups: i64 = e.cfg.o_groups / e.o.world;
    const hg = @divExact(real_heads, groups);
    // blocks._segment: a segment the slot's ring cannot hold runs on the staging ring, the window before it in
    const rows = try s.swa(L);
    const slot = try slotRing(e, L);
    const W: i64 = e.cfg.window;
    const ring0 = 2 * W;
    if (s.staged) try e.glue("stage_in", &.{ rows[0], rows[1], .{ .i = s.ring }, slot[0], slot[1], .{ .i = ring0 }, .{ .i = @max(0, e.start - W + 1) }, .{ .i = e.start } });
    try s.kvStore(try rowsOf(e, kv, .bf16, e.row0, n, hd), rows, 0);
    if (own_kv) try s.compress(L, try rowsOf(e, cproj, .f32, e.row0, n, cproj.t.stride[0]));
    if (indexer) try s.select(L);
    // the fused prefill core over the SWA ring and (a compressed layer) the source's pool at the latest selection
    const lo = try e.buf("w.lo", .{}, .i32, &.{n});
    var cv = rows[0];
    var cs = rows[1];
    var tok = lo;
    var cnt = lo;
    var t_stride: i64 = 0;
    var fpg: E.Paging = .{ .pt = .none, .psh = 0, .vs = 0, .ss = 0 };
    if (ratio > 0) {
        const src = e.cfg.kvSource(L).?;
        const sel = try selection(e, e.cfg.indexSource(L).?);
        cnt = sel[1];
        t_stride = e.cfg.index_topk;
        if (e.o.pool != null and e.o.pool.?.split) {
            // split: an index layer's selection brings its rows over the exchange; its reuse layers read them too
            const got = if (indexer) try e.exchange(src, e.cfg.compressRatio(src), sel[0]) else try e.received(src, e.cfg.compressRatio(src));
            cv = got.cv;
            cs = got.cs;
            tok = got.tok;
        } else {
            const p = try pool(e, src);
            cv = p[0];
            cs = p[1];
            tok = sel[0];
            fpg = try e.paging(e.cfg.compressRatio(src), false);
        }
    }
    const real_sink = try e.weight("L{d}.attn.sink", .{L}, .f32, &.{real_heads});
    const sink = if (real_heads == 16) try pf_heads.sink(e, real_sink) else real_sink;
    const output = if (real_heads == 16) try e.buf("L.pf.o32", .{}, .bf16, &.{ 2 * groups, total, hg * hd }) else o;
    const out_view = try e.view(output.t.role, .bf16, &.{ @divExact(H, hg), n, hg * hd }, &.{ total * hg * hd, hg * hd, 1 }, e.row0 * hg * hd * 2);
    try e.triton("_fused", .{ n, @divExact(H, bmq), 1 }, &.{
        .{ .name = "Q", .arg = try e.view(q.t.role, .bf16, &.{ n, H, hd }, &.{ H * hd, hd, 1 }, e.row0 * H * hd * 2) },
        .{ .name = "CV", .arg = cv },
        .{ .name = "CSC", .arg = cs },
        .{ .name = "cvs", .arg = .{ .i = cv.t.stride[0] } },
        .{ .name = "css", .arg = .{ .i = cs.t.stride[0] } },
        .{ .name = "TOK", .arg = tok },
        .{ .name = "t_stride", .arg = .{ .i = t_stride } },
        .{ .name = "CNT", .arg = cnt },
        .{ .name = "SV", .arg = rows[0] },
        .{ .name = "SSC", .arg = rows[1] },
        .{ .name = "svs", .arg = .{ .i = 576 } },
        .{ .name = "sss", .arg = .{ .i = 8 } },
        .{ .name = "LO", .arg = lo },
        .{ .name = "POS", .arg = try pos(e) },
        .{ .name = "SINK", .arg = sink },
        .{ .name = "CS", .arg = try e.ropeTable() },
        .{ .name = "cs_stride", .arg = .{ .i = 64 } },
        .{ .name = "OUT", .arg = out_view },
        .{ .name = "H", .arg = .{ .i = H } },
        .{ .name = "KT", .arg = .{ .i = kt } },
        .{ .name = "BMQ", .arg = .{ .i = bmq } },
        // D ** -0.5 (libm pow, correctly rounded): sqrt of the exact 1 / 512 rounds the same way
        .{ .name = "SCALE", .arg = .{ .f = @sqrt(1.0 / @as(f64, @floatFromInt(hd))) } },
        .{ .name = "RING", .arg = .{ .i = s.ring } },
        .{ .name = "WINDOW", .arg = .{ .i = W } },
        .{ .name = "COMP", .arg = .{ .b = ratio > 0 } },
        .{ .name = "PT", .arg = fpg.pt },
        .{ .name = "PSH", .arg = .{ .i = fpg.psh } },
        .{ .name = "HG", .arg = .{ .i = hg } },
        .{ .name = "OR", .arg = .{ .i = total } },
    });
    if (real_heads == 16) try pf_heads.extract(e, out_view, try e.view(o.t.role, .bf16, &.{ groups, n, hg * hd }, &.{ total * hg * hd, hg * hd, 1 }, e.row0 * hg * hd * 2));
    if (s.staged) try e.glue("stage_out", &.{ slot[0], slot[1], .{ .i = ring0 }, rows[0], rows[1], .{ .i = s.ring }, .{ .i = @max(0, e.end() - W) }, .{ .i = e.end() } });
}
