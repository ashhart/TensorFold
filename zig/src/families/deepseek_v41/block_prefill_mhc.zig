//! A prefill segment's mHC sites: the coefficient sets, site / boundary pieces, `_finish_k`, the head.

const std = @import("std");
const calls = @import("block_calls.zig");
const moe = @import("block_moe.zig");
const prefill = @import("block_prefill.zig");
const E = prefill.E;
const Arg = calls.Arg;
const pf_rows = prefill.pf_rows;
const mhc_bm = prefill.mhc_bm;
const split_rows = prefill.split_rows;
const cdiv = prefill.cdiv;
const Seg = prefill.Seg;
const pieceAt = prefill.Seg.pieceAt;
const rowsOf = prefill.Seg.rowsOf;

// -- mHC -----------------------------------------------------------------------------------------------------

pub fn coefs(e: *E, set: u1) ![3]Arg {
    return .{
        try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
    };
}

/// DSpark's tap (forward._layers): the new streams' mean at a boundary entering a target layer.
fn tapOf(e: *E, L: u32, which: []const u8, mode: moe.Mode) !?Arg {
    if (!e.o.taps or mode != .boundary or !std.mem.eql(u8, which, "hc_attn")) return null;
    const t = e.cfg.dspark_targets.items();
    const j = std.mem.indexOfScalar(u32, t, L) orelse return null;
    const D: i64 = e.cfg.hidden;
    return try e.view(.{ .buf = "w.taps" }, .bf16, &.{ e.n, D }, &.{ @as(i64, @intCast(t.len)) * D, 1 }, @as(i64, @intCast(j)) * D * 2);
}

/// The exchange a post reads: the attention's partials at the FFN site, the MoE's anywhere else (layer-scoped).
fn gathered(e: *E, which: []const u8) !Arg {
    const D: i64 = e.cfg.hidden;
    if (std.mem.eql(u8, which, "hc_ffn")) return e.buf("L.recv.attn", .{}, .bf16, &.{ e.o.world, e.n, D });
    return e.buf("L.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D });
}

/// One mixing site of layer L (`which`: hc_attn / hc_ffn): site or boundary, then `_finish_k` (short: mhc_cuda).
pub fn site(s: *Seg, L: u32, which: []const u8, mode: moe.Mode) !void {
    const e = s.e;
    const pend = s.pend;
    s.pend = 0;
    // the decoder replay's boundaries write the streams in place (replay.finish's Streams has no `alt`)
    if (s.short) return moe.mhcInto(e, L, which, mode, s.part == .decoder);
    const n = e.n;
    if (pend > 1 and mode == .boundary and n >= pend * pf_rows and n > split_rows) return s.sitePieces(L, which, pend, true);
    // TF_DSV41_MHC_SITE_ROWS: the boundary in row pieces, each piece's inputs still in L2 for its `_finish_k`
    if (mode == .boundary and e.o.mhc_site_rows > 0 and n > e.o.mhc_site_rows) {
        const k = cdiv(n, e.o.mhc_site_rows);
        if (n >= k * pf_rows) return s.sitePieces(L, which, k, false);
    }
    const D: i64 = e.cfg.hidden;
    const post = mode == .boundary;
    const x = try e.streams();
    const prev = try coefs(e, e.coef);
    const next = try coefs(e, 1 - e.coef);
    const b0 = try e.empty(.bf16);
    const f0 = try e.empty(.f32);
    const fnw = try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D });
    const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
    const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
    const tap = try tapOf(e, L, which, mode);
    // the forward's second streams buffer (mhc.out_of_place): a boundary of at most 32 rows writes it
    const spare = post and n <= split_rows and s.part != .decoder;
    const xout = if (spare) try e.spare() else x;
    if (n >= pf_rows) {
        try e.ext("tf_dsv41_mhc_pf_v1.run", &.{
            x,                                        if (post) xout else b0,
            if (post) try gathered(e, which) else b0, if (post) prev[1] else f0,
            if (post) prev[2] else f0,                if (mode != .site1) prev[0] else f0,
            fnw,                                      part,
            c,                                        tap orelse b0,
            .{ .i = n },                              .{ .i = @backingInt(mode) },
        });
    } else try s.siteTriton(x, if (post) xout else null, if (post) try gathered(e, which) else null, prev, if (mode != .site1) prev[0] else null, fnw, part, c, tap, if (mode == .site1) 1 else 2, true);
    try s.finish(.{ L, which }, next, try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D}), part, c, 0);
    e.coef = 1 - e.coef;
    if (spare) e.cur = 1 - e.cur;
}

/// A boundary site over the last exchange's k row pieces; row-wise, so the same bits (`overlapped`: with marks).
pub fn sitePieces(s: *Seg, L: u32, which: []const u8, k: i64, overlapped: bool) !void {
    const e = s.e;
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const W: i64 = e.o.world;
    const x = try e.streams();
    const prev = try coefs(e, e.coef);
    const next = try coefs(e, 1 - e.coef);
    const b0 = try e.empty(.bf16);
    const fnw = try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D });
    const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
    const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
    const g = try gathered(e, which);
    const tap = try tapOf(e, L, which, .boundary);
    const nw = try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D});
    if (overlapped) e.br_join = false; // piece p waits for its own exchange piece; the last joins
    defer e.n = n;
    var p: i64 = 0;
    while (p < k) : (p += 1) {
        const r = pieceAt(n, k, p);
        const m = r[1] - r[0];
        e.n = m;
        if (overlapped) {
            if (p + 1 < k) e.br_wait = @intCast(p + 1) else e.br_join = true;
        }
        const xr = try rowsOf(e, x, .bf16, r[0], m, 4 * D);
        const nx: [3]Arg = .{ try rowsOf(e, next[0], .f32, r[0], m, 4), try rowsOf(e, next[1], .f32, r[0], m, 4), try rowsOf(e, next[2], .f32, r[0], m, 16) };
        const pr = try e.view(part.t.role, .f32, &.{ m, 4, 40, 32 }, &.{ 4 * 40 * 32, 40 * 32, 32, 1 }, r[0] * 4 * 40 * 32 * 4);
        const cr = try rowsOf(e, c, .bf16, r[0], m, D);
        try e.ext("tf_dsv41_mhc_pf_v1.run", &.{
            xr,                                                                          xr,
            try e.view(g.t.role, .bf16, &.{ W, m, D }, &.{ n * D, D, 1 }, r[0] * D * 2), try rowsOf(e, prev[1], .f32, r[0], m, 4),
            try rowsOf(e, prev[2], .f32, r[0], m, 16),                                   try rowsOf(e, prev[0], .f32, r[0], m, 4),
            fnw,                                                                         pr,
            cr,                                                                          if (tap) |t| try e.view(t.t.role, .bf16, &.{ m, D }, t.t.stride, t.t.offset + r[0] * t.t.stride[0] * 2) else b0,
            .{ .i = m },                                                                 .{ .i = @backingInt(moe.Mode.boundary) },
        });
        try s.finish(.{ L, which }, nx, nw, pr, cr, r[0]);
    }
    e.coef = 1 - e.coef;
}

/// mhc._launch's Triton `_site`: `xout` / `g` with the post, `pre` with collapse 2, `fnw` with the mix.
pub fn siteTriton(s: *Seg, x: Arg, xout: ?Arg, g: ?Arg, prev: [3]Arg, pre: ?Arg, fnw: ?Arg, part: Arg, c: Arg, tap: ?Arg, collapse: i64, mix: bool) !void {
    const e = s.e;
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const post = g != null;
    const bm: i64 = if (n >= pf_rows) mhc_bm else 16;
    // a program a stream for small mixing windows, never in place with a post
    const split = mix and n <= split_rows and !(post and std.mem.eql(u8, xout.?.t.role.buf, x.t.role.buf));
    try e.triton("_site", .{ cdiv(n, bm), 40, if (split) 4 else 1 }, &.{
        .{ .name = "X", .arg = x },                                                    .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
        .{ .name = "XOUT", .arg = xout orelse x },                                     .{ .name = "G", .arg = g orelse x },
        .{ .name = "g_rank", .arg = .{ .i = if (post) n * D else 0 } },                .{ .name = "POST", .arg = if (post) prev[1] else x },
        .{ .name = "COMB", .arg = if (post) prev[2] else x },                          .{ .name = "PRE", .arg = pre orelse x },
        .{ .name = "FN", .arg = fnw orelse x },                                        .{ .name = "PART", .arg = part },
        .{ .name = "C", .arg = c },                                                    .{ .name = "TAP", .arg = tap orelse x },
        .{ .name = "tap_stride", .arg = .{ .i = if (tap) |t| t.t.stride[0] else 0 } }, .{ .name = "R", .arg = .{ .i = n } },
        .{ .name = "D", .arg = .{ .i = D } },                                          .{ .name = "NB", .arg = .{ .i = 40 } },
        .{ .name = "BM", .arg = .{ .i = bm } },                                        .{ .name = "BK", .arg = .{ .i = 16 } },
        .{ .name = "WORLD", .arg = .{ .i = if (post) e.o.world else 1 } },             .{ .name = "POST_ON", .arg = .{ .b = post } },
        .{ .name = "COLLAPSE", .arg = .{ .i = collapse } },                            .{ .name = "MIX", .arg = .{ .b = mix } },
        .{ .name = "TAP_ON", .arg = .{ .b = tap != null } },                           .{ .name = "SPLIT", .arg = .{ .b = split } },
    });
}

/// The partials' sums, the next coefficients and the normed input into "w.out" rows from `r0`.
pub fn finish(s: *Seg, hc: ?struct { u32, []const u8 }, next: [3]Arg, nw: Arg, part: Arg, c: Arg, r0: i64) !void {
    const e = s.e;
    const D: i64 = e.cfg.hidden;
    const coef = hc != null;
    const base = if (hc) |h| try e.weight("L{d}.{s}.base", .{ h[0], h[1] }, .f32, &.{6 * @as(i64, e.cfg.hc_mult)}) else part;
    const scale = if (hc) |h| try e.weight("L{d}.{s}.scale", .{ h[0], h[1] }, .f32, &.{3}) else part;
    try e.triton("_finish_k", .{ e.n, 1, 1 }, &.{
        .{ .name = "PART", .arg = part },                                                                          .{ .name = "BASE", .arg = base },
        .{ .name = "SCALE", .arg = scale },                                                                        .{ .name = "PRE", .arg = if (coef) next[0] else part },
        .{ .name = "POST", .arg = if (coef) next[1] else part },                                                   .{ .name = "COMB", .arg = if (coef) next[2] else part },
        .{ .name = "C", .arg = c },                                                                                .{ .name = "NW", .arg = nw },
        .{ .name = "OUT", .arg = try rowsOf(e, try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), .bf16, r0, e.n, D) }, .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
        .{ .name = "hc_eps", .arg = .{ .f = E.dec(e.cfg.hc_eps) } },                                               .{ .name = "post_alpha", .arg = .{ .f = E.dec(e.cfg.hc_post_alpha) } },
        .{ .name = "D", .arg = .{ .i = D } },                                                                      .{ .name = "NB", .arg = .{ .i = 40 } },
        .{ .name = "ITERS", .arg = .{ .i = e.cfg.hc_sinkhorn_iters } },                                            .{ .name = "COEF", .arg = .{ .b = coef } },
        .{ .name = "BLOCK", .arg = .{ .i = @intCast(std.math.gcd(@as(u64, 1024), @as(u64, @intCast(D)))) } },      .{ .name = "UNROLL", .arg = .{ .b = true } },
        .{ .name = "NORM_FIRST", .arg = .{ .b = true } },
    });
}

/// mhc.post_only: the last exchange's post into `xout` (the streams in place, or the trace's copy).
pub fn postOnly(s: *Seg, xout: Arg) !void {
    const e = s.e;
    s.pend = 0; // reads the whole gathered buffer: it joined (sendPiece)
    if (s.short) return moe.postOnly(e, xout);
    const prev = try coefs(e, e.coef);
    const x = try e.streams();
    try s.siteTriton(x, xout, try gathered(e, ""), prev, null, null, x, x, null, 0, false);
}

/// mhc.final: the last post, collapse 2, the final norm; then the head's columns over every row.
pub fn final(s: *Seg) !void {
    const e = s.e;
    s.pend = 0;
    if (s.short) return moe.finish(e); // mhc_cuda's final, then the same head
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const x = try e.streams();
    const prev = try coefs(e, e.coef);
    const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
    const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
    try s.siteTriton(x, x, try gathered(e, ""), prev, prev[0], null, part, c, null, 2, false);
    try s.finish(null, prev, try e.weight("norm", .{}, .f32, &.{D}), part, c, 0);
    // the head: the decode path's linear (prefill_mm.fast off), upstream's 128 rows a launch
    const V: i64 = e.cfg.vocab / e.o.world;
    var a: i64 = 0;
    while (a < n) : (a += 128) {
        const m = @min(n - a, 128);
        e.n = m;
        defer e.n = n;
        try e.upstream("head", try e.view(.{ .buf = "w.out" }, .bf16, &.{ m, D }, &.{ D, 1 }, a * D * 2), D, V, try e.view(.{ .buf = "w.logits" }, .f32, &.{ m, V }, &.{ V, 1 }, a * V * 4));
    }
    try e.glue("kit_logits", &.{try e.buf("w.logits", .{}, .f32, &.{ n, V })});
}
