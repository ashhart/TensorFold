//! A prefill segment on the fast-prefill path, emitted as calls (block_calls.zig) by the Emitter in reference order.

// Short segments (<= 16 rows) take the decode window's row paths; past TF_DSV41_INDEX_STREAM_MIN keys, stream_topk.

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const config = @import("config.zig");
const moe = @import("block_moe.zig");
const longpf = @import("longpf.zig");
const gm2pf = @import("gm2pf.zig");
const ced = @import("ced.zig");
const Config = config.Config;
pub const E = block.Emitter;
const Arg = calls.Arg;
const Dt = calls.Dt;
const dense = dk.dense;
const PfDense = @import("knob_types.zig").PfDense;
const block_prefill_mhc = @import("block_prefill_mhc.zig");
const block_prefill_attn = @import("block_prefill_attn.zig");
const block_prefill_select = @import("block_prefill_select.zig");
const block_prefill_moe = @import("block_prefill_moe.zig");
const block_prefill_emit = @import("block_prefill_emit.zig");

// reference knobs and constants
const ws_limit: i64 = 64 << 20; // TF_DSV41_PREFILL_WS_MIB: a workspace W_q's bytes at most (column blocks)
const had_scale: f64 = 0.08838834764831845; // exl3.prefill.HAD_SCALE (the literal: 1 / sqrt(128) rounds the other way)
pub const pf_rows: i64 = 256; // TF_DSV41_MHC_PF_ROWS / TF_DSV41_MHC_ROWS: mhc_pf and the wide `_site` tile from here
pub const mhc_bm: i64 = 64; // TF_DSV41_MHC_BM
pub const split_rows: i64 = 32; // TF_DSV41_MHC_SPLIT_ROWS: out-of-place boundaries, a program a stream
pub const stream_min = @import("forward_types.zig").stream_min;
pub const bmq: i64 = 32; // TF_DSV41_PREFILL_ATTN_BMQ
pub const kt: i64 = 32; // csa2.attn.KT
pub const router_chunk: i64 = 2048; // router_gemv.CHUNK: rows a route launch
pub const gm_bm: i64 = 64; // x3gm: members a pass of the gate/up (cfg 1) and down (cfg 0) tiles below 3,072 rows
pub const gm_widths: u32 = 7; // x3gm.MAX_WIDTHS: the ticket's down half starts there
const index_budget: i64 = 256 << 20; // TF_DSV41_INDEX_BUDGET_MIB's default: block.Options.index_budget is the run's
pub const dtopk_rows: i64 = 64; // dtopk.MAX_ROWS: windows up to this many rows select with attn_cuda's top-k (backend._dt)
const overlap_min: i64 = 256; // TF_DSV41_PF_OVERLAP: min rows a piece (smaller GEMMs cost more than they hide)

pub fn size(dt: Dt) i64 {
    return switch (dt) {
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
        .i8, .u8, .bool => 1,
    };
}

pub fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// How a matrix's words are kept: dense3's lanes, upstream's strips, or the checkpoint's tiles (the shared expert).
const Form = enum { lanes, strips, stored };

/// Where a matmul's [m, N] result goes: `role`'s rows of `stride` elements from byte `offset`.
pub const Out = struct { role: calls.Role, dt: Dt, stride: i64, offset: i64 = 0 };

/// One prefill segment: the emitter, the SWA ring its rows go through, and what its layers left (the candidates).
pub const Seg = struct {
    e: *E,
    /// the SWA ring the layers store into and attend over: the slot's (2 x window rows) or the staging ring
    staged: bool,
    ring: i64,
    /// the candidate source ran in this segment (reindex layers read its candidates)
    cand: bool = false,
    /// at most 16 rows: the short segment's kernel mix
    short: bool = false,
    /// CED replay's passes (`emitReplay`, ced.zig); `.whole`: a full segment
    part: Part = .whole,
    multi: ?[]const Span = null, // a multi-segment run: core and compressor a segment at a time (glue "pf_seg")
    sent_attn: bool = false, // TF_DSV41_PF_OVERLAP: the partial already went out in pieces; no whole exchange
    sent_moe: bool = false,
    pend: i64 = 0, // TF_DSV41_PF_OVERLAP_SITE: the last exchange's k side pieces (marks 1..k); 0: none pending

    // -- TF_DSV41_PF_OVERLAP: the exchanges in row pieces, overlapped with their producers' next piece -------------

    /// Row pieces of this segment's exchanges: Options.pf_overlap when it applies, else 1.
    pub fn pieces(s: *const Seg) i64 {
        const e = s.e;
        const k = e.o.pf_overlap;
        if (k < 2 or s.short or !e.o.branches or e.o.pf_tbo or e.o.world != 2 or e.n < k * overlap_min) return 1;
        return k;
    }

    /// Piece `p` of `k` over `n` rows: [r0, r1) on the 16-row grid, the last to `n`.
    pub fn pieceAt(n: i64, k: i64, p: i64) [2]i64 {
        const cut = struct {
            fn at(nn: i64, kk: i64, q: i64) i64 {
                return if (q >= kk) nn else @divFloor(@divFloor(nn * q, kk), 16) * 16;
            }
        };
        return .{ cut.at(n, k, p), cut.at(n, k, p + 1) };
    }

    /// Exchange of rows [r0, r0 + rows) of `part` into `into` [world, n, D]: side-stream pieces, the last joins main.
    pub fn sendPiece(s: *Seg, part: Arg, into: Arg, r0: i64, rows: i64, p: i64, k: i64) !void {
        const e = s.e;
        const send = try rowsOf(e, part, .bf16, r0, rows, e.cfg.hidden);
        const last = p + 1 == k;
        if (last and !e.o.pf_overlap_site) e.br_join = true else {
            e.br_fork = true;
            e.br_side = true;
            if (e.o.pf_overlap_site) e.br_mark = @intCast(p + 1);
        }
        defer e.br_side = false;
        // the piece's all-gather lands in "L.xrows" [world, rows, D], then each rank's rows are copied into `into`
        const tmp = try e.buf("L.xrows", .{}, .bf16, &.{ e.o.world, rows, e.cfg.hidden });
        try e.glue("exchange_rows", &.{ send, into, .{ .i = r0 }, tmp });
        if (last and e.o.pf_overlap_site) {
            s.pend = k;
            e.br_join = true;
        }
    }

    // -- dense EXL3 projections ----------------------------------------------------------------------------------

    /// y = x [m, K] @ W [K, N] (`w`'s prefix): per column block, upstream's rot_in, the W_q unpack, `_gemm`.
    pub fn matmul(s: *Seg, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Out) !void {
        const e = s.e;
        const m = e.n;
        const k2: i64 = try e.w.k2(w);
        const tw = 4 * k2;
        const nb = @divExact(n, 128);
        if (e.o.pfd) |p| if (PfDense.eligible(k, n, @intCast(k2), form == .lanes)) return s.fused(p, w, form, x, k, n, y);
        // prefill_mm.column_blocks: whole 128-column blocks whose W_q fits the workspace (strips / lanes only)
        const per: i64 = if (k * n * 2 <= ws_limit or form == .stored) nb else @max(1, @divFloor(ws_limit, k * 2 * 128));
        const st = dense.strides(@intCast(k), @intCast(k2));
        var b0: i64 = 0;
        while (b0 < nb) : (b0 += per) {
            const b1 = @min(nb, b0 + per);
            const c0 = 128 * b0;
            const nc = 128 * (b1 - b0);
            const xh = try e.buf("s.pf.xh", .{}, .f16, &.{ m, k });
            try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
            const wq = try e.buf("s.pf.w", .{}, .f16, &.{ k, nc });
            const blk_words = @divExact(k, 16) * 8 * tw; // words of a 128-column block
            switch (form) {
                .lanes => try e.ext("tf_dsv41_dense3_v1.unpack", &.{
                    try e.view(.{ .buf = try e.fmt("s.{s}.lanes", .{w}) }, .i32, &.{ b1 - b0, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, b0 * blk_words * 4),
                    wq,
                    .{ .i = st[0] },
                    .{ .i = st[1] },
                    .{ .i = k2 },
                }),
                .strips => try e.ext("tensorfold_exl3_linear_v4.unpack", &.{
                    // one block of 128 columns: the stored trellis is the strips (block.zig's upstream)
                    if (n == 128)
                        try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, tw }, &.{ 8 * tw, 8 * tw, tw, 1 }, 0)
                    else
                        try e.view(.{ .buf = try e.fmt("s.{s}.T", .{w}) }, .i32, &.{ b1 - b0, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, b0 * blk_words * 4),
                    wq,
                    .{ .i = st[0] },
                    .{ .i = st[1] },
                    .{ .i = k2 },
                    .{ .i = 2 },
                }),
                // Exl3Linear.strides of the stored layout: (words between k tiles, between 16-column tiles)
                .stored => try e.ext("tensorfold_exl3_linear_v4.unpack", &.{
                    try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ @divExact(k, 16), @divExact(n, 16), tw }, &.{ @divExact(n, 16) * tw, tw, 1 }, 0),
                    wq,
                    .{ .i = @divExact(n, 16) * tw },
                    .{ .i = 8 * tw },
                    .{ .i = k2 },
                    .{ .i = 2 },
                }),
            }
            const svh = try e.view(.{ .weight = try e.fmt("{s}.svh", .{w}) }, .f16, &.{nc}, &.{1}, c0 * 2);
            const out = try e.view(y.role, y.dt, &.{ m, nc }, &.{ y.stride, 1 }, y.offset + c0 * size(y.dt));
            // exl3.prefill.tiles: BM 128, BK 32, group 8 (the shape's alone)
            try e.triton("_gemm", .{ cdiv(m, 128) * @divExact(nc, 128), 1, 1 }, &.{
                .{ .name = "X", .arg = xh },                                                .{ .name = "W", .arg = wq },
                .{ .name = "H", .arg = try e.buf("s.pf.had", .{}, .bf16, &.{ 128, 128 }) }, .{ .name = "SVH", .arg = svh },
                .{ .name = "BIAS", .arg = svh },                                            .{ .name = "OUT", .arg = out },
                .{ .name = "M", .arg = .{ .i = m } },                                       .{ .name = "o_stride", .arg = .{ .i = y.stride } },
                .{ .name = "K", .arg = .{ .i = k } },                                       .{ .name = "N", .arg = .{ .i = nc } },
                .{ .name = "BM", .arg = .{ .i = 128 } },                                    .{ .name = "BK", .arg = .{ .i = 32 } },
                .{ .name = "GROUP", .arg = .{ .i = 8 } },                                   .{ .name = "HAS_BIAS", .arg = .{ .b = false } },
                .{ .name = "SCALE", .arg = .{ .f = had_scale } },
            });
        }
    }

    /// TF_DSV41_PF_DENSE=fused: rot_in into the workspace, then one fused GEMM over the whole layer (no column blocks).
    fn fused(s: *Seg, p: *const PfDense, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Out) !void {
        const e = s.e;
        const m = e.n;
        const k2: u32 = try e.w.k2(w);
        const tw: i64 = 4 * @as(i64, k2);
        const nb = @divExact(n, 128);
        const blk_words = @divExact(k, 16) * 8 * tw;
        const xh = try e.buf("s.pf.xh", .{}, .f16, &.{ m, k });
        try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
        const st = dense.strides(@intCast(k), k2);
        const words: Arg, const sk: i64, const snb: i64 = switch (form) {
            .lanes => .{ try e.view(.{ .buf = try e.fmt("s.{s}.lanes", .{w}) }, .i32, &.{ nb, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, 0), st[0], st[1] },
            .strips => .{ if (n == 128)
                try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, tw }, &.{ 8 * tw, 8 * tw, tw, 1 }, 0)
            else
                try e.view(.{ .buf = try e.fmt("s.{s}.T", .{w}) }, .i32, &.{ nb, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, 0), st[0], st[1] },
            .stored => .{ try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ @divExact(k, 16), @divExact(n, 16), tw }, &.{ @divExact(n, 16) * tw, tw, 1 }, 0), @divExact(n, 16) * tw, 8 * tw },
        };
        const c = p.pick(@intCast(k), @intCast(n), k2, @intCast(m));
        try e.ext("tf_dsv41_pfdense_v1.gemm", &.{
            xh,                                                                words,
            .{ .i = sk },                                                      .{ .i = snb },
            try e.weight("{s}.svh", .{w}, .f16, &.{n}),                        .none,
            try e.view(y.role, y.dt, &.{ m, n }, &.{ y.stride, 1 }, y.offset), .{ .i = k2 },
            .{ .b = form == .lanes },                                          .{ .i = c[0] },
            .{ .i = c[1] },
        });
    }

    /// A projection into a whole contiguous buffer.
    pub fn proj(s: *Seg, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Arg) !void {
        return s.matmul(w, form, x, k, n, .{ .role = y.t.role, .dt = y.t.dt, .stride = n, .offset = y.t.offset });
    }

    pub const coefs = block_prefill_mhc.coefs;
    pub const site = block_prefill_mhc.site;
    pub const sitePieces = block_prefill_mhc.sitePieces;
    pub const siteTriton = block_prefill_mhc.siteTriton;
    pub const finish = block_prefill_mhc.finish;
    pub const postOnly = block_prefill_mhc.postOnly;
    pub const final = block_prefill_mhc.final;
    pub const pos = block_prefill_attn.pos;
    pub const swa = block_prefill_attn.swa;
    pub const indexKeys = block_prefill_attn.indexKeys;
    pub const kvStore = block_prefill_attn.kvStore;
    pub const compress = block_prefill_attn.compress;
    pub const selection = block_prefill_select.selection;
    pub const rowsOf = block_prefill_select.rowsOf;
    pub const scores = block_prefill_select.scores;
    pub const sortKeys = block_prefill_select.sortKeys;
    pub const candidates = block_prefill_select.candidates;
    pub const select = block_prefill_select.select;
    pub const streamSelect = block_prefill_select.streamSelect;
    pub const blockPos = block_prefill_attn.blockPos;
    pub const compProj = block_prefill_attn.compProj;
    pub const encoderEnd = block_prefill_attn.encoderEnd;
    pub const attention = block_prefill_attn.attention;
    pub const core = block_prefill_attn.core;

    /// A multi-segment run's segment `j`: the emitter at its rows and position, the glue on its slot (pf_seg).
    pub fn enter(s: *Seg, sp: Span, j: usize) !void {
        const e = s.e;
        e.row0 = sp.row0;
        e.cnt0 = sp.cnt0;
        e.n = sp.n;
        e.start = sp.start;
        s.setRing();
        // the decoder replay: every row's window from the segment's first row (replay.finish's ctx.lo)
        try e.glue("pf_seg", &.{ .{ .i = @intCast(j) }, try pos(e), try e.buf("w.pos64", .{}, .i64, &.{sp.n}), try e.buf("w.lo", .{}, .i32, &.{sp.n}), .{ .b = s.part == .decoder } });
    }

    /// Back at the whole run (`total` rows from its first segment's start): the glue's state the run's again.
    pub fn leave(s: *Seg, total: i64) !void {
        const e = s.e;
        const first = s.multi.?[0];
        e.row0 = 0;
        e.cnt0 = 0;
        e.n = total;
        e.start = first.start;
        s.setRing();
        try e.glue("pf_seg", &.{.{ .i = -1 }});
    }

    /// forward.staging for `e.n` rows: the slot's ring (2 x window rows), or the staging ring past it.
    pub fn setRing(s: *Seg) void {
        const e = s.e;
        const W: i64 = e.cfg.window;
        s.staged = e.n + W - 1 > 2 * W;
        s.ring = if (s.staged) stageRows(e) else 2 * W;
    }

    pub const moeBlock = block_prefill_moe.moeBlock;
    pub const moeSplit = block_prefill_moe.moeSplit;
    pub const moeRows = block_prefill_moe.moeRows;
    pub const engram = block_prefill_moe.engram;

    // -- the segment ---------------------------------------------------------------------------------------------

    /// Where an exchange (`which`: attn / moe) lands: the layer's (`gathered`), or a short segment's "w.recv.*".
    pub fn recv(s: *Seg, comptime which: []const u8) !Arg {
        const e = s.e;
        return e.buf("{s}.recv." ++ which, .{if (s.short) "w" else "L"}, .bf16, &.{ e.o.world, e.n, e.cfg.hidden });
    }

    pub fn run(s: *Seg, layers: []const u32, head: bool) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        if (s.part == .decoder and s.multi != null) {
            // several slots' decoder replays: every segment's positions, then each one's entering rows from its stash
            try e.glue("mpositions", &.{try e.buf("w.mpos64", .{}, .i64, &.{n})});
            for (s.multi.?, 0..) |sp, j| {
                try s.enter(sp, j);
                try e.glue("ced_load", &try stashArgs(e));
            }
            try s.leave(n);
        } else if (s.part == .decoder) {
            // decoder replay: every row's window starts at R0 (ctx.lo); the entering rows come from the stash
            try e.glue("positions", &.{ try pos(e), try e.buf("w.pos64", .{}, .i64, &.{n}), try e.buf("w.lo", .{}, .i32, &.{n}), .{ .i = e.start } });
            const st = try stashArgs(e);
            try e.glue("ced_load", &st);
        } else if (s.multi != null) {
            // every segment's positions (rope over the run's rows); each segment's own at its CSA2 steps (pf_seg)
            try e.glue("mpositions", &.{try e.buf("w.mpos64", .{}, .i64, &.{n})});
            try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) });
        } else {
            try e.glue("positions", &.{ try pos(e), try e.buf("w.pos64", .{}, .i64, &.{n}), try e.buf("w.lo", .{}, .i32, &.{n}) });
            try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) });
        }
        for (layers, 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            // TF_DSV41_PF_TBO: a layer's attention part starts here (the first's at the program's first call)
            if (i > 0) if (e.marks) |m| try m.append(e.a, e.out.items.len);
            e.layer_ratio = e.cfg.compressRatio(L);
            if (s.part == .decoder and i == 0) {
                // entered (forward._layers): the first block's attention site is the stash's
            } else if (i == 0 or e.cfg.isEngram(L)) {
                // the run's first block, or an Engram block: post alone, Engram, then a site collapsing with the pre
                if (i > 0) try s.postOnly(try e.streams());
                if (e.cfg.isEngram(L)) try s.engram(L);
                try s.site(L, "hc_attn", if (i == 0) .site1 else .site2);
            } else try s.site(L, "hc_attn", .boundary);
            // CED's encoder pass stops at the decoder's first layer once its site and compressor ran (`upto`)
            if (s.part == .encoder and i + 1 == layers.len) return s.encoderEnd(L);
            try s.attention(L);
            if (s.short) try e.prefetchF(L);
            // TF_DSV41_PF_OVERLAP: the partial went out a piece at a time behind its producer
            if (!s.sent_attn) try e.glue("exchange", &.{ try e.buf("L.part", .{}, .bf16, &.{ n, D }), try s.recv("attn") });
            s.sent_attn = false;
            if (e.marks) |m| try m.append(e.a, e.out.items.len); // its MoE part
            try s.site(L, "hc_ffn", .boundary);
            try s.moeBlock(L);
            if (s.short and i + 1 < layers.len) try e.prefetchX(layers[i + 1]);
            if (!s.sent_moe) try e.glue("exchange", &.{ try e.buf("L.moe16", .{}, .bf16, &.{ n, D }), try s.recv("moe") });
            s.sent_moe = false;
            if (e.o.trace) {
                const y = try e.buf("w.trace", .{}, .bf16, &.{ n, 4 * D });
                try s.postOnly(y);
                try e.glue("trace", &.{ y, .{ .i = L } });
            }
        }
        if (head) try s.final();
        s.unpend();
    }

    /// A program ending on an exchange's pieces: its last piece back on the main stream after a join (no open fork).
    fn unpend(s: *Seg) void {
        const e = s.e;
        if (s.pend == 0) return;
        s.pend = 0;
        e.br_join = false;
        const c = &e.out.items[e.out.items.len - 1];
        std.debug.assert(std.mem.eql(u8, c.name, "glue.exchange_rows") and c.side);
        c.side = false;
        c.fork = false;
        c.mark = 0;
        c.join = true;
    }
};

/// The CED stash glue's tensors: the run's streams, normed input and coefficients, then the ring's roles.
pub fn stashArgs(e: *E) ![10]Arg {
    const D: i64 = e.cfg.hidden;
    const R: i64 = ced.ring;
    const c = try Seg.coefs(e, e.coef);
    const x = try e.streams();
    const r0 = e.row0;
    return .{
        try Seg.rowsOf(e, x, .bf16, r0, e.n, 4 * D),     try Seg.rowsOf(e, try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), .bf16, r0, e.n, D),
        try Seg.rowsOf(e, c[0], .f32, r0, e.n, 4),       try Seg.rowsOf(e, c[1], .f32, r0, e.n, 4),
        try Seg.rowsOf(e, c[2], .f32, r0, e.n, 16),      try e.buf(ced.roles[0], .{}, .bf16, &.{ R, 4 * D }),
        try e.buf(ced.roles[1], .{}, .bf16, &.{ R, D }), try e.buf(ced.roles[2], .{}, .f32, &.{ R, 4 }),
        try e.buf(ced.roles[3], .{}, .f32, &.{ R, 4 }),  try e.buf(ced.roles[4], .{}, .f32, &.{ R, 16 }),
    };
}

/// One segment of a multi-segment run: its slot's rows from position `start`, rows [row0, row0 + n) of the run.
pub const Span = struct { slot: u32, start: i64, n: i64, row0: i64 = 0, cnt0: i64 = 0 };

/// forward.staging's ring rows for a prefill segment the slot's ring cannot hold.
fn stageRows(e: *const E) i64 {
    const W: i64 = e.cfg.window;
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(e.o.prefill_rows + W - 1)) catch unreachable);
}

pub const emitMulti = block_prefill_emit.emitMulti;
pub const planRuns = block_prefill_emit.planRuns;

/// A prefill program's part: a whole segment, or one of CED replay's passes (ced.zig).
pub const Part = enum { whole, encoder, decoder };

pub const emitReplay = block_prefill_emit.emitReplay;
pub const emitPrefill = block_prefill_emit.emitPrefill;
pub const tbo_min = block_prefill_emit.tbo_min;
pub const own_scratch = block_prefill_emit.own_scratch;
pub const ownRole = block_prefill_emit.ownRole;
pub const tboRole = block_prefill_emit.tboRole;
pub const Keep = block_prefill_emit.Keep;
pub const ownCalls = block_prefill_emit.ownCalls;
pub const ownCallsKeep = block_prefill_emit.ownCallsKeep;
pub const emitTbo = block_prefill_emit.emitTbo;
