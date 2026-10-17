//! DSpark's drafting pass (row mode, one slot) as calls: ingest, then the pass; candidates and chain stay on the host.

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const moe = @import("block_moe.zig");
const Config = @import("config.zig").Config;
const E = block.Emitter;
const Arg = calls.Arg;
const dense = dk.dense;
const wide = @import("block_wide.zig");
const buffers = @import("buffers.zig");
const layout = @import("dspark_layout.zig");

pub const ringRows = layout.ringRows;
pub const blocks = layout.blocks;
pub const Statics = layout.Statics;
pub const Layout = layout.Layout;
pub const Slots = layout.Slots;

const Ds = struct {
    e: *E,
    st: Statics,
    sl: Slots = .{},

    fn cfg(d: *const Ds) *const Config {
        return d.e.cfg;
    }

    fn i64s(d: *Ds, at: i64, shape: []const i64) !Arg {
        return d.e.view(.{ .buf = "w.ds.i64" }, .i64, shape, &.{1}, 8 * at);
    }

    fn i32s(d: *Ds, at: i64, shape: []const i64, stride: []const i64) !Arg {
        return d.e.view(.{ .buf = "w.ds.i32" }, .i32, shape, stride, 4 * at);
    }

    /// The block's rings, every slot's (a pass: rows of the stacked ring), or (`one`) slot `sl.slot`'s ring alone.
    fn ring(d: *Ds, L: u32, one: bool) ![2]Arg {
        const r = ringRows(d.cfg());
        const e = d.e;
        if (one and d.sl.ring_slots > 1) {
            const v = try e.fmt("s.L{d}.ring.v", .{L});
            const s = try e.fmt("s.L{d}.ring.s", .{L});
            // a view at the slot's ring (the pass's calls size the stacked role)
            return .{ try e.view(.{ .buf = v }, .u8, &.{ r, 576 }, &.{ 576, 1 }, d.sl.slot * r * 576), try e.view(.{ .buf = s }, .u8, &.{ r, 8 }, &.{ 8, 1 }, d.sl.slot * r * 8) };
        }
        const rr = d.sl.ring_slots * r;
        return .{ try e.buf("s.L{d}.ring.v", .{L}, .u8, &.{ rr, 576 }), try e.buf("s.L{d}.ring.s", .{L}, .u8, &.{ rr, 8 }) };
    }

    /// Triton `_rms` (rms with a weight, bf16 in and out).
    fn rms(d: *Ds, x: Arg, w: Arg, out: Arg, k: i64) !void {
        try d.e.triton("_rms", .{ d.e.n, 1, 1 }, &.{
            .{ .name = "X", .arg = x },                                              .{ .name = "x_rs", .arg = .{ .i = k } },
            .{ .name = "W", .arg = w },                                              .{ .name = "OUT", .arg = out },
            .{ .name = "o_rs", .arg = .{ .i = k } },                                 .{ .name = "K", .arg = .{ .i = k } },
            .{ .name = "inv_k", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(k)) } }, .{ .name = "eps", .arg = .{ .f = E.dec(d.cfg().eps) } },
            .{ .name = "BK", .arg = .{ .i = 512 } },                                 .{ .name = "HAS_W", .arg = .{ .b = true } },
            .{ .name = "NARROW", .arg = .{ .b = true } },                            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// `_kv_store` into a block's ring (ratio 0): ingest at "w.ds.pos", a pass at each row's own position (ROWS).
    fn store(d: *Ds, lat: Arg, L: u32, rows: bool) !void {
        const r = try d.ring(L, !rows);
        const n = d.e.n;
        try d.e.triton("_kv_store", .{ n, 1, 1 }, &.{
            .{ .name = "LAT", .arg = lat },                                                                                          .{ .name = "l_stride", .arg = .{ .i = d.cfg().head_dim } },
            .{ .name = "CS", .arg = try d.e.ropeTable() },                                                                           .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "V", .arg = r[0] },                                                                                           .{ .name = "S", .arg = r[1] },
            .{ .name = "v_stride", .arg = .{ .i = 576 } },                                                                           .{ .name = "s_stride", .arg = .{ .i = 8 } },
            .{ .name = "POS", .arg = if (rows) try d.i64s(d.st.positions(), &.{n}) else try d.e.buf("w.ds.pos", .{}, .i32, &.{1}) }, .{ .name = "RATIO", .arg = .{ .i = 0 } },
            .{ .name = "RING", .arg = .{ .i = ringRows(d.cfg()) } },                                                                 .{ .name = "PT", .arg = .none },
            .{ .name = "PSH", .arg = .{ .i = 0 } },                                                                                  .{ .name = "SL", .arg = if (rows) try d.i64s(d.st.writeRows(), &.{n}) else .none },
            .{ .name = "PTS", .arg = .{ .i = 0 } },                                                                                  .{ .name = "ROWS", .arg = .{ .b = rows } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    fn normStore(d: *Ds, kv: Arg, L: u32, rows: bool) !void {
        const r = try d.ring(L, !rows);
        try d.e.normStore(kv, try d.e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{d.cfg().head_dim}), r[0], r[1], if (rows) try d.i64s(d.st.positions(), &.{d.e.n}) else try d.e.buf("w.ds.pos", .{}, .i32, &.{1}), if (rows) try d.i64s(d.st.writeRows(), &.{d.e.n}) else .none, ringRows(d.cfg()), rows);
    }

    /// main_proj: upstream rot_in + linear with split K (its workspace "L.ds.z"), [n, 3 D] -> [n, D].
    fn mainProj(d: *Ds, x: Arg, y: Arg) !void {
        const e = d.e;
        const w = "dspark.main_proj";
        const D: i64 = d.cfg().hidden;
        const K: i64 = @as(i64, @intCast(d.cfg().dspark_targets.items().len)) * D;
        const k2: i64 = try e.w.k2(w);
        const xh = try e.buf("L.{s}.xh", .{w}, .f16, &.{ e.n, K });
        try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{K}), xh });
        const pl = dense.plan(@intCast(K), @intCast(D));
        const st = dense.strides(@intCast(K), @intCast(k2));
        const sk: i64 = @intCast(pl[0]);
        try e.ext("tensorfold_exl3_linear_v4.linear", &.{
            xh,                                                                      try e.strips(w, K, D),                                         .{ .i = st[0] }, .{ .i = st[1] }, try e.weight("{s}.svh", .{w}, .f16, &.{D}), .none,                     y,
            if (sk > 1) try e.buf("L.ds.z", .{}, .f32, &.{sk * e.n * D}) else .none, try e.buf("L.{s}.cnt", .{w}, .i32, &.{8 * @divExact(D, 128)}), .{ .i = k2 },    .{ .i = 2 },     .{ .i = sk },                               .{ .i = @intCast(pl[1]) },
        });
    }

    /// One ingest: `n` committed rows' taps (from row `skip` of "w.taps") into every block's ring.
    fn ingest(d: *Ds, skip: i64) !void {
        try d.ingestProj(skip);
        try d.ingestBlocks(0);
    }

    /// An ingest's first half: main_proj and main_norm of `n` taps rows (from row `skip`) into "w.ds.mx" rows [0, n).
    fn ingestProj(d: *Ds, skip: i64) !void {
        const e = d.e;
        const D: i64 = d.cfg().hidden;
        const T: i64 = @as(i64, @intCast(d.cfg().dspark_targets.items().len)) * D;
        const taps = try e.view(.{ .buf = d.sl.taps_role }, .bf16, &.{ e.n, T }, &.{ T, 1 }, skip * T * 2);
        const mp = try e.buf("L.ds.mp", .{}, .bf16, &.{ e.n, D });
        const mx = try e.buf("w.ds.mx", .{}, .bf16, &.{ e.n, D });
        try d.mainProj(taps, mp);
        try d.rms(mp, try e.weight("dspark.main_norm", .{}, .f32, &.{D}), mx, D);
    }

    /// An ingest's second half: each block's wkv, kv_norm and store of `n` "w.ds.mx" rows into slot `sl.slot`'s rings.
    fn ingestBlocks(d: *Ds, mx_row: i64) !void {
        const e = d.e;
        const D: i64 = d.cfg().hidden;
        const hd: i64 = d.cfg().head_dim;
        const mx = try e.view(.{ .buf = "w.ds.mx" }, .bf16, &.{ e.n, D }, &.{ D, 1 }, mx_row * D * 2);
        var bs: [8]u32 = undefined;
        for (blocks(d.cfg(), &bs), 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            e.layer_ratio = 0;
            const kv = try e.buf("L.kv", .{}, .bf16, &.{ e.n, hd });
            const kvn = try e.buf("L.kvn", .{}, .bf16, &.{ e.n, hd });
            try e.fused("x", &.{.{ .w = try e.fmt("L{d}.attn.wkv", .{L}), .x = mx, .y = kv, .k = @intCast(D), .n = @intCast(hd) }});
            if (e.kvNormStoreOn()) {
                try d.normStore(kv, L, false);
            } else {
                try d.rms(kv, try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}), kvn, hd);
                try d.store(kvn, L, false);
            }
        }
    }

    fn coefs(d: *Ds, set: u1) ![3]Arg {
        const e = d.e;
        return .{
            try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
            try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
            try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
        };
    }

    /// A boundary on the Triton mHC path: `_site` over the exchanged partials `recv`, then `_finish_k`.
    fn site(d: *Ds, L: u32, which: []const u8, recv: []const u8) !void {
        const e = d.e;
        // MHC_DEFER: block 0's site may have left its coefficients on the deferred stream; this boundary reads them
        moe.awaitCoefs(e);
        // TF_DSV41_MHC_PFDEC: mhc_pf in place (a CTA reads only what it writes) + coef_kernel (block_wide.zig)
        if (wide.pfOn(e, wide.Mode.boundary)) return wide.mhc(e, L, which, .boundary, true, null, null);
        const n = e.n;
        const D: i64 = d.cfg().hidden;
        const x = try e.streams();
        const prev = try d.coefs(e.coef);
        const next = try d.coefs(1 - e.coef);
        const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
        const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
        const hm: i64 = d.cfg().hc_mult;
        try e.triton("_site", .{ std.math.divCeil(i64, n, 16) catch unreachable, 40, 1 }, &.{
            .{ .name = "X", .arg = x },                                                                         .{ .name = "x_stride", .arg = .{ .i = hm * D } },
            .{ .name = "XOUT", .arg = x },                                                                      .{ .name = "G", .arg = try e.buf("w.recv.{s}", .{recv}, .bf16, &.{ e.o.world, n, D }) },
            .{ .name = "g_rank", .arg = .{ .i = n * D } },                                                      .{ .name = "POST", .arg = prev[1] },
            .{ .name = "COMB", .arg = prev[2] },                                                                .{ .name = "PRE", .arg = prev[0] },
            .{ .name = "FN", .arg = try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * hm, hm * D }) }, .{ .name = "PART", .arg = part },
            .{ .name = "C", .arg = c },                                                                         .{ .name = "TAP", .arg = x },
            .{ .name = "tap_stride", .arg = .{ .i = 0 } },                                                      .{ .name = "R", .arg = .{ .i = n } },
            .{ .name = "D", .arg = .{ .i = D } },                                                               .{ .name = "NB", .arg = .{ .i = 40 } },
            .{ .name = "BM", .arg = .{ .i = 16 } },                                                             .{ .name = "BK", .arg = .{ .i = 16 } },
            .{ .name = "WORLD", .arg = .{ .i = e.o.world } },                                                   .{ .name = "POST_ON", .arg = .{ .b = true } },
            .{ .name = "COLLAPSE", .arg = .{ .i = 2 } },                                                        .{ .name = "MIX", .arg = .{ .b = true } },
            .{ .name = "TAP_ON", .arg = .{ .b = false } },                                                      .{ .name = "SPLIT", .arg = .{ .b = false } },
        });
        const norm = if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm";
        try e.triton("_finish_k", .{ n, 1, 1 }, &.{
            .{ .name = "PART", .arg = part },
            .{ .name = "BASE", .arg = try e.weight("L{d}.{s}.base", .{ L, which }, .f32, &.{6 * hm}) },
            .{ .name = "SCALE", .arg = try e.weight("L{d}.{s}.scale", .{ L, which }, .f32, &.{3}) },
            .{ .name = "PRE", .arg = next[0] },
            .{ .name = "POST", .arg = next[1] },
            .{ .name = "COMB", .arg = next[2] },
            .{ .name = "C", .arg = c },
            .{ .name = "NW", .arg = try e.weight("L{d}.{s}", .{ L, norm }, .f32, &.{D}) },
            .{ .name = "OUT", .arg = try e.buf("w.out", .{}, .bf16, &.{ n, D }) },
            .{ .name = "eps", .arg = .{ .f = E.dec(d.cfg().eps) } },
            .{ .name = "hc_eps", .arg = .{ .f = E.dec(d.cfg().hc_eps) } },
            .{ .name = "post_alpha", .arg = .{ .f = E.dec(d.cfg().hc_post_alpha) } },
            .{ .name = "D", .arg = .{ .i = D } },
            .{ .name = "NB", .arg = .{ .i = 40 } },
            .{ .name = "ITERS", .arg = .{ .i = d.cfg().hc_sinkhorn_iters } },
            .{ .name = "COEF", .arg = .{ .b = true } },
            .{ .name = "BLOCK", .arg = .{ .i = 1024 } },
            .{ .name = "UNROLL", .arg = .{ .b = true } },
            .{ .name = "NORM_FIRST", .arg = .{ .b = true } },
        });
        e.coef = 1 - e.coef;
    }

    /// A block's attention over its ring (drafter._attention, row mode): this rank's fp32 partial in "L.part".
    fn attention(d: *Ds, L: u32) !void {
        const e = d.e;
        const cf = d.cfg();
        const n = e.n;
        const D: i64 = cf.hidden;
        const H: i64 = cf.heads / e.o.world;
        const hd: i64 = cf.head_dim;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, cf.q_lora });
        const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
        try e.fused("x", &.{
            .{ .w = try e.fmt("L{d}.attn.wq_a", .{L}), .x = x, .y = qa, .k = @intCast(D), .n = cf.q_lora },
            .{ .w = try e.fmt("L{d}.attn.wkv", .{L}), .x = x, .y = kv, .k = @intCast(D), .n = @intCast(hd) },
        });
        const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, cf.q_lora });
        const kvn = try e.buf("L.kvn", .{}, .bf16, &.{ n, hd });
        try d.rms(qa, try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{cf.q_lora}), qn, cf.q_lora);
        if (!e.kvNormStoreOn()) try d.rms(kv, try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}), kvn, hd);
        const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
        try e.fused("q", &.{.{ .w = try e.fmt("L{d}.attn.wq_b", .{L}), .x = qn, .y = q, .k = cf.q_lora, .n = @intCast(H * hd) }});
        // table.apply(q.to(fp32)): the cast is torch's (glue), the rotation copies into bf16
        const qf = try e.buf("L.qf", .{}, .f32, &.{ n, H, hd });
        try e.glue("widen", &.{ q, qf });
        const qr = try e.buf("L.qr", .{}, .bf16, &.{ n, H, hd });
        try e.triton("_rope", .{ n, @divExact(H, 4), 1 }, &.{
            .{ .name = "X", .arg = qf },                                    .{ .name = "x_rs", .arg = .{ .i = H * hd } },
            .{ .name = "x_hs", .arg = .{ .i = hd } },                       .{ .name = "OUT", .arg = qr },
            .{ .name = "o_rs", .arg = .{ .i = H * hd } },                   .{ .name = "o_hs", .arg = .{ .i = hd } },
            .{ .name = "CS", .arg = try e.ropeTable() },                    .{ .name = "cs_rs", .arg = .{ .i = 64 } },
            .{ .name = "POS", .arg = try d.i64s(d.st.positions(), &.{n}) }, .{ .name = "H", .arg = .{ .i = H } },
            .{ .name = "NOPE", .arg = .{ .i = hd - 64 } },                  .{ .name = "NB", .arg = .{ .i = 64 } },
            .{ .name = "HALF", .arg = .{ .i = 32 } },                       .{ .name = "BH", .arg = .{ .i = 4 } },
            .{ .name = "INV", .arg = .{ .b = false } },                     .{ .name = "COPY", .arg = .{ .b = true } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
        if (e.kvNormStoreOn()) try d.normStore(kv, L, true) else try d.store(kvn, L, true);
        const r = try d.ring(L, false);
        const W: i64 = cf.window;
        const rows_scr = n * 5 * H;
        const o = try e.buf("L.o", .{}, .bf16, &.{ n, H, hd });
        try e.ext("tf_dsv41_attn_cuda_v1.attn", &.{
            qr,                                                       r[0],
            r[1],                                                     try d.i32s(d.st.tokens(), &.{ n, W }, &.{ W, 1 }),
            try d.i32s(d.st.counts(), &.{n}, &.{1}),                  r[0],
            r[1],                                                     try d.i32s(d.st.lo(), &.{n}, &.{1}),
            try d.i32s(d.st.hi(), &.{n}, &.{1}),                      try d.i64s(d.st.positions(), &.{n}),
            try d.i64s(d.st.readRows(), &.{n}),                       try e.empty(.i32),
            try e.weight("L{d}.attn.sink", .{L}, .f32, &.{H}),        try e.ropeTable(),
            try e.buf("s.attn.po", .{}, .f32, &.{rows_scr * 512}),    try e.buf("s.attn.pm", .{}, .f32, &.{rows_scr}),
            try e.buf("s.attn.pl", .{}, .f32, &.{rows_scr}),          o,
            try e.buf("s.attn.ticket", .{}, .i32, &.{e.ticketLen()}), .{ .i = ringRows(cf) },
            .{ .i = 0 },                                              .{ .i = 0 },
            .{ .i = 4 },
        });
        // `r1`: the split kernel's CTAs a chunk; q arrives rotated (no ATTN_ROPE fold)
        if (e.o.r1) try d.appendArgs(&.{ .{ .i = E.attnSplit(n) }, .{ .i = 0 } }) else if (e.o.r1_sig) try d.appendArgs(&.{ .{ .i = 0 }, .{ .i = 0 } });
        // o: wo_a's groups over the heads' columns, then wo_b (this rank's fp32 partial)
        const g: usize = cf.o_groups / e.o.world;
        const gk: i64 = @divExact(H * hd, @as(i64, @intCast(g)));
        const oa = try e.buf("L.oa", .{}, .bf16, &.{ n, @as(i64, @intCast(g)) * cf.o_lora });
        var po: [8]E.Proj = undefined;
        for (po[0..g], 0..) |*p, i| p.* = .{
            .w = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i }),
            .x = try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ H * hd, 1 }, @as(i64, @intCast(i)) * gk * 2),
            .y = oa,
            .col = i * cf.o_lora,
            .k = @intCast(gk),
            .n = cf.o_lora,
        };
        try e.fused("o", po[0..g]);
        try e.fused("wo_b", &.{.{ .w = try e.fmt("L{d}.attn.wo_b", .{L}), .x = oa, .y = try e.buf("L.part", .{}, .f32, &.{ n, D }), .k = g * cf.o_lora, .n = @intCast(D) }});
    }

    /// The final norm with DSpark's weight (mhc_cuda final), its hidden left in "L.mhc.c" for the confidence head.
    fn final(d: *Ds) !void {
        const e = d.e;
        const n = e.n;
        // a pass of more than 16 rows (4 slots: 20, TF_DSV41_ROWS_CAP past 16): mhc.final on Triton (block_wide.zig)
        if (n > wide.cuda_rows) return wide.mhc(e, null, "", .final, false, "dspark.norm", null);
        const D: i64 = d.cfg().hidden;
        const f0 = try e.empty(.f32);
        const b0 = try e.empty(.bf16);
        const x = try e.streams();
        const prev = try d.coefs(e.coef);
        moe.awaitCoefs(e); // mhc.final: await_coefs(prev)
        try e.ext("tf_dsv41_mhc_cuda_v1.run", &.{
            x,                                                          x,
            try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }), prev[1],
            prev[2],                                                    prev[0],
            b0,                                                         f0,
            f0,                                                         try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 }),
            try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D }),               b0,
            try e.weight("dspark.norm", .{}, .f32, &.{D}),              try e.buf("w.out", .{}, .bf16, &.{ n, D }),
            f0,                                                         f0,
            f0,                                                         try e.buf("w.mhc.cnt", .{}, .i32, &.{4}),
            .{ .i = n },                                                .{ .i = @backingInt(moe.Mode.final) },
            .{ .f = E.dec(d.cfg().eps) },                               .{ .f = E.dec(d.cfg().hc_eps) },
            .{ .f = E.dec(d.cfg().hc_post_alpha) },                     .{ .i = d.cfg().hc_sinkhorn_iters },
        });
        if (e.o.r1) try d.appendArgs(&.{ .{ .i = e.o.mhc_spin }, .{ .i = 0 } }) // MHC_TAIL; a final mixes nothing
        else if (e.o.r1_sig) try d.appendArgs(&.{ .{ .i = -1 }, .{ .i = 0 } });
    }

    /// The `r1` trailing arguments onto the last call.
    fn appendArgs(d: *Ds, args: []const Arg) !void {
        const e = d.e;
        const last = &e.out.items[e.out.items.len - 1];
        const more = try e.a.alloc(calls.Named, last.args.len + args.len);
        @memcpy(more[0..last.args.len], last.args);
        for (more[last.args.len..], args) |*m, x| m.* = .{ .arg = x };
        last.args = more;
    }

    /// One pass of `n` rows: the embedded block, the blocks, the final norm, the head, then the host's step.
    fn pass(d: *Ds) !void {
        const e = d.e;
        const D: i64 = d.cfg().hidden;
        const n = e.n;
        try e.glue("ds_embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }), try d.i64s(d.st.ids(), &.{n}) });
        var bs: [8]u32 = undefined;
        for (blocks(d.cfg(), &bs), 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            e.layer_ratio = 0;
            if (i == 0) try moe.mhc(e, L, "hc_attn", .site1) else try d.site(L, "hc_attn", "moe");
            try d.attention(L);
            try e.glue("exchange_f32", &.{ try e.buf("L.part", .{}, .f32, &.{ n, D }), try e.buf("w.recv.attn", .{}, .bf16, &.{ e.o.world, n, D }) });
            try d.site(L, "hc_ffn", "attn");
            try moe.moe(e, L);
            try e.glue("exchange_f32", &.{ try e.buf("L.moe", .{}, .f32, &.{ n, D }), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) });
        }
        try d.final();
        const V: i64 = d.cfg().vocab / e.o.world;
        const logits = try e.buf("w.ds.logits", .{}, .f32, &.{ n, V });
        try e.upstream("head", try e.buf("w.out", .{}, .bf16, &.{ n, D }), D, V, logits);
        try e.glue("ds_out", &.{ logits, try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D }) });
    }
};

/// An ingest of `n` rows from row `skip` of "w.taps" (arena-owned by `a`).
pub fn emitIngest(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64, skip: i64) ![]calls.Call {
    return emitIngestAt(a, cfg, w, o, n, skip, .{});
}

/// TF_DSV41_DRAFT_BATCH_PROJ: main_proj + main_norm of `n` taps rows in one call; rows stay independent, bit for bit.
pub fn emitIngestProj(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64, skip: i64, taps_role: []const u8) ![]calls.Call {
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = 0 };
    var d: Ds = .{ .e = &e, .st = .{ .n = n, .window = cfg.window }, .sl = .{ .taps_role = taps_role } };
    try d.ingestProj(skip);
    return e.out.items;
}

/// The rest of a slot's ingest after emitIngestProj: `n` rows of "w.ds.mx" from `mx_row` into slot `sl.slot`'s rings.
pub fn emitIngestBlocks(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64, mx_row: i64, sl: Slots) ![]calls.Call {
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = 0, .begin = .none };
    var d: Ds = .{ .e = &e, .st = .{ .n = n, .window = cfg.window }, .sl = sl };
    try d.ingestBlocks(mx_row);
    return e.out.items;
}

/// An ingest of `n` rows from row `skip` of `sl.taps_role` into slot `sl.slot`'s rings.
pub fn emitIngestAt(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64, skip: i64, sl: Slots) ![]calls.Call {
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = 0 };
    var d: Ds = .{ .e = &e, .st = .{ .n = n, .window = cfg.window }, .sl = sl };
    try d.ingest(skip);
    return e.out.items;
}

pub const sideRoles = layout.sideRoles;
pub const renameRoles = layout.renameRoles;
pub const sideRole = layout.sideRole;

/// The chain's roles (TF_DSV41_DRAFT_CHAIN; dspark_dev.zig binds them): `_chain`'s tensors at the most slots a round.
pub const chain_roles = struct {
    pub const cand = "s.ds.chain.cand";
    pub const cval = "s.ds.chain.cval";
    pub const hid = "s.ds.chain.hid";
    pub const anchor = "s.ds.chain.anchor";
    pub const ipar = "s.ds.chain.ipar";
    pub const fpar = "s.ds.chain.fpar";
    pub const draft = "s.ds.chain.draft";
    pub const conf = "s.ds.chain.conf";
    pub const scr = "s.ds.chain.scr";
};

/// The chain kernel's constants (KP, CH, RB).
pub const chain_kp = 128;
pub const chain_ch = 16;
pub const chain_rb = 32;

/// TF_DSV41_DRAFT_CHAIN: the `_chain` kernel over `slots` slots of `n` rows, `c` candidates a row, optional conf head.
pub fn emitChain(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64, slots: i64, c: i64, conf: bool) ![]calls.Call {
    if (c > chain_kp) return error.CandidateWidth;
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = slots * n, .start = 0 };
    const S = slots;
    const R: i64 = cfg.dspark_markov_rank;
    const D: i64 = cfg.hidden;
    const V: i64 = cfg.vocab;
    const d: i64 = if (conf) D else 1024;
    const r = chain_roles;
    const cand = try e.buf(r.cand, .{}, .i32, &.{ S, n, c });
    const cval = try e.buf(r.cval, .{}, .f32, &.{ S, n, c });
    const hid = if (conf) try e.buf(r.hid, .{}, .bf16, &.{ S, n, D }) else cval;
    const cw = if (conf) try e.weight("dspark.conf", .{}, .f32, &.{D + R}) else cval;
    const cf = if (conf) try e.buf(r.conf, .{}, .f32, &.{ S, n }) else cval;
    try e.triton("_chain", .{ S, 1, 1 }, &.{
        .{ .name = "CAND", .arg = cand },                                                        .{ .name = "CVAL", .arg = cval },
        .{ .name = "c_stride", .arg = .{ .i = c } },                                             .{ .name = "n_cand", .arg = .{ .i = c } },
        .{ .name = "W1", .arg = try e.weight("dspark.markov.w1", .{}, .bf16, &.{ V, R }) },      .{ .name = "W2", .arg = try e.weight("dspark.markov.w2", .{}, .bf16, &.{ V, R }) },
        .{ .name = "HID", .arg = hid },                                                          .{ .name = "CW", .arg = cw },
        .{ .name = "ANCHOR", .arg = try e.buf(r.anchor, .{}, .i32, &.{S}) },                     .{ .name = "IPAR", .arg = try e.buf(r.ipar, .{}, .i64, &.{ S, 4 }) },
        .{ .name = "FPAR", .arg = try e.buf(r.fpar, .{}, .f32, &.{ S, 3 }) },                    .{ .name = "DRAFT", .arg = try e.buf(r.draft, .{}, .i32, &.{ S, n }) },
        .{ .name = "CONF", .arg = cf },                                                          .{ .name = "SCR", .arg = try e.buf(r.scr, .{}, .f64, &.{ S, 2, chain_kp }) },
        .{ .name = "D", .arg = .{ .i = d } },                                                    .{ .name = "N", .arg = .{ .i = n } },
        .{ .name = "KP", .arg = .{ .i = chain_kp } },                                            .{ .name = "CH", .arg = .{ .i = chain_ch } },
        .{ .name = "R", .arg = .{ .i = R } },                                                    .{ .name = "RB", .arg = .{ .i = @max(16, @min(chain_rb, R)) } },
        .{ .name = "HB", .arg = .{ .i = @intCast(std.math.gcd(@as(u64, @intCast(d)), 1024)) } }, .{ .name = "HAS_CONF", .arg = .{ .b = conf } },
    });
    return e.out.items;
}

/// A pass of `n` rows (the block: dspark_block).
pub fn emitPass(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64) ![]calls.Call {
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = 0 };
    var d: Ds = .{ .e = &e, .st = .{ .n = n, .window = cfg.window } };
    try d.pass();
    if (o.holdsDeferred(e.n)) calls.holdDeferred(e.out.items);
    return e.out.items;
}

/// A pass over `slots` slots of `n` rows each (row mode over `ring_slots` stacked rings; statics by `Layout.of`).
pub fn emitPassSlots(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, n: i64, slots: i64, ring_slots: i64) ![]calls.Call {
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = slots * n, .start = 0 };
    var d: Ds = .{ .e = &e, .st = .{ .n = n, .window = cfg.window, .lay = Layout.of(slots, n, cfg.window) }, .sl = .{ .ring_slots = ring_slots } };
    try d.pass();
    if (o.holdsDeferred(e.n)) calls.holdDeferred(e.out.items);
    return e.out.items;
}

/// A capture's set P ops by drafter phase (i0, d0, i1, d1), without `_keys` / `_chain`, with rows and taps row skip.
pub const Phase = struct { name: []const u8, n: i64, skip: i64, ops: []std.json.ObjectMap };

pub fn phases(a: std.mem.Allocator, io: std.Io, dir: []const u8, cfg: *const Config) ![]Phase {
    const cwd = std.Io.Dir.cwd();
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "meta.json" }), a, .limited(1 << 24)), .{});
    const prompt = meta.object.get("prompt").?.integer;
    var by: std.StringArrayHashMapUnmanaged(std.ArrayList(std.json.ObjectMap)) = .empty;
    const text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "ops.jsonl" }), a, .limited(1 << 34));
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const op = (try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{})).object;
        if (!std.mem.eql(u8, op.get("set").?.string, "P")) continue;
        const name = op.get("name").?.string;
        if (std.mem.eql(u8, name, "_keys") or std.mem.eql(u8, name, "_chain")) continue;
        const g = try by.getOrPut(a, op.get("phase").?.string);
        if (!g.found_existing) g.value_ptr.* = .empty;
        try g.value_ptr.append(a, op);
    }
    const out = try a.alloc(Phase, by.count());
    for (out, by.keys(), by.values()) |*p, ph, ops| {
        // rows: the prompt's last `window` (i0) or the decode row (i1); a pass: the block; "q<k>": k blocks
        const n: i64 = if (ph[0] == 'i') (if (std.mem.eql(u8, ph, "i0")) cfg.window else 1) else if (ph[0] == 'q') try slotsOf(ph) * cfg.dspark_block else cfg.dspark_block;
        p.* = .{ .name = ph, .n = n, .skip = if (std.mem.eql(u8, ph, "i0")) prompt - cfg.window else 0, .ops = ops.items };
    }
    return out;
}

/// "q<k>": a capture's pass over k slots (row mode, slots padded to a power of two).
fn slotsOf(name: []const u8) !i64 {
    return std.fmt.parseInt(i64, name[1..], 10);
}

/// A phase's launches (glue left out).
pub fn phaseCalls(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, p: Phase) ![]const calls.Call {
    const all = if (p.name[0] == 'i') try emitIngest(a, cfg, w, o, p.n, p.skip) else if (p.name[0] == 'q') blk: {
        const k = try slotsOf(p.name);
        break :blk try emitPassSlots(a, cfg, w, o, cfg.dspark_block, k, k);
    } else try emitPass(a, cfg, w, o, p.n);
    return calls.launches(a, all);
}

/// Host gate of the emitters against a capture's set P: each drafter phase's launches, Triton variants and dataflow.
pub fn checkDraft(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, experts: []const u8, log: *std.Io.Writer, show: usize) !usize {
    const check = @import("calls_check.zig");
    const cuda = @import("cuda");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const w = try check.widthsFromCapture(a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.json" }), a, .limited(1 << 28)), experts);
    const cfg: Config = .{};
    var variants: ?check.Variants = null;
    if (cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "aot", "aot.json" }), a, .limited(1 << 26))) |text| {
        variants = .{ .specs = (try cuda.aot.parseSpecs(a, text)).value.kernels };
    } else |_| {}
    var roles: calls.Roles = .{ .a = a };
    var flow: check.Flow = .{ .a = a };
    var bad: usize = 0;
    var opts = try check.optionsOf(a, io, dir);
    const ps = try phases(a, io, dir, &cfg);
    {
        const lists = try a.alloc([]std.json.ObjectMap, ps.len);
        for (lists, ps) |*l, p| l.* = p.ops;
        opts.r1_sig = !opts.r1 and check.r1Sig(lists);
    }
    for (ps) |p| {
        const cs = try phaseCalls(a, &cfg, &w, opts, p);
        var equal: usize = 0;
        var shown: usize = 0;
        const m = @min(cs.len, p.ops.len);
        for (cs[0..m], p.ops[0..m], 0..) |*c, op, i| {
            roles.begin(c.begin);
            flow.begin(c.begin);
            const where = try std.fmt.allocPrint(a, "P {s} #{d}", .{ p.name, i });
            try flow.call(c, op, log, where);
            const vwhy = if (variants) |*v| try v.check(a, c, op) else null;
            const why = vwhy orelse try calls.check(a, c, op, &roles);
            if (why) |y| {
                bad += 1;
                if (shown < show) try log.print("  {s} {s}: {s}\n", .{ where, c.name, y });
                shown += 1;
            } else equal += 1;
        }
        if (cs.len != p.ops.len) bad += 1;
        try log.print("P {s} (n {d}): {d}/{d} calls equal, {d} emitted\n", .{ p.name, p.n, equal, p.ops.len, cs.len });
        try log.flush();
    }
    try log.print("dataflow: {d} miswired; glue-written roles:", .{flow.miswired});
    for (flow.glue.keys(), flow.glue.values()) |k, v| try log.print(" {s}={d}", .{ k, v });
    if (variants) |v| try log.print("\nTriton variants: {d} calls, {d} picked wrong", .{ v.checked, v.wrong });
    try log.print("\ninputs:", .{});
    for (flow.inputs.keys(), flow.inputs.values()) |k, v| try log.print(" {s}={d}", .{ k, v });
    try log.print("\n", .{});
    return bad + flow.miswired;
}
