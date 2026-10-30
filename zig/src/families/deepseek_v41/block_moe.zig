//! The decode window's mHC sites, MoE (router, prune, group, x3ld, epilogues), Engram and head, as calls.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const E = block.Emitter;
const Arg = calls.Arg;
const wide = @import("block_wide.zig");

/// mhc_cuda.run modes: first site (no post), boundary (post, collapse 2, mix), final norm (post, collapse 2, no mix).
pub const Mode = enum(i64) { boundary = 0, site1 = 1, site2 = 2, final = 3 };

fn coefs(e: *E, set: u1) ![3]Arg {
    return .{
        try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
    };
}

/// One mHC site of layer L (`which`: hc_attn / hc_ffn), or the final norm (L null).
pub fn mhc(e: *E, layer: ?u32, which: []const u8, mode: Mode) !void {
    return mhcInto(e, layer, which, mode, false);
}

/// `mhc`, writing the streams in place when `in_place` (a run without the second streams buffer: CED's replay).
pub fn mhcInto(e: *E, layer: ?u32, which: []const u8, mode: Mode, in_place: bool) !void {
    // past 16 rows or with TF_DSV41_MHC_PFDEC at the knob's rows: the wide path (block_wide.zig)
    if (e.n > wide.cuda_rows or wide.pfOn(e, mode)) return wide.decode(e, layer, which, mode, in_place);
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const post = mode == .boundary or mode == .final;
    const mix = mode != .final;
    const f0 = try e.empty(.f32);
    const b0 = try e.empty(.bf16);
    const x = try e.streams();
    const prev = try coefs(e, e.coef);
    const next = try coefs(e, 1 - e.coef);
    var nw: Arg = undefined;
    var hc = [3]Arg{ b0, f0, f0 };
    if (layer) |L| {
        hc = .{
            try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D }),
            try e.weight("L{d}.{s}.base", .{ L, which }, .f32, &.{6 * @as(i64, e.cfg.hc_mult)}),
            try e.weight("L{d}.{s}.scale", .{ L, which }, .f32, &.{3}),
        };
        nw = try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D});
    } else nw = try e.weight("norm", .{}, .f32, &.{D});
    // DSpark's tap: the mean of the new streams at a boundary entering a target layer (not the window's first site)
    var tap: Arg = b0;
    if (e.o.taps and mode == .boundary and std.mem.eql(u8, which, "hc_attn")) if (layer) |L| {
        const t = e.cfg.dspark_targets.items();
        if (std.mem.indexOfScalar(u32, t, L)) |j| {
            const w: i64 = @as(i64, @intCast(t.len)) * D;
            tap = try e.view(.{ .buf = "w.taps" }, .bf16, &.{ n, D }, &.{ w, 1 }, @as(i64, @intCast(j)) * D * 2);
        }
    };
    const xout: Arg = switch (mode) {
        .site1, .site2 => b0,
        .final => x,
        .boundary => if (in_place) x else try e.spare(),
    };
    awaitCoefs(e);
    try e.ext("tf_dsv41_mhc_cuda_v1.run", &.{
        x,                                                                                                                                   xout,
        // the exchange that ran last: the attention's partials at the FFN site, the MoE's at the next attention / final
        if (post) try e.buf("w.recv.{s}", .{if (std.mem.eql(u8, which, "hc_ffn")) "attn" else "moe"}, .bf16, &.{ e.o.world, n, D }) else b0, if (post) prev[1] else f0,
        if (post) prev[2] else f0,                                                                                                           if (post or mode == .site2) prev[0] else f0,
        hc[0],                                                                                                                               hc[1],
        hc[2],                                                                                                                               try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 }),
        try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D }),                                                                                        tap,
        nw,                                                                                                                                  try e.buf("w.out", .{}, .bf16, &.{ n, D }),
        if (mix) next[0] else f0,                                                                                                            if (mix) next[1] else f0,
        if (mix) next[2] else f0,                                                                                                            try e.buf("w.mhc.cnt", .{}, .i32, &.{4}),
        .{ .i = n },                                                                                                                         .{ .i = @backingInt(mode) },
        .{ .f = E.dec(e.cfg.eps) },                                                                                                          .{ .f = E.dec(e.cfg.hc_eps) },
        .{ .f = E.dec(e.cfg.hc_post_alpha) },                                                                                                .{ .i = e.cfg.hc_sinkhorn_iters },
    });
    if (e.o.r1) {
        // MHC_TAIL (parallel finish) and MHC_DEFER (a mixing site's coefficients as their own launch, joined later)
        const last = &e.out.items[e.out.items.len - 1];
        const more = try e.a.alloc(calls.Named, last.args.len + 2);
        @memcpy(more[0..last.args.len], last.args);
        more[last.args.len] = .{ .arg = .{ .i = e.o.mhc_spin } };
        more[last.args.len + 1] = .{ .arg = .{ .i = @intFromBool(mix) } };
        last.args = more;
        if (mix and e.o.mhc_defer) {
            e.df_side = true; // MHC_DEFER: on the deferred stream, the next mHC call joins it
            e.df_pending = true;
        }
        if (mix) try e.ext("tf_dsv41_mhc_cuda_v1.coef", &.{
            try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 }), hc[1],
            hc[2],                                                  next[0],
            next[1],                                                next[2],
            .{ .i = n },                                            .{ .f = E.dec(e.cfg.eps) },
            .{ .f = E.dec(e.cfg.hc_eps) },                          .{ .f = E.dec(e.cfg.hc_post_alpha) },
            .{ .i = e.cfg.hc_sinkhorn_iters },
        });
    } else if (e.o.r1_sig) try e.appendLast(&.{ .{ .i = -1 }, .{ .i = 0 } }); // the twin's binding, `r1` off: no tail
    if (mix) e.coef = 1 - e.coef;
    if (mode == .boundary and !in_place) e.cur = 1 - e.cur;
}

/// mhc.await_coefs: an mHC call reads the coefficients a deferred launch wrote: the main stream joins it first.
pub fn awaitCoefs(e: *E) void {
    if (!e.df_pending) return;
    e.df_pending = false;
    e.df_join = true;
}

/// mHC post only: the last exchange's post applied to the streams in place, before a non-first Engram block.
pub fn postOnly(e: *E, out: ?calls.Arg) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const x = try e.streams();
    const xout = out orelse x;
    const prev = try coefs(e, e.coef);
    awaitCoefs(e);
    try e.triton("_site", .{ std.math.divCeil(i64, n, 16) catch unreachable, 40, 1 }, &.{
        .{ .name = "X", .arg = x },                       .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
        .{ .name = "XOUT", .arg = xout },                 .{ .name = "G", .arg = try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) },
        .{ .name = "g_rank", .arg = .{ .i = n * D } },    .{ .name = "POST", .arg = prev[1] },
        .{ .name = "COMB", .arg = prev[2] },              .{ .name = "PRE", .arg = x },
        .{ .name = "FN", .arg = x },                      .{ .name = "PART", .arg = x },
        .{ .name = "C", .arg = x },                       .{ .name = "TAP", .arg = x },
        .{ .name = "tap_stride", .arg = .{ .i = 0 } },    .{ .name = "R", .arg = .{ .i = n } },
        .{ .name = "D", .arg = .{ .i = D } },             .{ .name = "NB", .arg = .{ .i = 40 } },
        .{ .name = "BM", .arg = .{ .i = 16 } },           .{ .name = "BK", .arg = .{ .i = 16 } },
        .{ .name = "WORLD", .arg = .{ .i = e.o.world } }, .{ .name = "POST_ON", .arg = .{ .b = true } },
        .{ .name = "COLLAPSE", .arg = .{ .i = 0 } },      .{ .name = "MIX", .arg = .{ .b = false } },
        .{ .name = "TAP_ON", .arg = .{ .b = false } },    .{ .name = "SPLIT", .arg = .{ .b = false } },
    });
}

/// Router GEMV, top-p prune, group, then routed + shared experts: rot_in, x3ld gate/up, epilogue, x3ld down, combine.
pub fn moe(e: *E, L: u32) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const backbone = e.cfg.isBackbone(L);
    // the router folds kit rounding + grouping into its tail when no prune follows; with `r1` it folds the prune too
    const fold = !backbone or e.o.expert_topp == null or e.o.r1;
    const ex = e.cfg.expertsOf(L);
    const slots: i64 = ex.topk + 1;
    const Et: i64 = ex.count + 1;
    const I: i64 = (try @import("partition.zig").share(e.cfg.expert_width, e.o.rank, e.o.world)).len;
    const P = n * slots;
    const rows: i64 = if (backbone) e.o.expert_rows else e.o.dspark_rows;
    const sc: []const u8 = if (backbone) "s.ex" else "s.dx";
    const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
    const pick = try e.buf("L.pick", .{}, .i32, &.{ n, slots });
    const wts = try e.buf("L.wts", .{}, .f32, &.{ n, slots });
    const uids = try e.buf("{s}.ids", .{sc}, .i32, &.{P});
    const ucount = try e.buf("{s}.count", .{sc}, .i32, &.{1});
    const members = try e.buf("{s}.members", .{sc}, .i32, &.{ P, n });
    // router_gemv.config / row_tile: the narrow kernel's (2, 0) to 16 rows, then the prefill config (block_wide.zig)
    const rc = wide.routerCfg(n);
    // the route groups in its tail only on the narrow kernel (<= 16 rows) and GE <= 512; else upstream's group follows
    const grouped = fold and rc[2] == 0 and Et <= 512;
    const scale: f64 = E.dec(e.cfg.routed_scale);
    try e.ext("tf_dsv41_router_gemv_v2.route", &.{
        x,                                                       try e.weight("L{d}.moe.gate", .{L}, .bf16, &.{ ex.count, D }),
        try e.weight("L{d}.moe.bias", .{L}, .f32, &.{ex.count}), pick,
        wts,                                                     try e.buf("L.logits", .{}, .f32, &.{ n, ex.count }),
        try e.buf("s.router.cnt", .{}, .i32, &.{128}),           .{ .i = ex.topk },
        .{ .i = slots },                                         .{ .f = scale },
        .{ .i = rc[0] },                                         .{ .i = rc[1] },
        .{ .i = rc[2] },                                         .{ .b = true },
        .{ .b = fold },                                          if (!grouped) try e.empty(.i32) else uids,
        if (!grouped) try e.empty(.i32) else ucount,             if (!grouped) try e.t(.empty, .i32, &.{ 0, 1 }) else members,
        .{ .i = if (!grouped) 0 else Et },                       .{ .b = grouped },
    });
    if (e.o.r1) {
        // prune fold args [min_w, topp, scale, min_k, orig, drop, dup] (empty for DSpark); drop = table size
        const pa: []const Arg = if (backbone and e.o.expert_topp != null) &.{
            .{ .f = 0.0 }, .{ .f = e.o.expert_topp.? }, .{ .f = scale }, .{ .i = 3 }, .{ .i = 0 }, .{ .i = Et }, .{ .i = 0 },
        } else &.{};
        const last = &e.out.items[e.out.items.len - 1];
        const more = try e.a.alloc(calls.Named, last.args.len + 1);
        @memcpy(more[0..last.args.len], last.args);
        more[last.args.len] = .{ .arg = .{ .list = try e.a.dupe(Arg, pa) } };
        last.args = more;
    } else if (e.o.r1_sig) try e.appendLast(&.{.{ .list = &.{} }}); // the twin's route always takes the prune list
    if (!fold) {
        try e.triton("_prune", .{ std.math.divCeil(i64, n, 16) catch unreachable, 1, 1 }, &.{
            .{ .name = "PICK", .arg = pick },             .{ .name = "WTS", .arg = wts },
            .{ .name = "R", .arg = .{ .i = n } },         .{ .name = "stride", .arg = .{ .i = slots } },
            .{ .name = "MIN_W", .arg = .{ .f = 0.0 } },   .{ .name = "TOPP", .arg = .{ .f = e.o.expert_topp.? } },
            .{ .name = "SCALE", .arg = .{ .f = scale } }, .{ .name = "DROP", .arg = .{ .i = Et } },
            .{ .name = "MIN_K", .arg = .{ .i = 3 } },     .{ .name = "K", .arg = .{ .i = ex.topk } },
            .{ .name = "KP", .arg = .{ .i = 8 } },        .{ .name = "DUP", .arg = .{ .b = false } },
            .{ .name = "BR", .arg = .{ .i = 16 } },       .{ .name = "ORIG", .arg = .{ .b = false } },
        });
        if (e.o.r1_sig) { // the twin's `_prune` takes KIT (`r1`'s PRUNE_KIT) as its last constexpr
            const last = &e.out.items[e.out.items.len - 1];
            const more = try e.a.alloc(calls.Named, last.args.len + 1);
            @memcpy(more[0..last.args.len], last.args);
            more[last.args.len] = .{ .name = "KIT", .arg = .{ .b = false } };
            last.args = more;
        }
        try e.glue("kit_weights", &.{ wts, try e.buf("L.wts16", .{}, .f32, &.{ n, slots }) });
    }
    const group_rot = !grouped and e.o.router_group_rot and n <= 64 and slots <= 9 and Et <= 512 and D == 5120;
    if (!grouped and !group_rot) try e.ext("tensorfold_exl3_experts_v1.group", &.{ pick, uids, ucount, members, .{ .i = n }, .{ .i = slots }, .{ .i = Et } });
    const pre = try e.fmt("s.L{d}.ex", .{L});
    const xg = try e.buf("{s}.xg", .{sc}, .f16, &.{ rows * slots, D });
    const xu = try e.buf("{s}.xu", .{sc}, .f16, &.{ rows * slots, D });
    const xd = try e.buf("{s}.xd", .{sc}, .f16, &.{ rows * slots, I });
    const z = try e.buf("{s}.z", .{sc}, .f32, &.{rows * slots * @max(4 * 2 * I, D)});
    try e.ext(if (group_rot) "tensorfold_exl3_experts_v1.group_rot_in" else "tensorfold_exl3_experts_v1.rot_in", &.{
        x,            .{ .i = D }, pick,        try e.buf("{s}.suh_g", .{pre}, .f16, &.{ Et, D }), try e.buf("{s}.suh_u", .{pre}, .f16, &.{ Et, D }),
        xg,           xu,          .{ .i = n }, .{ .i = D },                                       .{ .i = slots },
        .{ .i = Et },
    });
    if (group_rot) try e.appendLast(&.{ uids, ucount, members });
    const r = e.w.experts.get(L) orelse return error.NoWidth;
    const gu = try block.k2Range(r[0][0], r[0][1]);
    const dn = try block.k2Range(r[1][0], r[1][1]);
    const svh_g = try e.buf("{s}.svh_g", .{pre}, .f16, &.{ Et, I });
    const svh_u = try e.buf("{s}.svh_u", .{pre}, .f16, &.{ Et, I });
    const suh_d = try e.buf("{s}.suh_d", .{pre}, .f16, &.{ Et, I });
    const tp_g = try e.buf("{s}.tp_g", .{pre}, .i64, &.{Et});
    const tp_u = try e.buf("{s}.tp_u", .{pre}, .i64, &.{Et});
    const k2_g = try e.buf("{s}.k2_g", .{pre}, .i32, &.{Et});
    const k2_u = try e.buf("{s}.k2_u", .{pre}, .i32, &.{Et});
    const tp_d = try e.buf("{s}.tp_d", .{pre}, .i64, &.{Et});
    const k2_d = try e.buf("{s}.k2_d", .{pre}, .i32, &.{Et});
    const svh_d = try e.buf("{s}.svh_d", .{pre}, .f16, &.{ Et, D });
    const y = try e.buf("{s}.y", .{sc}, .f32, &.{ rows * slots, D });
    // after a prune, kit_weights' fp16 rounding (a pointwise op the forward runs) gives new weights
    const cw = if (!fold) try e.buf("L.wts16", .{}, .f32, &.{ n, slots }) else wts;
    const out = try e.buf("L.moe", .{}, if (e.o.r1 and backbone) .bf16 else .f32, &.{ n, D });
    const limit: Arg = .{ .f = E.dec(e.cfg.swiglu_limit) };
    if (e.o.x3ld_epi) {
        // TF_DSV41_X3LD_EPI: the same x3ld work with the epilogues in its tail behind a last-arrival ticket; same bits
        const mt = std.math.divCeil(i64, n, 16) catch unreachable;
        const ticket = try e.buf("{s}.epi", .{sc}, .i32, &.{@max(P * mt * @divExact(I, 128), n * @divExact(D, 128))});
        try e.ext("tf_dsv41_x3ld_epi_v1.gateup", &.{
            xg,          xu,          tp_g,         tp_u,            k2_g,            k2_u,        uids,        ucount,
            members,     z,           .{ .i = 2 },  .{ .i = D },     .{ .i = I },     .{ .i = P }, .{ .i = 4 }, .{ .i = slots },
            .{ .i = 2 }, .{ .i = 8 }, .{ .i = 1 },  .{ .i = gu[0] }, .{ .i = gu[1] }, pick,        svh_g,       svh_u,
            suh_d,       xd,          .{ .i = Et }, limit,           .{ .i = 1 },     ticket,
        });
        try e.ext("tf_dsv41_x3ld_epi_v1.down", &.{
            xd,          xd,          tp_d,         tp_d,            k2_d,            k2_d,        uids,        ucount,
            members,     z,           .{ .i = 1 },  .{ .i = I },     .{ .i = D },     .{ .i = P }, .{ .i = 1 }, .{ .i = slots },
            .{ .i = 2 }, .{ .i = 8 }, .{ .i = 1 },  .{ .i = dn[0] }, .{ .i = dn[1] }, pick,        svh_d,       y,
            cw,          out,         .{ .i = Et }, ticket,
        });
        return;
    }
    try e.ext("tf_dsv41_x3ld_v1.grouped", &.{
        xg,              xu,
        tp_g,            tp_u,
        k2_g,            k2_u,
        uids,            ucount,
        members,         z,
        .{ .i = 2 },     .{ .i = D },
        .{ .i = I },     .{ .i = P },
        .{ .i = 4 },     .{ .i = slots },
        .{ .i = 2 },     .{ .i = 8 },
        .{ .i = 1 },     .{ .i = 0 },
        .{ .i = gu[0] }, .{ .i = gu[1] },
        .{ .b = false },
    });
    try e.ext("tensorfold_exl3_experts_v1.gateup_epilogue", &.{
        z,               pick,
        svh_g,           svh_u,
        suh_d,           xd,
        .{ .i = n },     .{ .i = P },
        .{ .i = I },     .{ .i = 4 },
        .{ .i = slots }, .{ .i = Et },
        limit,           .{ .i = 1 },
    });
    try e.ext("tf_dsv41_x3ld_v1.grouped", &.{
        xd,          xd,          tp_d,        tp_d,        k2_d,            k2_d,            uids,            ucount,
        members,     z,           .{ .i = 1 }, .{ .i = I }, .{ .i = D },     .{ .i = P },     .{ .i = 1 },     .{ .i = slots },
        .{ .i = 2 }, .{ .i = 8 }, .{ .i = 1 }, .{ .i = 0 }, .{ .i = dn[0] }, .{ .i = dn[1] }, .{ .b = false },
    });
    try e.ext("tensorfold_exl3_experts_v1.down_combine", &.{
        z,               pick,
        svh_d,           y,
        cw,              out,
        .{ .i = n },     .{ .i = P },
        .{ .i = D },     .{ .i = 1 },
        .{ .i = slots }, .{ .i = Et },
    });
}

/// Engram (layers 1 and 14), before the layer's first mHC site: wkv over the hashed rows, gated fusion into streams.
pub fn engram(e: *E, L: u32) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const K: i64 = @as(i64, e.cfg.hashCols()) * e.cfg.engram_head_dim;
    const N: i64 = (@as(i64, e.cfg.engram_max_ngram) + 1) * D;
    const kv = try e.buf("L.engram.kv", .{}, .bf16, &.{ n, N });
    try e.glue("engram_rows", &.{ try e.buf("L.engram.rows", .{}, .bf16, &.{ n, K }), .{ .i = L } });
    try e.upstream(try e.fmt("L{d}.engram.wkv", .{L}), try e.buf("L.engram.rows", .{}, .bf16, &.{ n, K }), K, N, kv);
    const qk = try e.weight("L{d}.engram.qk", .{L}, .f32, &.{ e.cfg.hc_mult, D });
    try e.triton("_fuse_dec", .{ n, 1, 1 }, &.{
        .{ .name = "X", .arg = try e.streams() },                              .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
        .{ .name = "KV", .arg = kv },                                          .{ .name = "kv_stride", .arg = .{ .i = N } },
        .{ .name = "QK", .arg = qk },                                          .{ .name = "KEEP", .arg = qk },
        .{ .name = "GATE", .arg = qk },                                        .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
        .{ .name = "clamp", .arg = .{ .f = E.dec(e.cfg.engram_gate_clamp) } }, .{ .name = "sqrt_d", .arg = .{ .f = @sqrt(@as(f64, @floatFromInt(D))) } },
        .{ .name = "D", .arg = .{ .i = D } },                                  .{ .name = "BK", .arg = .{ .i = 64 } },
        .{ .name = "UB", .arg = .{ .i = 1024 } },                              .{ .name = "HAS_KEEP", .arg = .{ .b = false } },
        .{ .name = "HAS_GATE", .arg = .{ .b = false } },
    });
}

/// The final norm (mHC final) and this rank's head columns (fp32 logits).
pub fn finish(e: *E) !void {
    try mhc(e, null, "", .final);
    const D: i64 = e.cfg.hidden;
    const V: i64 = (try @import("partition.zig").share(e.cfg.vocab, e.o.rank, e.o.world)).len;
    const logits = try e.buf("w.logits", .{}, .f32, &.{ e.n, V });
    try e.upstream("head", try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), D, V, logits);
    try e.glue("kit_logits", &.{logits});
}
