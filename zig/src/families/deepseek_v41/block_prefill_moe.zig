//! A prefill segment's MoE: router, x3gm's ragged launches, the shared expert's GEMMs, and Engram.

const std = @import("std");
const calls = @import("block_calls.zig");
const gm2pf = @import("gm2pf.zig");
const prefill = @import("block_prefill.zig");
const E = prefill.E;
const Arg = calls.Arg;
const router_chunk = prefill.router_chunk;
const gm_bm = prefill.gm_bm;
const gm_widths = prefill.gm_widths;
const Seg = prefill.Seg;
const pieceAt = prefill.Seg.pieceAt;
const rowsOf = prefill.Seg.rowsOf;

// -- MoE: router, x3gm's ragged launches, the shared expert's GEMMs -------------------------------------------

pub fn moeBlock(s: *Seg, L: u32) !void {
    const e = s.e;
    if (e.o.image_rows != 0) return s.moeSplit(L);
    const D: i64 = e.cfg.hidden;
    return s.moeRows(L, e.n, try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), "bias", try e.buf("L.moe16", .{}, .bf16, &.{ e.n, D }), false);
}

/// TF_DSV41_BIAS_VL: image rows through the MoE with gate.bias_vl, text rows with gate.bias, partials merged back.
pub fn moeSplit(s: *Seg, L: u32) !void {
    const e = s.e;
    const D: i64 = e.cfg.hidden;
    const n = e.n;
    const ni: i64 = if (e.o.image_rows < 0) n else e.o.image_rows;
    const nt: i64 = if (e.o.image_rows < 0) n else n - ni;
    const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
    const xi = try e.buf("w.out.img", .{}, .bf16, &.{ ni, D });
    const xt: Arg = if (nt > 0) try e.buf("w.out.txt", .{}, .bf16, &.{ nt, D }) else try e.empty(.bf16);
    try e.glue("image_split", &.{ x, xi, xt });
    const pi = try e.buf("L.moe16.img", .{}, .bf16, &.{ ni, D });
    try s.moeRows(L, ni, xi, "bias_vl", pi, false);
    var pt = try e.empty(.bf16);
    if (nt > 0) {
        pt = try e.buf("L.moe16.txt", .{}, .bf16, &.{ nt, D });
        try s.moeRows(L, nt, xt, "bias", pt, true);
    }
    try e.glue("image_merge", &.{ pi, pt, try e.buf("L.moe16", .{}, .bf16, &.{ n, D }) });
}

/// The MoE over `n` rows at `x` into `part`; `again`: the layer's second call (its own MoE ticket).
pub fn moeRows(s: *Seg, L: u32, n: i64, x: Arg, comptime bias: []const u8, part: Arg, again: bool) !void {
    const e = s.e;
    // the call's own rows are the emitter's rows (the projections' M, their scratch): a plain n-row segment's MoE
    const seg_rows = e.n;
    e.n = n;
    defer e.n = seg_rows;
    const D: i64 = e.cfg.hidden;
    const ex = e.cfg.expertsOf(L);
    const topk: i64 = ex.topk;
    const slots: i64 = topk + 1;
    const Et: i64 = ex.count;
    const I: i64 = e.cfg.expert_width / e.o.world;
    const P = n * topk;
    const pick = try e.buf("L.pick", .{}, .i32, &.{ n, slots });
    const wts = try e.buf("L.wts", .{}, .f32, &.{ n, slots });
    // the router a launch a CHUNK of rows; calls of <= 16 rows the narrow kernel (TF_DSV41_RG_NARROW 2)
    const logits = try e.buf("L.logits", .{}, .f32, &.{ n, Et });
    var c0: i64 = 0;
    while (c0 < n) : (c0 += router_chunk) {
        const m = @min(router_chunk, n - c0);
        const whole = m == n;
        const tile: i64 = for ([_]i64{ 1, 2, 4, 8, 16 }) |t| {
            if (t >= @min(m, 16)) break t;
        } else 16;
        const cfg: [2]i64 = if (n <= 16) .{ 2, 0 } else if (m > 16) .{ 8, 2 } else .{ 8, 1 };
        try e.ext("tf_dsv41_router_gemv_v2.route", &.{
            if (whole) x else try rowsOf(e, x, .bf16, c0, m, D),        try e.weight("L{d}.moe.gate", .{L}, .bf16, &.{ Et, D }),
            try e.weight("L{d}.moe." ++ bias, .{L}, .f32, &.{Et}),      if (whole) pick else try rowsOf(e, pick, .i32, c0, m, slots),
            if (whole) wts else try rowsOf(e, wts, .f32, c0, m, slots), if (whole) logits else try rowsOf(e, logits, .f32, c0, m, Et),
            try e.buf("s.router.cnt", .{}, .i32, &.{128}),              .{ .i = topk },
            .{ .i = slots },                                            .{ .f = E.dec(e.cfg.routed_scale) },
            .{ .i = tile },                                             .{ .i = cfg[0] },
            .{ .i = cfg[1] },                                           .{ .b = true },
            .{ .b = false },                                            try e.empty(.i32),
            try e.empty(.i32),                                          try e.t(.empty, .i32, &.{ 0, 1 }),
            .{ .i = 0 },                                                .{ .b = false },
        });
        if (e.o.r1) {
            // the R1 twin's route always takes the prune list; prefill's is empty (the prune runs after it)
            const last = &e.out.items[e.out.items.len - 1];
            const more = try e.a.alloc(calls.Named, last.args.len + 1);
            @memcpy(more[0..last.args.len], last.args);
            more[last.args.len] = .{ .arg = .{ .list = &.{} } };
            last.args = more;
        } else if (e.o.r1_sig) try e.appendLast(&.{.{ .list = &.{} }});
    }
    // the kit's fp16 weights, the shared slot masked, the routed slots' contiguous copies
    const rs: i64 = if (n == 1) slots else topk;
    const pk = try e.view(.{ .buf = "L.gm.pick" }, .i32, &.{ n, topk }, &.{ rs, 1 }, 0);
    const w6 = try e.view(.{ .buf = "L.gm.wts" }, .f32, &.{ n, topk }, &.{ rs, 1 }, 0);
    try e.glue("gm_picks", &.{ pick, wts, pk, w6 });
    // one ragged row block: rot, a gate/up launch a gate width present, a down launch a down width, combine
    const widths = e.w.gm.get(L);
    const rows: i64 = e.o.expert_rows;
    const zb = rows * (topk + 1) * @max(4 * 2 * I, D); // the scratch z's floats (block_moe)
    // past the 1,024-row scratch rows halve; under TF_DSV41_PF_4K the block is the whole segment
    if (!e.o.pf4k and (P * D > zb or P * I > rows * (topk + 1) * D)) return error.Unsupported;
    const pre = try e.fmt("L{d}.moe", .{L});
    const xg = try e.view(.{ .buf = "s.ex.z" }, .f16, &.{ P, D }, &.{ D, 1 }, 0);
    const xu = try e.view(.{ .buf = "s.ex.z" }, .f16, &.{ P, D }, &.{ D, 1 }, P * D * 2);
    const xd = try e.view(.{ .buf = "s.ex.xg" }, .f16, &.{ P, I }, &.{ I, 1 }, 0);
    const y = try e.view(.{ .buf = "s.ex.z" }, .f32, &.{ P, D }, &.{ D, 1 }, 0);
    try e.ext("tf_dsv41_x3gm_v1.rot", &.{
        x,           .{ .i = D }, pk,          try e.weight("{s}.w1.suh", .{pre}, .f16, &.{ Et, D }), try e.weight("{s}.w3.suh", .{pre}, .f16, &.{ Et, D }),
        xg,          xu,          .{ .i = D }, .{ .i = topk },                                        .{ .i = P },
        .{ .i = 2 },
    });
    const ticket = try e.buf("L.gm.ticket", .{}, .i32, &.{2 * gm_widths});
    try e.glue(if (again) "gm_ticket_again" else "gm_ticket", &.{ticket});
    const tmax = @divFloor(P, gm_bm) + Et + 1;
    const lim: f64 = E.dec(e.cfg.swiglu_limit);
    // TF_DSV41_GM_V2 (gm2pf.zig): x3gm._run_v2's gm2_kernel launches in place of the loop below
    if (e.o.gm_v2 != .off) try gm2pf.emit(e, e.o.gm_v2, .{ .L = L, .pre = pre, .P = P, .Et = Et, .D = D, .I = I, .pk = pk, .xg = xg, .xu = xu, .xd = xd, .y = y, .ticket = ticket, .limit = lim, .widths = widths }) else for (0..2) |down| {
        const wd = (widths orelse return error.NoWidth)[down];
        var j: i64 = 0;
        for (0..32) |kk| {
            const k2: i64 = @intCast(kk);
            if (wd >> @intCast(kk) & 1 == 0) continue;
            const tag = if (down == 1) "d" else "g";
            const plan = [5]Arg{
                try e.buf("L.gm.{s}{d}.order", .{ tag, j }, .i32, &.{P}),
                try e.buf("L.gm.{s}{d}.pe", .{ tag, j }, .i32, &.{tmax}),
                try e.buf("L.gm.{s}{d}.poff", .{ tag, j }, .i32, &.{tmax}),
                try e.buf("L.gm.{s}{d}.pcnt", .{ tag, j }, .i32, &.{tmax}),
                try e.buf("L.gm.{s}{d}.npass", .{ tag, j }, .i32, &.{1}),
            };
            try e.glue("gm_plan", &.{ pk, plan[0], plan[1], plan[2], plan[3], plan[4], .{ .i = k2 }, .{ .b = down == 1 }, .{ .i = gm_bm }, .{ .i = Et } });
            const t = try e.view(ticket.t.role, .i32, &.{1}, &.{1}, (j + if (down == 1) @as(i64, gm_widths) else 0) * 4);
            if (down == 0) {
                try e.ext("tf_dsv41_x3gm_v1.gateup", &.{
                    xg,                                                    xu,
                    try e.buf("s.L{d}.gm.tg", .{L}, .i64, &.{Et}),         try e.buf("s.L{d}.gm.tu", .{L}, .i64, &.{Et}),
                    plan[0],                                               plan[1],
                    plan[2],                                               plan[3],
                    plan[4],                                               t,
                    try e.weight("{s}.w1.svh", .{pre}, .f16, &.{ Et, I }), try e.weight("{s}.w3.svh", .{pre}, .f16, &.{ Et, I }),
                    try e.weight("{s}.w2.suh", .{pre}, .f16, &.{ Et, I }), xd,
                    .{ .i = D },                                           .{ .i = I },
                    .{ .i = k2 },                                          .{ .b = false },
                    .{ .i = 1 },                                           .{ .b = true },
                    .{ .f = lim },                                         .{ .b = true },
                });
            } else {
                try e.ext("tf_dsv41_x3gm_v1.down", &.{
                    xd,                                                    try e.buf("s.L{d}.gm.td", .{L}, .i64, &.{Et}), plan[0],        plan[1],
                    plan[2],                                               plan[3],                                       plan[4],        t,
                    try e.weight("{s}.w2.svh", .{pre}, .f16, &.{ Et, D }), y,                                             .{ .i = I },    .{ .i = D },
                    .{ .i = k2 },                                          .{ .i = 0 },                                   .{ .b = true }, .{ .b = true },
                });
            }
            j += 1;
        }
    }
    const routed = try e.buf("L.routed", .{}, .f32, &.{ n, D });
    try e.ext("tensorfold_exl3_experts_v1.combine", &.{ y, w6, routed, .{ .i = n }, .{ .i = D }, .{ .i = topk } });
    // the shared expert (gate / up GEMMs, clamped SiLU, down); TF_DSV41_PF_OVERLAP: a row piece at a time
    const k: i64 = if (e.cfg.shared_experts > 0 and n == seg_rows and part.t.role == .buf and std.mem.eql(u8, part.t.role.buf, "L.moe16")) s.pieces() else 1;
    if (k > 1) {
        const sh = try e.fmt("{s}.shared.0", .{pre});
        const g = try e.buf("L.sh.g", .{}, .f32, &.{ n, I });
        const u = try e.buf("L.sh.u", .{}, .f32, &.{ n, I });
        const act = try e.buf("L.sh.act", .{}, .f32, &.{ n, I });
        const sy = try e.buf("L.sh.y", .{}, .f32, &.{ n, D });
        const gath = try s.recv("moe");
        var p: i64 = 0;
        while (p < k) : (p += 1) {
            const r = pieceAt(n, k, p);
            const m = r[1] - r[0];
            e.n = m;
            const xr = try rowsOf(e, x, .bf16, r[0], m, D);
            const gr = try rowsOf(e, g, .f32, r[0], m, I);
            const ur = try rowsOf(e, u, .f32, r[0], m, I);
            const ar = try rowsOf(e, act, .f32, r[0], m, I);
            const yr = try rowsOf(e, sy, .f32, r[0], m, D);
            try s.proj(try e.fmt("{s}.w1", .{sh}), .stored, xr, D, I, gr);
            try s.proj(try e.fmt("{s}.w3", .{sh}), .stored, xr, D, I, ur);
            try e.glue("swiglu", &.{ gr, ur, ar, .{ .f = lim } });
            try s.proj(try e.fmt("{s}.w2", .{sh}), .stored, ar, I, D, yr);
            try e.glue("moe_sum", &.{ try rowsOf(e, routed, .f32, r[0], m, D), yr, try rowsOf(e, part, .bf16, r[0], m, D) });
            e.n = n;
            try s.sendPiece(part, gath, r[0], m, p, k);
        }
        s.sent_moe = true;
        return;
    }
    if (e.cfg.shared_experts > 0) {
        const sh = try e.fmt("{s}.shared.0", .{pre});
        const g = try e.buf("L.sh.g", .{}, .f32, &.{ n, I });
        const u = try e.buf("L.sh.u", .{}, .f32, &.{ n, I });
        const act = try e.buf("L.sh.act", .{}, .f32, &.{ n, I });
        const sy = try e.buf("L.sh.y", .{}, .f32, &.{ n, D });
        try s.proj(try e.fmt("{s}.w1", .{sh}), .stored, x, D, I, g);
        try s.proj(try e.fmt("{s}.w3", .{sh}), .stored, x, D, I, u);
        try e.glue("swiglu", &.{ g, u, act, .{ .f = lim } });
        try s.proj(try e.fmt("{s}.w2", .{sh}), .stored, act, I, D, sy);
        try e.glue("moe_sum", &.{ routed, sy, part });
    } else try e.glue("moe_sum", &.{ routed, part });
}

/// Engram before the layer's first site (blocks.engram): wkv over the hashed rows (a prefill matmul), the fusion.
pub fn engram(s: *Seg, L: u32) !void {
    const e = s.e;
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const K: i64 = @as(i64, e.cfg.hashCols()) * e.cfg.engram_head_dim;
    const N: i64 = (@as(i64, e.cfg.engram_max_ngram) + 1) * D;
    const kv = try e.buf("L.engram.kv", .{}, .bf16, &.{ n, N });
    const rws = try e.buf("L.engram.rows", .{}, .bf16, &.{ n, K });
    try e.glue("engram_rows", &.{ rws, .{ .i = L } });
    try s.proj(try e.fmt("L{d}.engram.wkv", .{L}), .strips, rws, K, N, kv);
    const qk = try e.weight("L{d}.engram.qk", .{L}, .f32, &.{ e.cfg.hc_mult, D });
    // image positions (TF_DSV41_IMAGES=native): the gate times vision.window's keep (0 there)
    const keep: ?Arg = if (e.o.image_keep) try e.buf("w.engram.keep", .{}, .f32, &.{n}) else null;
    if (keep) |k| try e.glue("image_keep", &.{k});
    try e.triton("_fuse_dec", .{ n, 1, 1 }, &.{
        .{ .name = "X", .arg = try e.streams() },                              .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
        .{ .name = "KV", .arg = kv },                                          .{ .name = "kv_stride", .arg = .{ .i = N } },
        .{ .name = "QK", .arg = qk },                                          .{ .name = "KEEP", .arg = keep orelse qk },
        .{ .name = "GATE", .arg = qk },                                        .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
        .{ .name = "clamp", .arg = .{ .f = E.dec(e.cfg.engram_gate_clamp) } }, .{ .name = "sqrt_d", .arg = .{ .f = @sqrt(@as(f64, @floatFromInt(D))) } },
        .{ .name = "D", .arg = .{ .i = D } },                                  .{ .name = "BK", .arg = .{ .i = 64 } },
        .{ .name = "UB", .arg = .{ .i = 1024 } },                              .{ .name = "HAS_KEEP", .arg = .{ .b = keep != null } },
        .{ .name = "HAS_GATE", .arg = .{ .b = false } },
    });
}
