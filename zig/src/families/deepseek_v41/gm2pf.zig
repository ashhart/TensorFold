//! x3gm v2 on prefill (TF_DSV41_GM_V2): a segment's routed experts through gm2_kernel, same bits as gm_kernel.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const calls = @import("block_calls.zig");
const knob_types = @import("knob_types.zig");
const block = @import("block.zig");
const E = block.Emitter;
const Arg = calls.Arg;

const Mode = knob_types.GmMode;

/// x3gm.MAX_WIDTHS: the ticket's down half starts there
pub const max_widths: i64 = 7;
/// x3gm.BM: members a pass of gm2_kernel's tiles (V2_GU 0 / 1, V2_DN 0)
pub const bm: i64 = 64;
/// x3gm.tuned2(shx False) without TF_DSV41_GM_CFG
pub const cfg_gu: i64 = 1;
/// TF_DSV41_GM_GU2: gate / up at x3gm.cu's GU2 (128-member passes, kernels_exl3.gu2_cfg) over a plan of bm2
pub const cfg_gu2: i64 = 2;
pub const bm2: i64 = 128;
/// GU2 runs where passes fill: a 4,096-row segment's block (~64 members an expert)
pub const gu2_rows: i64 = 4096;

fn gu2On(e: *const E, r: Routed) bool {
    _ = r;
    return e.o.gm_gu2 and e.n >= gu2_rows;
}
pub const cfg_dn: i64 = 0;
/// x3gm.K2S: the widths gm2_kernel dispatches (x3gm.supported refuses a block with any other)
pub const k2s: u32 = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 10);

pub const Get = *const fn ([]const u8) ?[]const u8;

/// TF_DSV41_GM_V2 by x3gm.config's rule: `(env or "0").strip()` one of "0", "1", "split", else refused.
pub fn mode(get: Get) !Mode {
    const raw = std.mem.trim(u8, get("TF_DSV41_GM_V2") orelse "", " \t");
    if (raw.len == 0 or std.mem.eql(u8, raw, "0")) return .off;
    if (std.mem.eql(u8, raw, "1")) return .one;
    if (std.mem.eql(u8, raw, "split")) return .split;
    if (!quiet) std.log.warn("TF_DSV41_GM_V2={s}: expected 0, 1 or split", .{raw});
    return error.BadKnob;
}

/// The mode a capture's prefill ran, read from its launches: no `gateup2` is off, one a rotation `one`, more `split`.
pub fn ofLaunches(names: anytype) Mode {
    var rot: usize = 0;
    var gu2: usize = 0;
    while (names.next()) |n| {
        if (std.mem.eql(u8, n, "tf_dsv41_x3gm_v1.rot")) rot += 1;
        if (std.mem.eql(u8, n, "tf_dsv41_x3gm_v1.gateup2")) gu2 += 1;
    }
    if (gu2 == 0) return .off;
    return if (gu2 > rot) .split else .one;
}

/// One MoE call's routed experts after the rotation and the ticket's zeroing (block_prefill.zig's moeRows).
pub const Routed = struct {
    L: u32,
    /// "L<i>.moe": the routed stacks' weight prefix
    pre: []const u8,
    /// routed pairs (rows x topk), routed experts, hidden, this rank's expert width
    P: i64,
    Et: i64,
    D: i64,
    I: i64,
    /// the routed picks [rows, topk] (gm_picks), the rotated inputs, Xd, Y, the ticket [2 x max_widths]
    pk: Arg,
    xg: Arg,
    xu: Arg,
    xd: Arg,
    y: Arg,
    ticket: Arg,
    limit: f64,
    /// gate/up and down widths present (Widths.gm; bit k2): split's launches; `one` checks them when known
    widths: ?[2]u32,
};

/// The calls of x3gm._run_v2's block for `m` (not off).
pub fn emit(e: *E, m: Mode, r: Routed) !void {
    std.debug.assert(m != .off);
    if (r.widths) |w| if ((w[0] | w[1]) & ~k2s != 0 or w[0] == 0 or w[1] == 0) return error.Unsupported; // x3gm.supported: "width"
    const L = r.L;
    const tmax = @divFloor(r.P, bm) + r.Et + 1;
    const tab = [3]Arg{
        try e.buf("s.L{d}.gm.tg", .{L}, .i64, &.{r.Et}),
        try e.buf("s.L{d}.gm.tu", .{L}, .i64, &.{r.Et}),
        try e.buf("s.L{d}.gm.td", .{L}, .i64, &.{r.Et}),
    };
    const k2g = try e.buf("s.L{d}.gm.k2g", .{L}, .i32, &.{r.Et + 1});
    const k2d = try e.buf("s.L{d}.gm.k2d", .{L}, .i32, &.{r.Et + 1});
    if (m == .one) {
        const p = try planOf(e, r, "g", 0, 0, false, tmax, bm);
        if (gu2On(e, r)) {
            // TF_DSV41_GM_GU2: 128-member passes over their own plan (same stable expert order as down's plan)
            const h = try planOf(e, r, "h", 0, 0, false, @divFloor(r.P, bm2) + r.Et + 1, bm2);
            try gateupCfg(e, r, tab, k2g, h, 0, cfg_gu2);
        } else try gateup(e, r, tab, k2g, p, 0);
        return down(e, r, tab[2], k2d, p, 0);
    }
    const w = r.widths orelse return error.NoWidth;
    for (0..2) |dn| {
        var j: i64 = 0;
        for (0..32) |kk| {
            if (w[dn] >> @intCast(kk) & 1 == 0) continue;
            const p = try planOf(e, r, if (dn == 1) "d" else "g", j, @intCast(kk), dn == 1, tmax, bm);
            if (dn == 0) try gateup(e, r, tab, k2g, p, j) else try down(e, r, tab[2], k2d, p, j);
            j += 1;
        }
    }
}

/// x3gm.plan's five tensors of one launch (k2 0: every pick in one plan; else that width's picks first).
fn planOf(e: *E, r: Routed, tag: []const u8, j: i64, k2: i64, dn: bool, tmax: i64, pass: i64) ![5]Arg {
    const p = [5]Arg{
        try e.buf("L.gm.{s}{d}.order", .{ tag, j }, .i32, &.{r.P}),
        try e.buf("L.gm.{s}{d}.pe", .{ tag, j }, .i32, &.{tmax}),
        try e.buf("L.gm.{s}{d}.poff", .{ tag, j }, .i32, &.{tmax}),
        try e.buf("L.gm.{s}{d}.pcnt", .{ tag, j }, .i32, &.{tmax}),
        try e.buf("L.gm.{s}{d}.npass", .{ tag, j }, .i32, &.{1}),
    };
    try e.glue("gm_plan", &.{ r.pk, p[0], p[1], p[2], p[3], p[4], .{ .i = k2 }, .{ .b = dn }, .{ .i = pass }, .{ .i = r.Et } });
    return p;
}

fn ticketAt(e: *E, r: Routed, i: i64) !Arg {
    return e.view(r.ticket.t.role, .i32, &.{1}, &.{1}, i * 4);
}

/// x3gm.cpp gateup2 (inputs, tables, plan, ticket, scales, xd, K, N, shx, cfg, use_ticket, limit, tables)
fn gateup(e: *E, r: Routed, tab: [3]Arg, k2g: Arg, p: [5]Arg, j: i64) !void {
    return gateupCfg(e, r, tab, k2g, p, j, cfg_gu);
}

fn gateupCfg(e: *E, r: Routed, tab: [3]Arg, k2g: Arg, p: [5]Arg, j: i64, cfg: i64) !void {
    try e.ext("tf_dsv41_x3gm_v1.gateup2", &.{
        r.xg,                                                        r.xu,
        tab[0],                                                      tab[1],
        k2g,                                                         p[0],
        p[1],                                                        p[2],
        p[3],                                                        p[4],
        try ticketAt(e, r, j),                                       try e.weight("{s}.w1.svh", .{r.pre}, .f16, &.{ r.Et, r.I }),
        try e.weight("{s}.w3.svh", .{r.pre}, .f16, &.{ r.Et, r.I }), try e.weight("{s}.w2.suh", .{r.pre}, .f16, &.{ r.Et, r.I }),
        r.xd,                                                        .{ .i = r.D },
        .{ .i = r.I },                                               .{ .b = false },
        .{ .i = cfg },                                               .{ .b = true },
        .{ .f = r.limit },                                           .{ .b = true },
    });
}

/// x3gm.cpp down2(xd, td, k2e, order, pe, poff, pcnt, npass, ticket, svh_d, y, K, N, cfg, use_ticket, tables)
fn down(e: *E, r: Routed, td: Arg, k2d: Arg, p: [5]Arg, j: i64) !void {
    try e.ext("tf_dsv41_x3gm_v1.down2", &.{
        r.xd,                               td,
        k2d,                                p[0],
        p[1],                               p[2],
        p[3],                               p[4],
        try ticketAt(e, r, max_widths + j), try e.weight("{s}.w2.svh", .{r.pre}, .f16, &.{ r.Et, r.D }),
        r.y,                                .{ .i = r.I },
        .{ .i = r.D },                      .{ .i = cfg_dn },
        .{ .b = true },                     .{ .b = true },
    });
}
