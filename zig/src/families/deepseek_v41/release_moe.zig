//! The release MoE as calls (D§5.3): route, FFN-input round trip, MXFP4 routed experts, FP8 shared expert, combine.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const rd = @import("release_dense.zig");
const wide = @import("block_wide.zig");
const partition = @import("partition.zig");
const Arg = calls.Arg;
const Dt = calls.Dt;
const E = block.Emitter;

/// The reference's routing-weight denominator guard: weights are s / (sum(s) + 1e-20), then x routed_scale.
pub const route_eps: f64 = 1e-20;
/// Rows a route launch takes (its 128 row-tile counters at 16 rows a tile).
const route_rows: i64 = 2048;

fn rowsOf(e: *E, x: Arg, dt: Dt, a: i64, rows: i64, cols: i64) !Arg {
    if (a == 0 and rows == x.t.shape[0]) return x;
    return e.view(x.t.role, dt, &.{ rows, cols }, &.{ cols, 1 }, x.t.offset + a * cols * @as(i64, @intCast(@import("buffers.zig").dtSize(dt))));
}

/// The rank's slot of every expert id of a layer with `count` experts (-1 for ids it does not hold), built at boot.
pub fn slotRole(e: *E, count: i64) !Arg {
    return e.buf("s.rel.slot_of{d}", .{count}, .i32, &.{count});
}

/// Block L's MoE over `n` FFN-normed rows `x` into `out` [n, D] (bf16 / fp32), rank-local, before the exchange.
pub fn body(e: *E, L: u32, n: i64, x: Arg, comptime bias: []const u8, out: Arg) !void {
    const D: i64 = e.cfg.hidden;
    const ex = e.cfg.expertsOf(L);
    const topk: i64 = ex.topk;
    const count: i64 = ex.count;
    const held = count; // expert degree 1: the rank holds every expert (an expert group would plan fewer slots)
    const I: i64 = (try partition.expertShare(e.cfg, e.o.rank, e.o.world)).len;
    const P = n * topk;
    // route: ids and FP32 weights [n, topk], no kit rounding and no shared slot (the shared expert runs dense)
    const pick = try e.buf("L.rel.pick", .{}, .i32, &.{ n, topk });
    const wts = try e.buf("L.rel.wts", .{}, .f32, &.{ n, topk });
    const logits = try e.buf("L.logits", .{}, .f32, &.{ n, count });
    const gate = try e.weight("L{d}.moe.gate", .{L}, .bf16, &.{ count, D });
    const b = try e.weight("L{d}.moe." ++ bias, .{L}, .f32, &.{count});
    var c0: i64 = 0;
    while (c0 < n) : (c0 += route_rows) {
        const m = @min(route_rows, n - c0);
        const rc = wide.routerCfg(m);
        try e.ext("tf_dsv41_router_gemv_v2.route", &.{
            try rowsOf(e, x, .bf16, c0, m, D),             gate,                                  b,
            try rowsOf(e, pick, .i32, c0, m, topk),        try rowsOf(e, wts, .f32, c0, m, topk), try rowsOf(e, logits, .f32, c0, m, count),
            try e.buf("s.router.cnt", .{}, .i32, &.{128}), .{ .i = topk },                        .{ .i = topk },
            .{ .f = E.dec(e.cfg.routed_scale) },           .{ .i = rc[0] },                       .{ .i = rc[1] },
            .{ .i = rc[2] },                               .{ .b = true },                        .{ .b = false },
            try e.empty(.i32),                             try e.empty(.i32),                     try e.t(.empty, .i32, &.{ 0, 1 }),
            .{ .i = 0 },                                   .{ .b = false },                       .{ .list = &.{} },
            .{ .f = route_eps },
        });
    }
    // the FFN input's round trip, read by the routed and the shared experts
    const xq = try rd.actq(e, "moe", 0, x, n, D);
    // grouping: pairs (row, k) bucketed by the rank's slot, ascending (slot, row, k)
    const offs = try e.buf("L.rel.offs", .{}, .i32, &.{held + 1});
    const pair_tok = try e.buf("L.rel.pair_tok", .{}, .i32, &.{P});
    const pair_of = try e.buf("L.rel.pair_of", .{}, .i32, &.{ n, topk });
    const active = try e.buf("L.rel.active", .{}, .i32, &.{held + 1});
    const nactive = try e.buf("L.rel.nactive", .{}, .i32, &.{1});
    try e.ext(rd.ext ++ ".group", &.{ pick, try slotRole(e, count), offs, pair_tok, pair_of, active, nactive });
    // gate / up of every pair, the epilogue and the round trip of v: the pairs' w2 operands
    const hq = try e.buf("L.rel.hq", .{}, .u8, &.{ P, I });
    const hs = try e.buf("L.rel.hqs", .{}, .u8, &.{ P, @divExact(I, 32) });
    try e.ext(rd.ext ++ ".gateup", &.{
        xq.codes,                                                                 xq.scales,
        try e.weight("L{d}.moe.w1.w", .{L}, .u8, &.{ held, I, @divExact(D, 2) }), try e.weight("L{d}.moe.w1.s", .{L}, .u8, &.{ held, I, @divExact(D, 32) }),
        try e.weight("L{d}.moe.w3.w", .{L}, .u8, &.{ held, I, @divExact(D, 2) }), try e.weight("L{d}.moe.w3.s", .{L}, .u8, &.{ held, I, @divExact(D, 32) }),
        wts,                                                                      pair_tok,
        offs,                                                                     active,
        nactive,                                                                  hq,
        hs,
    });
    // down: each pair's bf16 partial over the rank's columns of the expert width
    const part = try e.buf("L.rel.ypart", .{}, .bf16, &.{ P, D });
    try e.ext(rd.ext ++ ".down", &.{
        hq,                                                                       hs,
        try e.weight("L{d}.moe.w2.w", .{L}, .u8, &.{ held, D, @divExact(I, 2) }), try e.weight("L{d}.moe.w2.s", .{L}, .u8, &.{ held, D, @divExact(I, 32) }),
        offs,                                                                     active,
        nactive,                                                                  part,
    });
    // the shared expert: w1 / w3 FP8 on the same round trip, SwiGLU and its round trip, w2 (bf16 partial)
    var shared: Arg = try e.empty(.bf16);
    if (e.cfg.shared_experts > 0) {
        const sh = try e.fmt("L{d}.moe.shared.0", .{L});
        const g = try e.buf("L.rel.sg", .{}, .bf16, &.{ n, I });
        const u = try e.buf("L.rel.su", .{}, .bf16, &.{ n, I });
        try rd.project(e, try e.fmt("{s}.w1", .{sh}), .fp8, .{ .fp8 = xq }, n, D, I, g, 0);
        try rd.project(e, try e.fmt("{s}.w3", .{sh}), .fp8, .{ .fp8 = xq }, n, D, I, u, 0);
        const sq = try e.buf("L.rel.sq", .{}, .u8, &.{ n, I });
        const ss = try e.buf("L.rel.sqs", .{}, .u8, &.{ n, @divExact(I, 32) });
        try e.ext(rd.ext ++ ".swiglu", &.{ g, u, sq, ss });
        shared = try e.buf("L.rel.sy", .{}, .bf16, &.{ n, D });
        try rd.project(e, try e.fmt("{s}.w2", .{sh}), .fp8, .{ .fp8 = .{ .codes = sq, .scales = ss } }, n, I, D, shared, 0);
    }
    // per row: its top-k partials in rank order into one FP32 sum, then the shared partial; out bf16 or fp32
    try e.ext(rd.ext ++ ".combine", &.{ part, pair_of, shared, out, .{ .i = 0 } });
}

const testing = std.testing;

test "the release MoE: route without kit or shared slot, one round trip, routed MXFP4 and shared FP8, combine" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: @import("config.zig").Config = .{};
    var w: block.Widths = .{};
    const L: u32 = 3;
    inline for (.{ "w1", "w2", "w3" }) |p| try w.release.put(a, "L3.moe.shared.0." ++ p, .fp8);
    var e: E = .{ .a = a, .cfg = &cfg, .w = &w, .o = .{ .format = .release_fp8, .world = 4 }, .n = 5, .start = 0 };
    const x = try e.buf("w.out", .{}, .bf16, &.{ 5, cfg.hidden });
    const out = try e.buf("L.moe", .{}, .f32, &.{ 5, cfg.hidden });
    try body(&e, L, 5, x, "bias", out);
    var names: std.ArrayList([]const u8) = .empty;
    for (e.out.items) |c| try names.append(a, c.name);
    const want = [_][]const u8{
        "tf_dsv41_router_gemv_v2.route", rd.ext ++ ".actq", rd.ext ++ ".group",  rd.ext ++ ".gateup", rd.ext ++ ".down",
        rd.ext ++ ".fp8",                rd.ext ++ ".fp8",  rd.ext ++ ".swiglu", rd.ext ++ ".fp8",    rd.ext ++ ".combine",
    };
    try testing.expectEqual(want.len, names.items.len);
    for (want, names.items) |x_, y| try testing.expectEqualStrings(x_, y);
    const route = e.out.items[0].args;
    try testing.expectEqual(false, route[14].arg.b); // kit off: FP32 weights
    try testing.expectEqual(@as(i64, 6), route[8].arg.i); // slots = top-k, no shared slot
    try testing.expectEqual(route_eps, route[21].arg.f);
    // the shared experts read the routed experts' round trip, and the combine writes the fp32 L.moe
    try testing.expectEqualStrings("L.moe.q0", e.out.items[5].args[0].arg.t.role.buf);
    try testing.expectEqualStrings("L.moe.q0", e.out.items[3].args[0].arg.t.role.buf);
    try testing.expectEqualStrings("L.moe", e.out.items[9].args[3].arg.t.role.buf);
    try testing.expectEqualStrings("L.rel.sy", e.out.items[9].args[2].arg.t.role.buf);
}
