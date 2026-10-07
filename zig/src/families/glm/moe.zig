//! GLM-5.3-Flash's MoE block on a decode window: shared expert, route, routed experts, combine.
const std = @import("std");
const mtl = @import("metal");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const moe_route = @import("../../core/moe_route.zig");
const route_shape = @import("kernels.zig").route_shape;
const Ref = wts.Ref;
const Ctx = fwd.Ctx;
const Class = fwd.Class;
const on = fwd.on;
const bind = fwd.bind;
const shape = fwd.shape;
const size = fwd.size;

/// The MoE block (shared expert, route, routed experts, combine); expert parallel: this Mac's experts, sent, the peer's received.
pub fn moe(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    std.debug.assert(rows <= st.max_rows);
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const top = c.topk;
    const s = x.skip;
    if (x.ep) |ep| if (c.byRows()) { // every expert's half on each Mac: both compute every pick, then swap their sums
        if (s & Class.exchange == 0 and on(x, "x_locpost")) ep.begin();
        if (s & Class.route == 0) route(x, e, w, x_in, rows);
        if (s & Class.routed == 0) halfExperts(x, e, w, x_in, rows);
        if (s & Class.exchange == 0 and on(x, "x_locpost")) ep.sendRows(e, sc.yp, sc.wts, rows); // with its begin
        if (s & Class.shared == 0) experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        if (s & Class.combine == 0) ep.receiveRows(e, sc.ys, sc.branch, rows);
        return;
    };
    if (x.ep) |ep| {
        if (s & Class.exchange == 0 and on(x, "x_locpost")) ep.begin();
        if (s & Class.route == 0) route(x, e, w, x_in, rows);
        if (!x.fused_route and s & Class.exchange == 0 and on(x, "x_locpost")) ep.localize(e, sc.pick, sc.uids, sc.umem, sc.ucount, rows);
        if (s & Class.routed == 0) experts(x, e, w, x_in, rows, 2, ep.group());
        if (s & Class.exchange == 0) ep.send(e, sc.ye, rows, on(x, "x_pack") and on(x, "x_locpost"), on(x, "x_locpost"));
        if (s & Class.shared == 0) experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        if (s & Class.exchange == 0 and on(x, "x_unpack")) ep.receive(e, sc.ye, rows);
    } else {
        if (s & Class.shared == 0) experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        if (s & Class.route == 0) route(x, e, w, x_in, rows);
        if (s & Class.routed == 0) experts(x, e, w, x_in, rows, 2, .{ sc.uids, sc.umem, sc.ucount });
    }
    if (s & Class.combine != 0 or !on(x, "combine")) return;
    e.setPipeline(k.moe_combine);
    bind(e, 0, .{ sc.ys, sc.ye, sc.wts });
    shape(e, 3, .{ rows, top });
    bind(e, 4, .{sc.branch});
    e.dispatchThreads(size(rows * c.hidden, 1, 1), size(256, 1, 1));
}

/// The router's fp32 logits and the route: picks, weights and the window's unique experts with their members.
fn route(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    if (x.fused_route) {
        if (on(x, "r_router")) moe_route.logits(e, k.route_logits, route_shape, x_in, w.router, sc.logits_r, rows);
        if (!on(x, "r_topk")) return;
        if (x.ep) |ep| if (!c.byRows()) return moe_route.select(e, k.route_select, sc.logits_r, w.bias, c.routed_scale, rows, c.own, ep.outputs(sc.pick, sc.wts));
        const rl = sc.rl;
        const T: usize = c.topk * st.max_rows * 4;
        return moe_route.select(e, k.route_select, sc.logits_r, w.bias, c.routed_scale, rows, .{ 0, c.experts }, .{ .pick = sc.pick, .wts = sc.wts, .ids = sc.uids, .members = sc.umem, .count = sc.ucount, .mine = rl, .theirs = rl.at(T), .counts = rl.at(2 * T), .word = rl.at(2 * T + 64) });
    }
    if (on(x, "r_cast")) {
        e.setPipeline(k.cast_f32);
        bind(e, 0, .{ x_in, sc.xf });
        e.setValue(rows * c.hidden, 2);
        e.dispatchThreads(size(rows * c.hidden, 1, 1), size(256, 1, 1));
    }
    if (on(x, "r_router")) {
        e.setPipeline(k.router[rows - 1]);
        bind(e, 0, .{ sc.xf, w.router, sc.logits_r });
        e.dispatchThreads(size(1024 * c.experts / 16, 1, 1), size(1024, 1, 1));
    }
    if (!on(x, "r_topk")) return;
    e.setPipeline(k.moe_route);
    bind(e, 0, .{sc.logits_r});
    shape(e, 1, .{ rows, c.experts });
    bind(e, 2, .{w.bias});
    e.setValue(c.routed_scale, 3);
    bind(e, 4, .{ sc.pick, sc.wts, sc.uids, sc.umem, sc.ucount });
    e.dispatchThreads(size(512, 1, 1), size(512, 1, 1));
}

/// Part 1: the shared expert into `ys`; part 2: the routed experts of `group` (unique ids, members, count) into `ye`.
fn experts(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32, part: u32, group: [3]Ref) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const D = c.hidden;
    const N = c.moe_inter;
    const zs: u32 = if (part == 1) 1 else rows * c.topk;
    const slots: u32 = if (part == 1) 1 else c.topk;
    const act = if (part == 1) sc.acts else sc.act;
    if (if (part == 1) on(x, "s_gateup") else on(x, "e_gateup")) gateUp(x, e, w, x_in, rows, part, group, zs, act);
    if (!(if (part == 1) on(x, "s_down") else on(x, "e_down"))) return;
    e.setPipeline(if (part == 1) k.moe_down_1 else k.moe_down_2);
    bind(e, 0, .{act});
    shape(e, 1, .{ rows, slots, N });
    bind(e, 2, .{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b, group[0], group[1], group[2], if (part == 1) sc.ys else sc.ye });
    e.dispatchThreads(size(32 * rows, D / 4, zs), size(32 * rows, 1, 1));
}

/// By rows: every routed pick's half (this Mac's intermediate rows), down's fp32 partials into `yp`.
fn halfExperts(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const D = c.hidden;
    const N: u32 = c.inter[1] - c.inter[0];
    const zs: u32 = rows * c.topk;
    if (on(x, "e_gateup")) {
        e.setPipeline(k.moe_gateup_2h);
        bind(e, 0, .{x_in});
        shape(e, 1, .{ rows, D });
        bind(e, 2, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b, sc.uids, sc.umem, sc.ucount });
        e.setValue(c.swiglu_limit, 14);
        bind(e, 15, .{sc.act});
        e.dispatchThreads(size(32 * rows, N / 4, zs), size(32 * rows, 1, 1));
    }
    if (!on(x, "e_down")) return;
    e.setPipeline(k.moe_down_2h);
    bind(e, 0, .{sc.act});
    shape(e, 1, .{ rows, c.topk, N });
    bind(e, 2, .{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b, sc.uids, sc.umem, sc.ucount, sc.yp });
    e.dispatchThreads(size(32 * rows, D / 4, zs), size(32 * rows, 1, 1));
}

fn gateUp(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32, part: u32, group: [3]Ref, zs: u32, act: Ref) void {
    const c = x.c;
    const k = x.k;
    const D = c.hidden;
    const N = c.moe_inter;
    e.setPipeline(if (part == 1) k.moe_gateup_1 else k.moe_gateup_2);
    bind(e, 0, .{x_in});
    shape(e, 1, .{ rows, D });
    bind(e, 2, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b, group[0], group[1], group[2] });
    e.setValue(c.swiglu_limit, 14);
    bind(e, 15, .{act});
    e.dispatchThreads(size(32 * rows, N / 4, zs), size(32 * rows, 1, 1));
}
