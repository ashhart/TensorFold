//! DeepSeek-V4.1-Flash's MoE: routing, the routed experts and the shared expert.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Enc = fwd.Enc;
const Buffer = mtl.Buffer;
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const pair = @import("route_pair.zig");
const FastRouter = @import("router_live.zig");

const RouteArgs = extern struct { experts: i32, topk: i32, scale: f32, trace_layer: i32 };
const GluArgs = extern struct { width: i32, weighted: i32, limit: f32 };
const SortArgs = extern struct { slots: i32, experts: i32 };

/// Routed experts (row kernels by id) and the shared expert, summed in expert order as the reference does.
pub fn moe(x: Enc, mo: *const wts.Moe, topk: usize, experts: usize, boundary: ?fwd.HcJob) !void {
    const s = x.s;
    const rows = x.rows;
    const slots = rows * topk;
    const side = !x.units; // decode-sized: the shared expert runs beside the routed path
    const fast_router = if (x.segs.len == 1 and rows >= 1 and rows <= FastRouter.max_rows and
        experts == c.experts and topk == c.topk and mo.router.offset % 8 == 0 and x.route_trace_layer < 0 and
        fwd.gemv_sgs == 8) x.m.router_variant else null;
    if (fast_router) |p| {
        if (p.audit != null) {
            x.gemv(true, mo.router, s.normed, 0, c.hidden, s.router_ref_logits, 0, rows, experts, c.hidden, experts);
            x.with();
        }
        x.fence();
        p.issue(x.ep.*, mo.router, s.normed, s.router, rows);
    } else if (rows >= fwd.nax_rows) {
        router(x, mo.router, rows, experts);
    } else x.gemv(true, mo.router, s.normed, 0, c.hidden, s.router, 0, rows, experts, c.hidden, experts);
    x.completeBoundary(boundary);
    if (side) {
        x.with();
        x.linear(mo.shared.w1, s.normed, 0, s.sh_gate, 0, rows);
        x.with();
        x.linear(mo.shared.w3, s.normed, 0, s.sh_up, 0, rows);
    }
    const audit_router = if (fast_router) |p| p.audit else null;
    if (audit_router != null) {
        routeInto(x, mo, topk, experts, s.router_ref_logits, s.router_ref_ids, s.router_ref_weights, null);
        x.with();
    }
    routeInto(x, mo, topk, experts, s.router, s.route_pick, s.route_weight, if (fast_router) |p| p.selector else null);
    if (audit_router) |p| {
        x.fence();
        FastRouter.Kernels.compare(p, x.ep.*, s, rows);
        x.segs[0].cache.router_variant_checked += rows * (experts + 2 * topk);
    }
    if (side) {
        x.with();
        gluTo(x, rows, false, s.route_weight, s.sh_gate, s.sh_up, s.sh_act);
    }
    if (x.nax(mo.w1.one)) |p| {
        // a prompt chunk: slots sorted by expert, each expert's rows through its weights once on the tensor units
        const blocks = (slots + st.sort_block - 1) / st.sort_block;
        const sort = SortArgs{ .slots = @intCast(slots), .experts = @intCast(experts) };
        std.debug.assert(experts <= 1024);
        x.pipe("ds_sort_count");
        x.buf(s.route_pick, 0, 0);
        x.buf(s.counts, 0, 1);
        x.args(sort, 2);
        x.groups(.{ blocks, 1, 1 }, .{ st.sort_block, 1, 1 });
        x.pipe("ds_sort_scan");
        x.buf(s.counts, 0, 0);
        x.buf(s.offs, 0, 1);
        x.args(sort, 2);
        x.groups(.{ 1, 1, 1 }, .{ 1024, 1, 1 });
        x.pipe("ds_sort_place");
        x.buf(s.route_pick, 0, 0);
        x.buf(s.route_weight, 0, 1);
        x.buf(s.counts, 0, 2);
        x.buf(s.order, 0, 3);
        x.buf(s.wsort, 0, 4);
        x.args(sort, 5);
        x.groups(.{ blocks, 1, 1 }, .{ st.sort_block, 1, 1 });
        const e = x.ep.*;
        const sums = fwd.Ref{ .buf = s.sums };
        const offs = fwd.Ref{ .buf = s.offs };
        const order = fwd.Ref{ .buf = s.order };
        x.fence();
        const normed = fwd.Ref{ .buf = s.normed };
        const act = fwd.Ref{ .buf = s.act };
        am.rowSums(e, p, 64, normed, sums, @intCast(rows), c.hidden);
        x.fence();
        e.setBuffer(s.wsort, 0, 14);
        am.gatherGlu(e, p, normed, sums, fwd.matrix(mo.w1.one), fwd.matrix(mo.w3.one), offs, order, act, @intCast(slots), @intCast(experts), @intCast(topk), c.swiglu_limit);
        x.fence();
        am.rowSums(e, p, 64, act, sums, @intCast(slots), c.moe_inter);
        x.fence();
        am.gatherTo(e, p, act, sums, fwd.matrix(mo.w2.one), offs, fwd.Ref{ .buf = s.experts }, order, @intCast(slots), @intCast(experts));
    } else {
        const small_window = x.segs.len == 1 and rows > 1 and rows <= 1 + c.draft_block;
        const paired = if (small_window) x.m.kernels.route_pair else null;
        const scheduled = if (paired) |p| p.inline_cap == 0 else false;
        if (scheduled) {
            x.fence();
            paired.?.encodePlan(x.ep.*, s.route_pick, s.route_mates, slots);
        }
        if (side and scheduled) {
            x.with();
            x.linear(mo.shared.w2, s.sh_act, 0, s.shared, 0, rows);
        }
        expertProjection(x, paired, mo.w1, slots, s.normed, s.gate);
        x.with();
        expertProjection(x, paired, mo.w3, slots, s.normed, s.up);
        if (side and !scheduled) {
            x.with();
            x.linear(mo.shared.w2, s.sh_act, 0, s.shared, 0, rows);
        }
        glu(x, slots, true);
        expertProjection(x, paired, mo.w2, slots, s.act, s.experts);
    }
    if (!side) {
        x.linear(mo.shared.w1, s.normed, 0, s.gate, 0, rows);
        x.linear(mo.shared.w3, s.normed, 0, s.up, 0, rows);
        glu(x, rows, false);
        x.linear(mo.shared.w2, s.act, 0, s.shared, 0, rows);
    }
    // the branch, routed rows plus the shared one, is summed where the expand that follows reads it
    fwd.pending_topk = @intCast(topk);
}

fn expertProjection(x: Enc, paired: ?pair.Kernels, wt: row.Experts, slots: usize, input: Buffer, output: Buffer) void {
    const s = x.s;
    if (paired) |p| if (p.choice(&x.m.kernels.rows, wt, slots)) |call| {
        x.fence();
        pair.encode(x.ep.*, call, wt.one, input, s.route_pick, if (p.inline_cap == 0) s.route_mates else null, output);
        return;
    };
    x.indexed(wt, slots, input, s.route_pick, output);
}

fn routeInto(x: Enc, mo: *const wts.Moe, topk: usize, experts: usize, logits: Buffer, ids: Buffer, weights: Buffer, pipeline: ?@import("metal").Pipeline) void {
    const s = x.s;
    x.pipe("ds_route");
    if (pipeline) |p| x.ep.*.setPipeline(p);
    x.buf(logits, 0, 0);
    x.tensor(mo.bias, 1);
    x.buf(ids, 0, 2);
    x.buf(weights, 0, 3);
    x.args(RouteArgs{ .experts = @intCast(experts), .topk = @intCast(topk), .scale = c.route_scale, .trace_layer = x.route_trace_layer }, 4);
    x.buf(s.img, 0, 5);
    x.tensor(mo.bias_vl, 6);
    x.buf(s.route_trace, 0, 7);
    x.groups(.{ x.rows, 1, 1 }, .{ (experts + 31) / 32 * 32, 1, 1 });
}

/// A chunk's router logits: `gemv`'s sums, 4 outputs a simdgroup so each input is read once for them.
fn router(x: Enc, w: wts.Tensor, rows: usize, experts: usize) void {
    x.pipe("ds_gemv_cols_bf_f32");
    x.buf(x.s.normed, 0, 0);
    x.tensor(w, 1);
    x.buf(x.s.router, 0, 2);
    x.args(fwd.GemvArgs{ .rows = @intCast(rows), .n = @intCast(experts), .k = c.hidden, .group_n = @intCast(experts), .x_stride = c.hidden, .y_stride = @intCast(experts) }, 3);
    x.groups(.{ (experts + 31) / 32, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });
}

pub fn glu(x: Enc, n: usize, weighted: bool) void {
    gluTo(x, n, weighted, x.s.route_weight, x.s.gate, x.s.up, x.s.act);
}

pub fn gluTo(x: Enc, n: usize, weighted: bool, weights: Buffer, gate: Buffer, up: Buffer, act: Buffer) void {
    x.pipe("ds_swiglu");
    x.buf(gate, 0, 0);
    x.buf(up, 0, 1);
    x.buf(weights, 0, 2);
    x.buf(act, 0, 3);
    x.args(GluArgs{ .width = c.moe_inter, .weighted = @intFromBool(weighted), .limit = c.swiglu_limit }, 4);
    x.threads(.{ c.moe_inter, n, 1 }, .{ 256, 1, 1 });
}
