//! A layer's MoE over an EXL3 pack for one launch: main's gate, route and plan, then the routed and shared experts' trellises.
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const moe = @import("moe.zig");

const Ctx = ctx_.Ctx;
const g = pipelines.glue;
const ex = pipelines.experts;
const f = pipelines.fused;
const lane = pipelines.exl3;
const cv = ctx_.cv;

/// moe.zig's `encodeWith` on the trellises: each expert reads act_quant(x), as the shared FP8 expert does.
pub fn encode(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, m: *const weights.X3, sh: moe.Shape, j: moe.Join, fin: ?f.Mix) !void {
    const c = x.c;
    const s = x.s;
    const R = x.rows;
    const p = if (x.p.x3) |*q| q else return error.Exl3PipelinesMissing;
    const call: ex.Call = .{ .rows = R, .k = c.hidden, .n = c.expert_width, .experts = sh.experts, .topk = sh.topk };
    try ex.check(call);
    const route: ex.Route = .{ .ids = cv(ex.Ref, s.picks), .weights = cv(ex.Ref, s.wts), .slots = cv(ex.Ref, s.plan[0]), .nslots = cv(ex.Ref, s.plan[1]), .pairs = cv(ex.Ref, s.plan[2]), .where = cv(ex.Ref, s.plan[3]) };
    const rt: g.Route = .{ .logits = s.logits, .bias = w.gate_b, .bias_vl = w.gate_bvl orelse w.gate_b, .image = s.image, .picks = s.picks, .weights = s.wts };
    // the gate's fp32 logits of the bf16 weight; the shared w1 and w3 with their clamped SwiGLU and its act_quant
    try g.mv(&x.p.glue, e, s.hn2, w.gate_w, s.logits, R, c.hidden, sh.experts, true);
    try lane.gateUp(p, e, m.m.shared[0], m.m.shared[2], s.hq2, s.sact, s.sactq, R, c.swiglu_limit, s.x3);
    if (fin) |mx| f.mixFinish(&x.p.fused, e, mx, R);
    // sqrt(softplus) + bias, top-k, weights times the route scale; then the routed plan
    g.route(sh.route, e, rt, R, sh.topk, c.routed_scale);
    e.barrier();
    ex.use(e, w.backing());
    ex.plan(&x.p.experts, e, route, call);
    e.barrier();
    // gate/up with the route weight inside, down into experts.zig's layout; the shared w2; the join
    try lane.experts(p, e, s.hq2, m.experts, route, s.act, s.eout, s.x3, call, c.swiglu_limit);
    try x.x3Proj(e, m.m.shared[1], s.sactq, s.sout, false);
    moe.join(x, e, sh, j, call);
}
