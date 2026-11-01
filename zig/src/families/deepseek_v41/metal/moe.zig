//! A layer's MoE for one launch: the route, the routed MXFP4 experts, the shared FP8 expert and their join.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");

const Ctx = ctx_.Ctx;
const g = pipelines.glue;
const ex = pipelines.experts;
const f = pipelines.fused;
const cv = ctx_.cv;

/// The routed experts a block has: their count, the experts a token takes and the route's kernel at those sizes.
pub const Shape = struct { experts: u32, topk: u32, route: mtl.Pipeline };

/// Where the MoE's output goes: hc_post of it with the residual streams `res` by `post` and `comb` into `out`.
pub const Join = struct { res: ctx_.Ref, post: ctx_.Ref, comb: ctx_.Ref, out: ctx_.Ref };

/// hn2 and hq2 [rows][hidden] through the routed and shared experts, joined and hc_post'ed into `j.out`.
pub fn encode(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, j: Join) !void {
    return encodeWith(x, e, w, .{ .experts = x.c.experts, .topk = x.c.topk, .route = x.p.route }, j, null);
}

/// The same for a block of another shape (a DSpark stage); `fin`: hc mixes whose sums ran beside the first wave.
pub fn encodeWith(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, sh: Shape, j: Join, fin: ?f.Mix) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const R = x.rows;
    const call: ex.Call = .{ .rows = R, .k = c.hidden, .n = c.expert_width, .experts = sh.experts, .topk = sh.topk };
    try ex.check(call);
    // the gate's fp32 logits of the bf16 weight beside the shared expert's FP8 w1 and w3 on act_quant(x)
    g.mv(gp, e, s.hn2, w.gate_w, s.logits, R, c.hidden, sh.experts, true);
    try sharedUp(x, e, w, s.hq2);
    e.barrier();
    if (fin) |m| f.mixFinish(&x.p.fused, e, m, R);
    // sqrt(softplus) + bias, top-k, weights times the route scale; the shared clamped SwiGLU and its act_quant
    g.route(sh.route, e, .{ .logits = s.logits, .bias = w.gate_b, .bias_vl = w.gate_bvl orelse w.gate_b, .image = s.image, .picks = s.picks, .weights = s.wts }, R, sh.topk, c.routed_scale);
    f.swiglu(&x.p.fused, e, cv(f.Ref, s.sg), cv(f.Ref, s.su), cv(f.Ref, s.sact), cv(f.Ref, s.sactq), R * c.expert_width, c.swiglu_limit);
    e.barrier();
    // the routed plan and act_quant beside the shared FP8 w2; then gate/up with the route weight inside, down, the sum
    ex.use(e, w.backing());
    const route: ex.Route = .{ .ids = cv(ex.Ref, s.picks), .weights = cv(ex.Ref, s.wts), .slots = cv(ex.Ref, s.plan[0]), .nslots = cv(ex.Ref, s.plan[1]), .pairs = cv(ex.Ref, s.plan[2]), .where = cv(ex.Ref, s.plan[3]) };
    const table: ex.Ref = .{ .buf = w.table };
    ex.plan(&x.p.experts, e, route, call);
    ex.quant(&x.p.experts, e, cv(ex.Ref, s.hn2), cv(ex.Ref, s.xq), cv(ex.Ref, s.xqs), call);
    try x.proj(e, s.sactq, w.shared[1], s.sout, false);
    e.barrier();
    ex.upCoded(&x.p.experts, e, cv(ex.Ref, s.xq), cv(ex.Ref, s.xqs), table, route, cv(ex.Ref, s.act), call);
    e.barrier();
    ex.down(&x.p.experts, e, cv(ex.Ref, s.act), table, route, cv(ex.Ref, s.eout), call);
    e.barrier();
    // Kimi's combine, the join with the shared expert and hc_post in one launch
    const o = pipelines.fold;
    const r: o.Routed = .{ .out = cv(o.Ref, s.eout), .where = cv(o.Ref, s.plan[3]), .weights = cv(o.Ref, s.ones), .ids = cv(o.Ref, s.picks), .sum = cv(o.Ref, s.yr), .topk = sh.topk, .per_group = ex.perGroup(call) };
    o.combpost(&x.p.fold, e, r, cv(o.Ref, s.sout), cv(o.Ref, s.mout), cv(o.Ref, j.res), cv(o.Ref, j.post), cv(o.Ref, j.comb), cv(o.Ref, j.out), R, c.hidden);
}

/// The shared expert's w1 and w3 on one input: one dispatch of two launches when both live in the same buffers.
fn sharedUp(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, in: ctx_.Ref) !void {
    const s = x.s;
    const w1 = w.shared[0];
    const w3 = w.shared[2];
    if (w1.w.buf.id != w3.w.buf.id or w1.s.buf.id != w3.s.buf.id) {
        try x.proj(e, in, w1, s.sg, false);
        return x.proj(e, in, w3, s.su, false);
    }
    const d = struct {
        fn of(a: ctx_.Ref, b: ctx_.Ref) i64 {
            return @as(i64, @intCast(b.off)) - @as(i64, @intCast(a.off));
        }
    };
    try x.projBatch(e, in, w1, s.sg, false, .{ .n = 2, .ws = d.of(w1.w, w3.w), .ss = d.of(w1.s, w3.s), .ys = d.of(s.sg, s.su) });
}
