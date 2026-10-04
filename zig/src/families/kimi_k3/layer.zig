//! One decoder layer on a round: attention residual + norm, KDA or MLA, attention residual + norm, dense or MoE.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const kernels = @import("kernels.zig");
const weights = @import("weights.zig");
const Scratch = @import("round.zig").Scratch;
const State = @import("state.zig").State;
const experts_mod = @import("experts.zig");
const store = @import("store.zig");
const Ref = kernels.Ref;

pub const Ctx = struct {
    k: *const kernels.Kernels,
    c: *const Config,
    sc: *Scratch,
    st: *const State,
    experts: experts_mod.Experts,
};

fn rows(x: Ctx, e: mtl.ComputeEncoder, in: Ref, w: anytype, out: Ref, K: u32, N: u32, x_stride: u32, y_stride: u32) void {
    x.k.rows(e, in, w.ref, out, x.sc.ref("part"), false, .{ .K = K, .N = N, .rows = x.sc.rows, .x_stride = x_stride, .y_stride = y_stride });
}

/// The residual step before attention (`pre`) or before the MLP, ending in that sublayer's normed input (xin).
pub fn residual(x: Ctx, e: mtl.ComputeEncoder, i: u32, w: *const weights.Layer, pre: bool) void {
    const c = x.c;
    const boundary = i % c.block == 0;
    var flags: u32 = kernels.res_delta | kernels.res_prefix;
    if (pre and i == 0) flags = kernels.res_prefix;
    if (pre and boundary) flags |= kernels.res_append;
    if (!pre and boundary) flags = kernels.res_delta;
    const blocks = if (pre) c.blocksBefore(i) else i / c.block + 1;
    const nw = if (pre) w.sa_res_norm else w.mlp_res_norm;
    const pw = if (pre) w.sa_res_proj else w.mlp_res_proj;
    x.k.resNorm(e, x.sc.ref("prefix"), x.sc.ref("delta"), x.sc.ref("blocks"), nw.ref, pw.ref, (if (pre) w.in_norm else w.post_norm).ref, x.sc.ref("xin"), .{ .D = c.hidden, .rows = x.sc.rows, .blocks = blocks, .flags = flags, .eps = c.eps });
}

pub fn kda(x: Ctx, e: mtl.ComputeEncoder, i: u32, w: *const weights.Kda) void {
    const c = x.c;
    const H = c.hidden;
    const W = c.kdaWidth();
    const sc = x.sc;
    const xin = sc.ref("xin");
    for ([_]store.Tensor{ w.q, w.k, w.v }, 0..) |t, j| rows(x, e, xin, t, sc.ref("proj").at(j * W * 2), H, W, H, 3 * W);
    rows(x, e, xin, w.fa, sc.ref("fa"), H, c.kda_dim, H, c.kda_dim);
    rows(x, e, sc.ref("fa"), w.fb, sc.ref("f"), c.kda_dim, W, c.kda_dim, W);
    rows(x, e, xin, w.b, sc.ref("braw"), H, c.kda_heads, H, c.kda_heads);
    rows(x, e, xin, w.g, sc.ref("g2"), H, W, H, W);
    const kw = kernels.KdaWeights{ .conv_q = kernels.addr(w.conv_q.ref), .conv_k = kernels.addr(w.conv_k.ref), .conv_v = kernels.addr(w.conv_v.ref), .A_log = kernels.addr(w.A_log.ref), .dt_bias = kernels.addr(w.dt_bias.ref), .o_norm = kernels.addr(w.o_norm.ref) };
    x.k.kdaRound(e, sc.ref("proj"), sc.ref("f"), sc.ref("braw"), sc.ref("g2"), sc.ref("y"), sc.ref("kseg"), sc.nseg, x.st.kdaState(i), kw, .{ .heads = c.kda_heads, .head0 = 0, .proj_stride = 3 * W, .gate_stride = W, .out_stride = W, .log_rows = x.st.limits.log_rows, .lower_bound = c.lower_bound, .eps = c.eps });
    rows(x, e, sc.ref("y"), w.o, sc.ref("delta"), W, H, W, H);
}

pub fn mlaArgs(x: Ctx, i: u32) kernels.MlaArgs {
    const c = x.c;
    return .{ .cache = x.st.mlaCache(i), .slot_keys = x.st.limits.max_ctx, .heads = c.mla_heads, .head0 = 0, .rows = x.sc.rows, .q_stride = c.mla_heads * c.qHead(), .kv_stride = c.kv_lora + c.rope, .gate_stride = c.mla_heads * c.v_dim, .out_stride = c.mla_heads * c.v_dim, .eps = c.eps, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.qHead()))) };
}

pub fn mla(x: Ctx, e: mtl.ComputeEncoder, i: u32, w: *const weights.Mla) void {
    const c = x.c;
    const H = c.hidden;
    const sc = x.sc;
    const xin = sc.ref("xin");
    const a = mlaArgs(x, i);
    const qw = c.mla_heads * c.qHead();
    const ow = c.mla_heads * c.v_dim;
    rows(x, e, xin, w.qa, sc.ref("qa"), H, c.q_lora, H, c.q_lora);
    x.k.rmsNorm(e, sc.ref("qa"), w.qa_norm.ref, sc.ref("qan"), sc.rows, .{ .D = c.q_lora, .x_stride = c.q_lora, .y_stride = c.q_lora, .eps = c.eps });
    rows(x, e, sc.ref("qan"), w.qb, sc.ref("q"), c.q_lora, qw, c.q_lora, qw);
    rows(x, e, xin, w.kva, sc.ref("kv"), H, c.kv_lora + c.rope, H, c.kv_lora + c.rope);
    x.k.mlaCache(e, sc.ref("kv"), w.kva_norm.ref, sc.ref("mrows"), a);
    x.k.mlaQlat(e, sc.ref("q"), w.kvb.ref, sc.ref("qlat"), a);
    x.k.mlaAttend(e, sc.ref("q"), sc.ref("qlat"), sc.ref("mrows"), sc.ref("partial"), a);
    x.k.mlaMerge(e, sc.ref("partial"), sc.ref("mrows"), sc.ref("olat"), a);
    rows(x, e, xin, w.g, sc.ref("g2"), H, ow, H, ow);
    x.k.mlaUv(e, sc.ref("olat"), w.kvb.ref, sc.ref("g2"), sc.ref("y"), a);
    rows(x, e, sc.ref("y"), w.o, sc.ref("delta"), ow, H, ow, H);
}

pub fn dense(x: Ctx, e: mtl.ComputeEncoder, w: *const weights.Dense) void {
    const c = x.c;
    const sc = x.sc;
    x.k.glu(e, sc.ref("xin"), w.gate.ref, w.up.ref, sc.ref("dact"), sc.ref("part"), .{ .K = c.hidden, .N = c.dense_inter, .rows = sc.rows, .x_stride = c.hidden, .y_stride = c.dense_inter, .beta = c.situ_beta, .lin = c.situ_linear });
    rows(x, e, sc.ref("dact"), w.down, sc.ref("delta"), c.dense_inter, c.hidden, c.dense_inter, c.hidden);
}

/// Latent MoE: router, the routed experts through `x.experts` (local or remote), norm and up-projection, shared GLU.
pub fn moe(x: Ctx, e: mtl.ComputeEncoder, w: *const weights.Moe) !void {
    const c = x.c;
    const H = c.hidden;
    const sc = x.sc;
    const xin = sc.ref("xin");
    x.k.rows(e, xin, w.router.ref, sc.ref("rlogits"), sc.ref("part"), true, .{ .K = H, .N = c.experts, .rows = sc.rows, .x_stride = H, .y_stride = c.experts });
    x.k.routeTopk(e, sc.ref("rlogits"), w.bias.ref, sc.ref("tids"), sc.ref("tw"), sc.rows, .{ .experts = c.experts, .topk = c.topk });
    rows(x, e, xin, w.down, sc.ref("lat"), H, c.latent, H, c.latent);
    try x.experts.run(e, w.table, sc);
    x.k.rmsNorm(e, sc.ref("ysum"), w.norm.ref, sc.ref("ynorm"), sc.rows, .{ .D = c.latent, .x_stride = c.latent, .y_stride = c.latent, .eps = c.eps });
    rows(x, e, sc.ref("ynorm"), w.up, sc.ref("up"), c.latent, H, c.latent, H);
    x.k.glu(e, xin, w.sh_gate.ref, w.sh_up.ref, sc.ref("sact"), sc.ref("part"), .{ .K = H, .N = c.shared_inter, .rows = sc.rows, .x_stride = H, .y_stride = c.shared_inter, .beta = c.situ_beta, .lin = c.situ_linear });
    rows(x, e, sc.ref("sact"), w.sh_down, sc.ref("sh"), c.shared_inter, H, c.shared_inter, H);
    x.k.add(e, sc.ref("up"), sc.ref("sh"), sc.ref("delta"), sc.rows * H);
}

/// Layer i on the round: prefix, blocks and delta in the scratch carry the residual stream between layers.
pub fn encode(x: Ctx, e: mtl.ComputeEncoder, i: u32, w: *const weights.Layer) !void {
    residual(x, e, i, w, true);
    switch (w.attn) {
        .kda => |*a| kda(x, e, i, a),
        .mla => |*a| mla(x, e, i, a),
    }
    residual(x, e, i, w, false);
    switch (w.mlp) {
        .dense => |*d| dense(x, e, d),
        .moe => |*m| try moe(x, e, m),
    }
}

/// After the last layer: the output attention residual, the final norm and the LM head's bf16 logits.
pub fn head(x: Ctx, e: mtl.ComputeEncoder, h: *const weights.Head, layers: u32) void {
    const c = x.c;
    const sc = x.sc;
    x.k.resNorm(e, sc.ref("prefix"), sc.ref("delta"), sc.ref("blocks"), h.res_norm.ref, h.res_proj.ref, h.norm.ref, sc.ref("xin"), .{ .D = c.hidden, .rows = sc.rows, .blocks = (layers - 1) / c.block + 1, .flags = kernels.res_delta | kernels.res_prefix, .eps = c.eps });
    x.k.rows(e, sc.ref("xin"), h.lm_head.ref, sc.ref("logits"), sc.ref("part"), false, .{ .K = c.hidden, .N = c.vocab, .rows = sc.rows, .x_stride = c.hidden, .y_stride = c.vocab });
}

test {
    std.testing.refAllDecls(@This());
}
