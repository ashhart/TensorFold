//! The MoE layer the HIP forward calls: this file is its interface; the kernels behind it plug in here.
const hip = @import("hip");
const model = @import("model.zig");

const Tensor = hip.ops.Tensor;

pub const Error = hip.ops.Error;

/// Whether `layer` runs: the native server refuses Nemotron until it does.
pub const ready = true;

/// layer: x (rows, hidden) act normed rows, any streams' rows together -> y (rows, hidden).
pub fn layer(o: hip.ops.Ops, d: model.Dims, w: model.Moe, x: Tensor, rows: usize) Error!Tensor {
    // y in act on one rank; fp32 when d.world > 1 (this rank's experts' share, which the caller sums)
    return hip.moe_nemotron.run(o, layerOf(d, w), x, rows);
}

/// One rank's layer before its sum: each row's experts' rows and weights, and the shared expert's row.
pub fn expertRows(o: hip.ops.Ops, d: model.Dims, w: model.Moe, x: Tensor, rows: usize) Error!hip.ops.Ops.MoeRows {
    return hip.moe_nemotron.expertRows(o, layerOf(d, w), x, rows);
}

fn layerOf(d: model.Dims, w: model.Moe) hip.moe_nemotron.Layer {
    return .{
        .router = w.router,
        .bias = w.bias,
        .fc1 = w.fc1,
        .fc2 = w.fc2,
        .shared_up = w.shared_up,
        .shared_down = w.shared_down,
        .experts = d.experts,
        .first = d.first_expert,
        .own = d.rank_experts,
        .top_k = d.top_k,
        .scaling = d.routed_scaling,
        .norm = d.norm_topk,
        .share = d.world > 1,
    };
}
