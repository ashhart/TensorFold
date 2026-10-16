//! The MoE layer the HIP forward calls: this file is its interface; the kernels behind it plug in here.
const hip = @import("hip");
const model = @import("model.zig");

const Tensor = hip.ops.Tensor;

pub const Error = hip.ops.Error;

/// Whether `layer` runs: the native server refuses Nemotron until it does.
pub const ready = true;

/// layer: x (rows, hidden) act normed rows, any streams' rows together -> y (rows, hidden).
pub fn layer(o: hip.ops.Ops, d: model.Dims, act: model.Kind, w: model.Moe, x: Tensor, rows: usize) Error!Tensor {
    _ = act;
    // y in act on one rank; fp32 when d.world > 1 (this rank's experts' share, which the caller sums)
    return hip.moe_nemotron.run(
        o,
        .{
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
        },
        x,
        rows,
    );
}
