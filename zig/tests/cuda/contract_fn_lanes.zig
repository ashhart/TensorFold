//! Flash Next's one-launch kernels in the contract run, attention then DeltaNet.
const check = @import("check.zig");

pub fn run(gpu: check.Gpu) !void {
    try @import("contract_fn_attn_lanes.zig").run(gpu);
    try @import("contract_fn_gdn_lanes.zig").run(gpu);
}
