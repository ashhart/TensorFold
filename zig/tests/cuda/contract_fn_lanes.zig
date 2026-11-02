//! Flash Next's page-table and one-launch kernels in the contract run.
const check = @import("check.zig");

pub fn run(gpu: check.Gpu) !void {
    try @import("contract_fn_pages.zig").run(gpu);
    try @import("contract_fn_attn_lanes.zig").run(gpu);
}
