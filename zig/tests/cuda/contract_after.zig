//! kernel-contract's words (`only`): the kernels so named, and the suites that run after the gates on a full run.
const std = @import("std");
const g = @import("core").contract.gates;
const check = @import("check.zig");

/// Keeps the kernels whose name holds one of `only`'s words; empty keeps every one.
pub fn keep(kernels: *std.ArrayList(g.Kernel), only: []const []const u8) void {
    if (only.len == 0) return;
    var kept: usize = 0;
    for (kernels.items) |k| for (only) |w| if (std.mem.indexOf(u8, k.spec.name, w) != null) {
        kernels.items[kept] = k;
        kept += 1;
        break;
    };
    kernels.shrinkRetainingCapacity(kept);
}

/// The served, paged, pending, learning and lane suites of a full run.
pub fn suites(gpu: check.Gpu) !void {
    try @import("contract_qwen_served.zig").run(gpu);
    try @import("contract_qwen_pages.zig").run(gpu);
    try @import("contract_qwen_pending.zig").run(gpu);
    try @import("contract_qwen_learning.zig").numerical(gpu);
    try @import("contract_fn_lanes.zig").run(gpu);
}
