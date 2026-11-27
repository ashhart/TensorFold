//! kernel-contract's words (`only`): the kernels so named, and the suites that run after the gates on a full run.
const std = @import("std");
const g = @import("core").contract.gates;
const check = @import("check.zig");

/// Whether `only` names kernels to keep: without plain words (none, or only `-word`s) the run is a full one.
pub fn filtered(only: []const []const u8) bool {
    for (only) |w| if (!std.mem.startsWith(u8, w, "-")) return true;
    return false;
}

/// Whether `name` holds no `-word` and, when plain words are given, one of them.
pub fn wanted(name: []const u8, only: []const []const u8) bool {
    var named = !filtered(only);
    for (only) |w| {
        const minus = std.mem.startsWith(u8, w, "-");
        const hit = std.mem.indexOf(u8, name, if (minus) w[1..] else w) != null;
        if (minus and hit) return false;
        named = named or hit;
    }
    return named;
}

/// Keeps the kernels `wanted` keeps; empty keeps every one.
pub fn keep(kernels: *std.ArrayList(g.Kernel), only: []const []const u8) void {
    var kept: usize = 0;
    for (kernels.items) |k| if (wanted(k.spec.name, only)) {
        kernels.items[kept] = k;
        kept += 1;
    };
    kernels.shrinkRetainingCapacity(kept);
}

/// The served, paged, pending, learning, lane and SwiGLU pair suites of a full run.
pub fn suites(gpu: check.Gpu) !void {
    try @import("tfq3_tests.zig").run(gpu);
    try @import("contract_qwen_served.zig").run(gpu);
    try @import("contract_qwen_pages.zig").run(gpu);
    try @import("contract_qwen_shared_prefix.zig").run(gpu, &.{});
    try @import("contract_qwen_pending.zig").run(gpu);
    try @import("contract_qwen_learning.zig").numerical(gpu);
    try @import("contract_fn_lanes.zig").run(gpu);
    try @import("contract_prompt_swiglu.zig").run(gpu);
}
