//! The kernel contract's closing phase: the standalone suites, gate 5 and the SUMMARY line.
const std = @import("std");
const core = @import("core");
const check = @import("check.zig");
const g = core.contract.gates;

pub fn run(gpu: check.Gpu, t: g.Tally, t0: i96) !void {
    try @import("contract_back_suites.zig").run(gpu);
    const t1 = check.now(gpu.io);
    try @import("contract_qwen_served.zig").run(gpu);
    try @import("contract_qwen_pages.zig").run(gpu);
    try @import("contract_qwen_learning.zig").numerical(gpu);
    try @import("contract_fn_pages.zig").run(gpu);
    const lim = try @import("contract_limits.zig").live(gpu);
    const t2 = check.now(gpu.io);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}
