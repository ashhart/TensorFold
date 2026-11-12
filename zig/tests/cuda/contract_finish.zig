//! The end of tf-cuda-test kernel-contract: the standalone suites and gate 5, once the registered fixtures are gone.
const std = @import("std");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const live = @import("contract_limits.zig").live;

pub fn run(gpu: check.Gpu, only: []const []const u8, t: g.Tally, t0: i96) !void {
    if (only.len == 0) try @import("contract_back_suites.zig").run(gpu);
    const t1 = check.now(gpu.io);
    if (only.len == 0) try @import("contract_after.zig").suites(gpu);
    const lim = try live(gpu, only);
    const t2 = check.now(gpu.io);
    std.debug.print("{s}SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ if (only.len > 0) "(filtered) " else "", t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}
