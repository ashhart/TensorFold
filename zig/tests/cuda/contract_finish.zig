//! The end of tf-cuda-test kernel-contract: the standalone suites and gate 5, once the registered fixtures are gone.
const std = @import("std");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const live = @import("contract_limits.zig").live;
const after = @import("contract_after.zig");
const shard = @import("contract_shard.zig");

/// `listed`: the kernels and controls the words named, of which `t` is this shard's share.
pub fn run(gpu: check.Gpu, opt: shard.Options, t: g.Tally, listed: usize, t0: i96) !void {
    const sh = opt.effectiveShard();
    const full = opt.words.len == 0;
    var suites: shard.Coverage = .{};
    if (full) try after.suites(gpu, .back, sh, &suites);
    if (shard.printsGradientPass(if (opt.shard == null) .serial else .shard, !full)) std.debug.print("{s}", .{@import("contract_back_suites.zig").passed});
    const t1 = check.now(gpu.io);
    if (full) try after.suites(gpu, .after, sh, &suites);
    var records: shard.Coverage = .{};
    const lim = try live(gpu, opt.words, sh, &records);
    const t2 = check.now(gpu.io);
    const gates_ms: u64 = @intCast(@divTrunc(t1 - t0, std.time.ns_per_ms));
    const limits_ms: u64 = @intCast(@divTrunc(t2 - t1, std.time.ns_per_ms));
    const sum: shard.Summary = .{ .filtered = !full, .tally = t, .limits = lim, .gates_ms = gates_ms, .limits_ms = limits_ms };
    if (opt.shard) |s| {
        const report: shard.Report = .{ .shard = s, .listed = listed, .tally = t, .limits = lim, .suites = suites, .records = records, .gates_ms = gates_ms, .limits_ms = limits_ms };
        std.debug.print("{f}\n", .{report});
    } else std.debug.print("{f}\n", .{sum});
    try check.expect(sum.problems() == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}
