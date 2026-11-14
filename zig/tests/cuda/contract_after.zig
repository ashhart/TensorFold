//! kernel-contract's gates over its words and shard, and the suites that run after them on a full run.
const std = @import("std");
const g = @import("core").contract.gates;
const check = @import("check.zig");
const shard = @import("contract_shard.zig");
const back = @import("contract_back_suites.zig");

/// Dumps if asked, then gates 1-4 on `opt`'s share of the kernels; `t0` is the clock at the first gate.
pub fn gates(gpu: check.Gpu, kernels: *std.ArrayList(g.Kernel), opt: shard.Options, t0: *i96, listed: *usize) !g.Tally {
    var t: g.Tally = .{};
    try @import("contract_dump.zig").run(gpu.gpa, kernels.items);
    t0.* = check.now(gpu.io);
    listed.* = shard.keep(g.Kernel, kernels, opt.words, opt.effectiveShard());
    try g.runAll(gpu.gpa, kernels.items, 0x6b63, &t);
    return t;
}

/// The served, paged, pending, learning and lane suites of a full run.
const served = [_]back.Suite{
    back.entry("qwen-served", @import("contract_qwen_served.zig").run),
    back.entry("qwen-pages", @import("contract_qwen_pages.zig").run),
    back.entry("qwen-pending", @import("contract_qwen_pending.zig").run),
    back.entry("qwen-learning", @import("contract_qwen_learning.zig").numerical),
    back.entry("fn-lanes", @import("contract_fn_lanes.zig").run),
};

pub const Part = enum { back, after };

/// The suites of a full run in order, the gradient ones first; `sh` runs those whose index it owns.
pub fn suites(gpu: check.Gpu, part: Part, sh: shard.Shard, coverage: *shard.Coverage) !void {
    const all = back.list ++ served;
    const from: usize = if (part == .back) 0 else back.list.len;
    const to: usize = if (part == .back) back.list.len else all.len;
    coverage.of += to - from;
    for (all[from..to], from..) |s, i| {
        if (!sh.owns(i)) continue;
        try s.run(gpu);
        coverage.ran += 1;
    }
}
