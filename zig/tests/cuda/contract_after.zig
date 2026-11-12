//! kernel-contract's gates over its words and shard, and the suites that run after them on a full run.
const std = @import("std");
const g = @import("core").contract.gates;
const check = @import("check.zig");
const shard = @import("contract_shard.zig");
const back = @import("contract_back_suites.zig");

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

/// Dumps if asked, then gates 1-4 on `opt`'s share of the kernels; `t0` is the clock at the first gate.
pub fn gates(gpu: check.Gpu, kernels: *std.ArrayList(g.Kernel), opt: shard.Options, t0: *i96, listed: *usize) !g.Tally {
    var t: g.Tally = .{};
    try @import("contract_dump.zig").run(gpu.gpa, kernels.items);
    t0.* = check.now(gpu.io);
    keep(kernels, opt.words);
    listed.* = shard.keep(g.Kernel, kernels, &.{}, opt.effectiveShard());
    try g.runAll(gpu.gpa, kernels.items, 0x6b63, &t);
    return t;
}

fn sharedPrefix(gpu: check.Gpu) !void {
    return @import("contract_qwen_shared_prefix.zig").run(gpu, &.{});
}

fn int4Skipped(gpu: check.Gpu) !void {
    return @import("contract_int4_run.zig").skippedAlone(gpu);
}

/// The standalone suites of a full run, in serial order.
const served = [_]back.Suite{
    back.entry("tfq3", @import("tfq3_tests.zig").run),
    back.entry("qwen-served", @import("contract_qwen_served.zig").run),
    back.entry("qwen-pages", @import("contract_qwen_pages.zig").run),
    back.entry("qwen-shared-prefix", sharedPrefix),
    back.entry("qwen-pending", @import("contract_qwen_pending.zig").run),
    back.entry("qwen-learning", @import("contract_qwen_learning.zig").numerical),
    back.entry("fn-lanes", @import("contract_fn_lanes.zig").run),
    back.entry("prompt-swiglu", @import("contract_prompt_swiglu.zig").run),
    back.entry("int4-skipped", int4Skipped),
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
