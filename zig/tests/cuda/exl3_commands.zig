//! The CUDA runner's one-GPU EXL3 commands (exl3 and exl3-*; exl3-tp2 stays in main.zig with its argument checks).
const std = @import("std");
const check = @import("check.zig");
const plans = @import("exl3_plan_bench.zig");

const Command = enum { exl3, @"exl3-grouped-tile", @"exl3-tiles", @"exl3-shapes", @"exl3-midm", @"exl3-plans", @"exl3-contract", @"exl3-prompt-k6" };

/// Runs `cmd` when it is one of these commands; false leaves it to the runner's others (and its usage).
pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !bool {
    const first: ?[]const u8 = if (rest.len > 0) rest[0] else null;
    switch (std.meta.stringToEnum(Command, cmd) orelse return false) {
        .exl3 => try @import("exl3_tests.zig").run(gpu, first orelse "none"),
        .@"exl3-grouped-tile" => try @import("exl3_grouped_tile_tests.zig").run(gpu),
        .@"exl3-tiles" => try @import("exl3_candidate_tests.zig").run(gpu, first),
        .@"exl3-shapes" => try plans.profileShapes(gpu, first),
        .@"exl3-midm" => try plans.midm(gpu),
        .@"exl3-plans" => try plans.run(gpu, first),
        .@"exl3-contract" => try @import("contract_exl3.zig").run(gpu),
        .@"exl3-prompt-k6" => try @import("exl3_prompt_k6_tests.zig").run(gpu, first orelse "none", if (rest.len > 1) rest[1] else null),
    }
    return true;
}
