//! `-Dfixed-cpu`: compile host code for one CPU, not whichever one this machine has. A compile's cache key carries its
//! CPU and GitHub's x86 runners differ, so without this a kept build cache only fits a runner with the same CPU.
const std = @import("std");

pub fn apply(b: *std.Build) void {
    const fixed = b.option(bool, "fixed-cpu", "Compile host code for one CPU, so a kept build cache fits other machines");
    if (!(fixed orelse false)) return;
    if (b.graph.host.result.cpu.arch != .x86_64) return;
    b.graph.host = b.resolveTargetQuery(.{ .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 } });
}
