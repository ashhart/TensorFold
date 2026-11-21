//! The core visit plan's host tests (no GPU): a looped stack's steps, passes, state slots and taps.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const root = b.path("zig/src/core/visit_plan.zig");
    const m = b.createModule(.{ .root_source_file = root, .target = target, .optimize = .Debug });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    const about = "Core visit plan: prelude, core passes, coda, state slots by kind and taps";
    b.step("test-visit-plan-host", about).dependOn(&run.step);
    test_step.dependOn(&run.step);
}
