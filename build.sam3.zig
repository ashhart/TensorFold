const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });
    const targets = @import("zig/build/sam3.zig").steps(b, target, optimize);
    b.step("test", "SAM host and tokenizer tests").dependOn(targets.tests);
    b.step("inspect", "Build SAM tools").dependOn(targets.tools);
    b.step("kernels", "Check SAM Metal arithmetic").dependOn(targets.kernels);
    b.default_step = targets.tests;
    @import("zig/build/profile.zig").walk(b);
}
