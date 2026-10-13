//! Experimental DS4.1 weight-loading qualification targets.
const std = @import("std");

pub fn weights(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
) *std.Build.Step {
    const host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_weights_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
        },
    }) });
    const step = b.step("test-deepseek-v41-weights", "V4.1 configuration, Engram table plans and resident weight staging");
    step.dependOn(&b.addRunArtifact(host).step);
    return step;
}
