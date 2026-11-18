//! Hy4-flash family host checks: config. No CUDA, no Metal, no GPU.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/host_test.zig"),
        .target = target,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-hy4-flash-host", "Hy4-flash family package: config and its refusals").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
