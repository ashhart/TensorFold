//! Ling family host checks: config, weight map, fixtures and synth JSON. No CUDA, no Metal, no GPU.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/ling/host_test.zig"),
        .target = target,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-ling-host", "Ling family package: config, names, tiny fixtures, synth JSON").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
