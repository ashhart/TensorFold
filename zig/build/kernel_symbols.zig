//! Literal device lookups must have a matching source token before any device build.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/kernel_symbols.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    }) });
    const run = b.addRunArtifact(cases);
    run.setCwd(b.path("."));
    run.has_side_effects = true;
    const step = b.step("test-kernel-symbols-host", "Every literal CUDA/HIP lookup has a source token, without a device");
    step.dependOn(&run.step);
    all.dependOn(step);
}
