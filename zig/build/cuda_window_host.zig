//! The exceptional CUDA window policy is CPU tested without a GPU module or shader producer.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, all: *std.Build.Step) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/window_fallback_test.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
    });
    m.addIncludePath(b.path("zig/kernels/cuda"));
    m.addCSourceFile(.{ .file = b.path("zig/tests/cuda/window_fallback_shim.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-window-fallback-host", "Exceptional prompt head policy through the CUDA source helper, without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
