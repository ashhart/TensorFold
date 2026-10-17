//! The MLX affine 4-bit quantizer's host checks: tf-merge-check and its tests, on every OS.

const std = @import("std");

/// `tf-merge-check --quantizer` and the tests of the MLX quantizer it uses: host only, no model, no GPU.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const merge_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/core/lora_merge_dir.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const merge_tests = b.addRunArtifact(b.addTest(.{ .root_module = merge_mod }));
    b.step("test-merge", "MLX quantizer and tf-merge-check tests (no model, no GPU)").dependOn(&merge_tests.step);
    const merge_check = b.addExecutable(.{ .name = "tf-merge-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/merge_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "lora_merge", .module = merge_mod }},
    }) });
    const about = "Check the MLX quantizer against shipped tensors from their dense source";
    b.step("tf-merge-check", about).dependOn(&b.addInstallArtifact(merge_check, .{}).step);
}
