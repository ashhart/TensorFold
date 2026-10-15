//! The image-input module (vision/qwen_image.zig), shared by the engine API and the CUDA test runner.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_image.zig"), .target = target, .optimize = optimize, .link_libc = true });
}
