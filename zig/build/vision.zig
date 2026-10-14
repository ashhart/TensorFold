//! The image-input module (zig/src/vision/qwen_image.zig with stb_image compiled in), shared by every build that
//! carries the engine API (the server reaches it as engine_api.qwen_image) and by the CUDA test runner.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_image.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addCSourceFile(.{ .file = b.path("zig/vendor/stb/stb_image_impl.c"), .flags = &.{ "-std=c11", "-O2" } });
    m.addIncludePath(b.path("zig/vendor/stb"));
    return m;
}
