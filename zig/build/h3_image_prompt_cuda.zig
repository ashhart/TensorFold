//! MiniMax-H3-Image's prompt path on CUDA: tf-h3-image-prompt-cuda, its kernel image built by `-Dnvcc` or compile-only
const std = @import("std");

/// `zig build tf-h3-image-prompt-cuda`: a prompt to the refiner's text rows on CUDA, and its synthetic selftest
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, opts: @import("options.zig").Cuda, test_step: *std.Build.Step) void {
    const image = if (opts.nvcc) |nvcc|
        @import("cuda_fatbin.zig").fatbin(b, nvcc, b.addSystemCommand(&.{ nvcc, "--version" }).captureStdOut(.{}), .{ .name = "h3_image_prompt", .flags = &.{"-O3"}, .min_sm = 80 }, opts.sms)
    else
        b.addWriteFiles().add("h3-image-prompt-compile-only.fatbin", "");
    const cuda = @import("cuda.zig").runtime(b, target, optimize, &.{});
    const core = @import("h3_host.zig").coreModule(b, target);
    const prompt = b.createModule(.{
        .root_source_file = b.path("zig/src/families/h3/prompt_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "core", .module = core },
        },
    });
    prompt.addAnonymousImport("fatbin_h3_image_prompt", .{ .root_source_file = image });
    const tool = b.createModule(.{
        .root_source_file = b.path("zig/tests/h3_image_prompt_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "h3_prompt_cuda", .module = prompt },
            .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }) },
            .{ .name = "core", .module = core },
            .{ .name = "moss_noise", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize }) },
        },
    });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt-cuda", .root_module = tool });
    b.step("tf-h3-image-prompt-cuda", "MiniMax-H3-Image's prompt to its text rows on CUDA: Qwen3-VL to layer 50, the token refiner").dependOn(&b.addInstallArtifact(exe, .{}).step);
    test_step.dependOn(&exe.step);
    const tests = b.addTest(.{ .root_module = prompt });
    test_step.dependOn(&b.addRunArtifact(tests).step);
    const tool_tests = b.addTest(.{ .root_module = tool });
    test_step.dependOn(&b.addRunArtifact(tool_tests).step);
}
