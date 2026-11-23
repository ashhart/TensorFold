//! `zig build tf-h3-image-server` (h3_metal.zig): MiniMax-H3-Image's three tools resident in one process.
const std = @import("std");

pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, h3_kernels: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    // the PNG writer and its resampler, which the decoder and the server's timing line read pictures through
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    // one module over the transformer, prompt path and decoder: they share source files
    const image = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "metal", .module = metal },
        .{ .name = "core", .module = core },
        .{ .name = "h3_kernels", .module = h3_kernels },
        .{ .name = "kernel_sources", .module = sources },
        .{ .name = "h3_prompt_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/prompt_kernels.zig") }) },
        .{ .name = "h3_image_decoder_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_decoder.zig") }) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-server", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_server.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "metal", .module = metal },
        .{ .name = "core", .module = core },
        .{ .name = "h3", .module = image },
        .{ .name = "h3_prompt", .module = image },
        .{ .name = "h3_image_decoder", .module = image },
        .{ .name = "keepalive", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/keepalive.zig"), .target = target, .optimize = optimize }) },
        .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }) },
        .{ .name = "moss_noise", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize }) },
        .{ .name = "picture_writer", .module = writer },
    } }) });
    b.step("tf-h3-image-server", "MiniMax-H3-Image resident: prompt encoder, transformer and decoder loaded once, one picture a stdin line").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const tests = b.addTest(.{ .root_module = exe.root_module });
    const run = b.addRunArtifact(tests);
    b.step("test-h3-image-server", "The resident server's binding reuse, failed-bind unwind and over-long line reader").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
