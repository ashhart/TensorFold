//! MiniMax-H3-Image's image decoder on CUDA: the fatbin via `-Dnvcc` (compile-only without it) and the CLI.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, opts: @import("options.zig").Cuda, test_step: *std.Build.Step) void {
    // no FMA contraction: the norms', attention's and SwiGLU's fp32 multiply-adds round as the host reference's do
    const image = if (opts.nvcc) |nvcc| @import("cuda_fatbin.zig").fatbin(b, nvcc, b.addSystemCommand(&.{ nvcc, "--version" }).captureStdOut(.{}), .{ .name = "h3_image_decoder", .flags = &.{ "-O3", "--fmad=false" }, .min_sm = 80 }, opts.sms) else b.addWriteFiles().add("h3-image-decoder-compile-only.fatbin", "");
    const cuda = @import("cuda.zig").runtime(b, target, optimize, &.{});
    const core = @import("h3_host.zig").coreModule(b, target);
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize });
    const decoder = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "picture_writer", .module = writer } } });
    decoder.addAnonymousImport("fatbin_h3_image_decoder", .{ .root_source_file = image });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-decoder-cuda", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_decoder_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "h3_image_decoder_cuda", .module = decoder },
    } }) });
    b.step("tf-h3-image-decoder-cuda", "MiniMax-H3-Image's image decoder on CUDA: a latent to a PNG, or its self-test").dependOn(&b.addInstallArtifact(exe, .{}).step);
    // a build compiles the Zig side only; nvcc and a CUDA device are needed for the fatbin and the run.
    test_step.dependOn(&exe.step);
}
