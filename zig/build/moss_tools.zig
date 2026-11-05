//! MOSS-Image's CUDA tools: the transformer's case, the vae/ decoder and the Qwen3.5 prompt encoder.

const std = @import("std");

/// The family module (`zig/src/families/moss_image/cuda.zig`) and the three tools installed over it.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, tokenizer: *std.Build.Module, strip: bool) void {
    const moss = b.createModule(.{
        .root_source_file = b.path("zig/src/families/moss_image/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    // tf-moss-image: the transformer on a reference case (tools/zig/moss_image_case.py), compared and timed
    const transformer = b.createModule(.{
        .root_source_file = b.path("zig/tests/moss_image_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "moss_image", .module = moss } },
    });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image", .root_module = transformer }));
    // tf-moss-image-decoder: the family's vae/ decoder, a latents file to its raw float32 picture before the clamp
    const decoder = b.createModule(.{
        .root_source_file = b.path("zig/tests/moss_image_decoder_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{.{ .name = "moss_image", .module = moss }},
    });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-decoder", .root_module = decoder }));
    // tf-moss-image-encoder: the family's prompt encoder (Qwen3.5's text model), each layer gated against float32
    const encoder = b.createModule(.{
        .root_source_file = b.path("zig/tests/moss_image_encoder_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "moss_image", .module = moss } },
    });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-encoder", .root_module = encoder }));
}
