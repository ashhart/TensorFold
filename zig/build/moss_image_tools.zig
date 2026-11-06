//! MOSS-Image's CUDA tools: tf-moss-image (a case), -blocks (each block alone), -decoder (the vae/) and -encoder.
const std = @import("std");

/// The four tools run tools/zig/moss_image_case.py's reference cases; cuda.zig installs them with its CUDA targets.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const moss = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    // tf-moss-image: the transformer on a reference case, its own prompt encoder and image decoder from that folder
    const moss_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "tokenizer", .module = core.import_table.get("tokenizer").? }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image", .root_module = moss_tool }));
    const moss_blocks = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_blocks_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-blocks", .root_module = moss_blocks }));
    // tf-moss-image-decoder: the family's vae/ decoder, a latents file to its raw float32 picture before the clamp
    const decoder = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_decoder_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "moss_image", .module = moss }} });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-decoder", .root_module = decoder }));
    // tf-moss-image-encoder: the family's prompt encoder (Qwen3.5's text model), each layer gated against float32
    const encoder = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_encoder_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "tokenizer", .module = core.import_table.get("tokenizer").? }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-encoder", .root_module = encoder }));
}
