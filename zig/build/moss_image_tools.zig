//! MOSS-Image's CUDA tools: tf-moss-image (a case), -blocks (each block alone), -decoder, -encoder, -lora (a gate).
const std = @import("std");

/// The first four run tools/zig/moss_image_case.py's reference cases, -lora real weights alone; cuda.zig installs them.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const moss = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const tokenizer = core.import_table.get("tokenizer").?;
    const moss_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image", .root_module = moss_tool }));
    const moss_blocks = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_blocks_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-blocks", .root_module = moss_blocks }));
    const decoder = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_decoder_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "moss_image", .module = moss }} });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-decoder", .root_module = decoder }));
    const encoder = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_encoder_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-encoder", .root_module = encoder }));
    const lora = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_lora_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "moss_image", .module = moss } } });
    const lora_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-moss-image-lora", .root_module = lora }), .{});
    b.getInstallStep().dependOn(&lora_install.step);
    b.step("moss-image-lora-cuda", "tf-moss-image-lora: MOSS-Image's runtime LoRAs per projection class against FP64, real weights").dependOn(&lora_install.step);
}
