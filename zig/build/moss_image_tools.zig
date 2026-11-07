//! MOSS-Image's CUDA tools: tf-moss-image (a case, gated and timed) and tf-moss-image-blocks (each block alone).
const std = @import("std");

/// Both tools run tools/zig/moss_image_case.py's reference cases; cuda.zig installs them with its CUDA targets.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const moss = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const moss_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image", .root_module = moss_tool }));
    const moss_blocks = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_blocks_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "moss_image", .module = moss } } });
    b.installArtifact(b.addExecutable(.{ .name = "tf-moss-image-blocks", .root_module = moss_blocks }));
}
