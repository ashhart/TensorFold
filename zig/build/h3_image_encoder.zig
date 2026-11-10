//! MiniMax-H3-Image's edit encoder: `tf-h3-image-encoder`, whose Metal is compiled at load, and its host compile check.
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) *std.Build.Module {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_encoder.zig") });
    const encoder = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_encoder.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = st }, .{ .name = "h3_image_encoder_kernels", .module = kernels } } });
    const codec = b.createModule(.{ .root_source_file = b.path("zig/src/core/picture/picture.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const noise = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_encoder.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3_image_encoder", .module = encoder }, .{ .name = "picture_codec", .module = codec }, .{ .name = "torch_noise", .module = noise } } });
}

/// `zig build tf-h3-image-encoder`: a reference picture to its clean rows, or `selftest`; needs no Metal compiler.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-h3-image-encoder", .root_module = module(b, target, optimize, metal) });
    b.step("tf-h3-image-encoder", "MiniMax-H3-Image's edit encoder on Metal: a reference to its clean rows, or its selftest").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// `zig build compile-h3-image-encoder`: the tool compiled in Debug, no shader compiled or run.
pub fn compileHost(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-h3-image-encoder-host-compile", .root_module = module(b, target, .Debug, metal) });
    b.step("compile-h3-image-encoder", "Compile tf-h3-image-encoder without compiling or running its shaders").dependOn(&exe.step);
}
