//! MiniMax-H3-Image's image decoder: tf-h3-image-decoder on Metal, and a compile check needing no Metal compiler.
const std = @import("std");

/// The tool on `metal`: the decoder and its kernels' source, and the picture writer (PNG and resampler).
fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) *std.Build.Module {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_decoder.zig") });
    const decoder = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = st }, .{ .name = "h3_image_decoder_kernels", .module = kernels } } });
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_decoder.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3_image_decoder", .module = decoder }, .{ .name = "picture_writer", .module = writer } } });
}

/// `zig build tf-h3-image-decoder` (metal_targets.zig): its shaders compile at load, on any Apple GPU.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-h3-image-decoder", .root_module = module(b, target, optimize, metal) });
    b.step("tf-h3-image-decoder", "MiniMax-H3-Image's image decoder on Metal: a latent to a PNG, or its self-test against the host").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// `zig build compile-h3-image-decoder-host`: the tool compiled in Debug on macOS, no shader compiled or run.
pub fn compileHost(b: *std.Build, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag != .macos) return;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const exe = b.addExecutable(.{ .name = "tf-h3-image-decoder-host-compile", .root_module = module(b, target, .Debug, metal) });
    b.step("compile-h3-image-decoder-host", "Compile tf-h3-image-decoder without compiling or executing shaders").dependOn(&exe.step);
}
