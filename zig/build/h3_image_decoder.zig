//! MiniMax-H3-Image's image decoder: tf-h3-image-decoder on Metal, its CPU-only sibling, and compile checks.
const std = @import("std");

/// The decoder on the CPU (image_decoder_host.zig): the reference chain over tiles, the PNG writer.
fn host(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module, writer: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_host.zig"), .target = target, .optimize = optimize, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "picture_writer", .module = writer } } });
}

/// The tool on `metal`: the decoder and its kernels' source, and the picture writer (PNG and resampler).
fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) *std.Build.Module {
    const core = @import("h3_host.zig").coreModule(b, target);
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_decoder.zig") });
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize });
    const decoder = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "h3_image_decoder_kernels", .module = kernels }, .{ .name = "picture_writer", .module = writer } } });
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

/// `zig build tf-h3-image-decoder-host` (any target, no GPU), and its tiling/blend regression on this host.
pub fn hostSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const core = @import("h3_host.zig").coreModule(b, target);
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-decoder-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_decoder_host.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "h3_image_decoder_host", .module = host(b, target, optimize, core, writer) }} }) });
    b.step("tf-h3-image-decoder-host", "MiniMax-H3-Image's image decoder on the CPU: a latent to a PNG, no GPU").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const native = b.graph.host;
    const cases = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_host_test.zig"), .target = native, .optimize = .ReleaseFast, .imports = &.{
        .{ .name = "core", .module = @import("h3_host.zig").coreModule(b, native) },
        .{ .name = "picture_writer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = native }) },
    } }) });
    const run = b.addRunArtifact(cases);
    b.step("test-h3-image-decoder-host", "The CPU image decoder: tiles across cores equal the serial reference chain").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
