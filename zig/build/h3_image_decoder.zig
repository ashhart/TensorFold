//! MiniMax-H3-Image's image decoder: tf-h3-image-decoder on Metal, and a compile check needing no Metal compiler.
const std = @import("std");

/// The tool on `metal`: the decoder and its kernels' source, and the picture writer (PNG and resampler).
fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) *std.Build.Module {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_decoder.zig") });
    const decoder = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "h3_image_decoder_kernels", .module = kernels }, .{ .name = "gpu_turn", .module = b.createModule(.{ .root_source_file = b.path("zig/src/server/gpu_turn.zig"), .target = target, .optimize = optimize }) } } });
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
    return b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_decoder.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "h3_image_decoder", .module = decoder }, .{ .name = "picture_writer", .module = writer } } });
}

/// `zig build tf-h3-image-decoder` (metal_targets.zig): its shaders compile at load, on any Apple GPU.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, test_step: *std.Build.Step) void {
    const root = module(b, target, optimize, metal);
    const exe = b.addExecutable(.{ .name = "tf-h3-image-decoder", .root_module = root });
    b.step("tf-h3-image-decoder", "MiniMax-H3-Image's image decoder on Metal: a latent to a PNG, or its self-test against the host").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const tests = b.addTest(.{ .root_module = root });
    const step = b.step("test-h3-image-decoder-load", "The decoder's weights loaded from a fixture, only its own tensors read ahead, prefetch on and off");
    for ([_][]const u8{ "16", "0" }) |count| {
        const run = b.addRunArtifact(tests);
        run.setEnvironmentVariable("TF_LOAD_PREFETCH", count);
        step.dependOn(&run.step);
    }
    test_step.dependOn(step);
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

/// `zig build tf-h3-image-decoder-host` (any OS, no GPU) and `test-h3-image-decoder-host`.
pub fn hostSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const exe = b.addExecutable(.{ .name = "tf-h3-image-decoder-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_decoder_host.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "h3_image_decoder_host", .module = host(b, target, optimize) }} }) });
    b.step("tf-h3-image-decoder-host", "MiniMax-H3-Image's image decoder on the CPU: a latent to a PNG, no GPU").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const native = b.graph.host;
    const cases = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_host_test.zig"), .target = native, .optimize = .ReleaseFast, .imports = hostImports(b, native, .ReleaseFast) }) });
    const run = b.addRunArtifact(cases);
    b.step("test-h3-image-decoder-host", "The CPU image decoder: tiles across cores equal the serial reference chain").dependOn(&run.step);
    test_step.dependOn(&run.step);
}

fn host(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_decoder_host.zig"), .target = target, .optimize = optimize, .imports = hostImports(b, target, optimize) });
}

/// `core` and the picture writer on it, for the decoder's CPU and CUDA paths.
pub fn hostImports(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) []const std.Build.Module.Import {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
    return b.allocator.dupe(std.Build.Module.Import, &.{ .{ .name = "core", .module = core }, .{ .name = "picture_writer", .module = writer } }) catch @panic("OOM");
}
