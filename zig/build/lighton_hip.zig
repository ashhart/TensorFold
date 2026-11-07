//! HIP LightOnOCR module construction and host checks, using the native server's shared engine types.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, hip: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, serve: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lighton_ocr/hip.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api }, .{ .name = "hip_serve", .module = serve } } });
}

pub fn vision(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, hip: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, serve: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_hip.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "hip_serve", .module = serve } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-lighton-hip-host", "LightOnOCR HIP image, state and admission tests without devices").dependOn(&run.step);
    step.dependOn(&run.step);
}
