//! ZONOS3 Mini's CUDA family module and its zonos3-generate tool.
const std = @import("std");

/// The family over a runtime and the core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/zonos3/zonos3.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const tool = b.createModule(.{ .root_source_file = b.path("zig/src/families/zonos3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda }, .{ .name = "zonos3", .module = module(b, target, optimize, cuda, core) } } });
    const exe = b.addExecutable(.{ .name = "zonos3-generate", .root_module = tool });
    b.step("zonos3-generate", "ZONOS3 Mini's backbone on CUDA against the reference's dumps").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
