//! Standalone build of the native FP8 gates without unrelated model assets.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-fp8-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/fp8_contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    b.step("fp8-prompt-contract-bin", "Build the focused native FP8 operator gates").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
