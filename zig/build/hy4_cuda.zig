//! Hy4-flash CUDA tools: the greedy text generator for EXL3 routed-expert checkpoints.
const std = @import("std");

pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/cuda_generate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    root.addImport("engram_hash_core", b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"),
        .target = target,
        .optimize = optimize,
    }));
    const exe = b.addExecutable(.{ .name = "tf-hy4-cuda-generate", .root_module = root });
    b.step("hy4-cuda-generate", "Build the Hy4-flash CUDA greedy generator for EXL3 checkpoints").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
