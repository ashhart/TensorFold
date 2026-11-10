//! Standalone build of the native NVFP4 gates without unrelated model assets.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-nvfp4-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/nvfp4_contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    b.step("nvfp4-prompt-contract-bin", "Build the focused native NVFP4 operator gates").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const tma = b.addExecutable(.{ .name = "tf-nvfp4-tma-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/nvfp4_tma_contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    b.step("nvfp4-tma-contract-bin", "Build the raw and prepared TMA FP4 operator gates").dependOn(&b.addInstallArtifact(tma, .{}).step);
}

pub fn host(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_nvfp4_tma.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{ "NVFP4 TMA", "FP4 reference" } }));
    b.step("test-nvfp4-tma-contract-host", "TMA variants, FP64 quantization and real narrow-sum controls without CUDA").dependOn(&run.step);
    all.dependOn(&run.step);
    const compile = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/nvfp4_tma_contract_main.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    b.step("compile-nvfp4-tma-contract-host", "Compile all TMA contract launch hooks without a CUDA device").dependOn(&b.addExecutable(.{ .name = "nvfp4-tma-contract-host-check", .root_module = compile }).step);
}
