//! LFM3's CUDA family over the maintained runtime, the core and the lane core; its host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    // weights.zig (the admission shared with Metal) imports safetensors by name: core's, through a shim
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm3/cuda_safetensors.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm3/cuda_family.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "safetensors", .module = st }, .{ .name = "liquid_cuda", .module = @import("liquid_cuda.zig").module(b, target, optimize, cuda, core, lanes) } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-lfm3-cuda-host", "LFM3 on CUDA: admission of its three exports, page, ring and state sizes, RoPE tables, refusals without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
