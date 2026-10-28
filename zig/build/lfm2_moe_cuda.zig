//! LFM2.5-24B-A2B's CUDA family over the maintained runtime, the core and the lane core; its host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm2_moe/cuda_family.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "liquid_cuda", .module = @import("liquid_cuda.zig").module(b, target, optimize, cuda, core, lanes) } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-lfm2-moe-cuda-host", "LFM2.5-24B-A2B on CUDA: config admission, page and state sizes, RoPE table, refusals without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
