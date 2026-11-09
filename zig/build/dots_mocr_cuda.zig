//! dots.mocr's CUDA text family over the runtime, the core, the lane core and the dense engine; its host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, dense: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/dots_mocr/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "cuda_dense", .module = dense } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-dots-mocr-cuda-host", "dots.mocr on CUDA: geometry, text bytes, Native decls and refusals without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
