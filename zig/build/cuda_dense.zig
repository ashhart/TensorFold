//! The shared CUDA dense-decoder engine (zig/src/cuda_dense) over the runtime, core and lane core; its host tests.
const std = @import("std");

/// The engine's module, made once a compilation and passed to each dense recipe's family module.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cuda_dense/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-cuda-dense-host", "The CUDA dense engine: attention shapes, pages, RoPE frequencies, Native decls, refusals without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
