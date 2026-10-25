//! The installable DeepSeek-V4.1 CUDA pair on real images: tensorfold-dsv41 (rank 0) and tf-dsv41 (the other ranks).
const std = @import("std");
const family = @import("dsv41_cuda.zig");
const images = @import("dsv41_images.zig");

/// `dsv41-serve` installs both over the CUDA runtime `cuda`; family images come from -Ddsv41-nvcc or -Ddsv41-fatbins.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) void {
    const step = b.step("dsv41-serve", "Install tensorfold-dsv41 and tf-dsv41 over the CUDA runtime and family images");
    const im = images.built orelse {
        step.dependOn(&b.addFail("dsv41-serve needs -Ddsv41-nvcc=PATH or -Ddsv41-fatbins=DIR").step);
        return;
    };
    const dk = images.module(b, target, optimize, cuda, im);
    const tool = b.addExecutable(.{ .name = "tf-dsv41", .root_module = family.assemble(b, target, optimize, "tool_main.zig", cuda, dk) });
    step.dependOn(&b.addInstallArtifact(tool, .{}).step);
    const engine = family.assemble(b, target, optimize, "serve_engine.zig", cuda, dk);
    step.dependOn(&b.addInstallArtifact(@import("dsv41_draft.zig").serverOn(b, target, optimize, engine), .{}).step);
}
