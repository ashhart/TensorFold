//! One source owner per compilation, shared by the family facade and legacy consumers.
const std = @import("std");
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, owner: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/kernels/metal/inkling2_facade.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "kernel_sources", .module = owner }},
    });
}
pub fn standalone(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return module(b, target, optimize, b.createModule(.{
        .root_source_file = b.path("zig/kernels/metal/sources.zig"),
        .target = target,
        .optimize = optimize,
    }));
}
