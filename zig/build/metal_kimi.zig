//! Kimi Metal module construction shared by native and release targets.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const kimi = b.createModule(.{
        .root_source_file = b.path("zig/src/families/kimi_k3/kimi_k3.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = b.createModule(.{
            .root_source_file = b.path("zig/src/families/kimi_k3/core_safetensors.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "core", .module = engine }},
        }) } },
    });
    kimi.addImport("lanes", lanes);
    return kimi;
}
