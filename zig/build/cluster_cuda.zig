//! The cluster commands (node, serve --cluster) on a GB10; canon and verbs are the engine's (one file, one module).
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lanes: *std.Build.Module, core: *std.Build.Module, canon: *std.Build.Module, verbs: *std.Build.Module, fabric: *std.Build.Module) *std.Build.Module {
    _ = verbs; // the fabric module, built with the engines', imports it
    return b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/cluster.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon } },
    });
}
