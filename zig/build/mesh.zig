const std = @import("std");

/// The mesh layer (zig/src/fabric/mesh) over the caller's `canon` module: one canon instance per compilation.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, canon: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/fabric/mesh/mesh.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "canon", .module = canon }} });
}
