const std = @import("std");

/// The mesh layer (zig/src/fabric/mesh) over the caller's `canon` module: one canon instance per compilation.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, canon: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/fabric/mesh/mesh.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "canon", .module = canon }} });
}

/// Qwen3.5's cuts as mesh declarations: a test-only module, so no binary roots canon or mesh twice.
pub fn qwenDeclTest(b: *std.Build, core: *std.Build.Module) *std.Build.Step {
    const host = b.graph.host;
    const mesh = module(b, host, .debug, @import("cuda_engines.zig").canonModule(b, host, .debug));
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/slicing_decl_test.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "mesh", .module = mesh } } });
    return &b.addRunArtifact(b.addTest(.{ .root_module = root })).step;
}
