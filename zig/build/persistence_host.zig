//! Portable persistence identity execution and Darwin-only semantic compilation without device or shader tools.
const std = @import("std");
pub fn steps(b: *std.Build, draft_ids: *std.Build.Module) void {
    const host = b.graph.host;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = host, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const root = b.createModule(.{ .root_source_file = b.path("zig/persistence_identity_test.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "engine_api", .module = api } } });
    root.addImport("tensorfold", root);
    const filter = b.option([]const u8, "persistence-identity-filter", "One persisted-state identity regression");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = if (filter) |f| &.{f} else &.{"persistence identity:"} }));
    b.step("test-persistence-identity-host", "Actual checkpoint and stable-content persistence identities without devices").dependOn(&run.step);
    const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos });
    const mods = @import("metal_targets.zig").metalEngineModules(b, target, .debug, draft_ids);
    mods.metal.frameworks.clearRetainingCapacity();
    mods.metal.link_objects.clearRetainingCapacity();
    const empty = b.addWriteFiles().add("persistence-compile-empty.metallib", "");
    mods.sources.addImport("nemotron_packed_metallib", @import("metal_targets.zig").embedded(b, empty, "nemotron_packed.metallib"));
    const target_api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    mods.engine.addImport("engine_api", target_api);
    const compile = b.createModule(.{ .root_source_file = b.path("zig/flashnext_persistence_compile.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "tensorfold", .module = mods.engine }, .{ .name = "engine_api", .module = target_api } } });
    b.step("compile-persistence-host", "Compile actual Metal persistence startup consumers as a Darwin object").dependOn(&b.addObject(.{ .name = "persistence-host-compile", .root_module = compile }).step);
    const flint = b.createModule(.{ .root_source_file = b.path("zig/src/native/flint.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "tensorfold", .module = mods.engine }, .{ .name = "engine_api", .module = target_api } } });
    const flint_context = b.createModule(.{ .root_source_file = b.path("zig/flint_draft_context_host.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "flint", .module = flint }} });
    b.step("compile-flint-draft-context-host", "Compile the served Flint DFlash context API as a Darwin object").dependOn(&b.addObject(.{ .name = "flint-draft-context-host", .root_module = flint_context }).step);
}
