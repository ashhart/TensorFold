//! Host-only artifact identity mutations, using the actual checkpoint, pack and persisted-state readers.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .link_libc = true });
    api.addImport("lanes", lanes);
    const root = b.createModule(.{ .root_source_file = b.path("zig/artifact_identity_host.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    root.addImport("engine_api", api);
    root.addImport("tensorfold", root);
    const filter = b.option([]const u8, "artifact-filter", "One artifact identity regression") orelse "artifact identity:";
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{filter} }));
    b.step("test-artifact-identity-host", "Checkpoint, converted-pack and restarted-state mutation refusals without a GPU").dependOn(&run.step);
    const source = b.createModule(.{ .root_source_file = b.path("zig/flashnext_pack_source_host.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const cache_root = b.createModule(.{ .root_source_file = b.path("zig/flashnext_pack_cache_cases.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    b.step("test-flashnext-pack-cache-host", "Pack cache source guards and atomic publication without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cache_root })).step);
    const source_run = b.addRunArtifact(b.addTest(.{ .root_module = source, .filters = &.{ "the reader", "identity string", "io and mapped", "the no-dump", "a config-only" } }));
    b.step("test-flashnext-pack-source-host", "Converted pack source compatibility through the actual builder, without a device").dependOn(&source_run.step);
}
