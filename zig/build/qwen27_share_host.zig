//! Native owner eligibility and complete adapter compilation with inert packed assets, never a shader producer.
const std = @import("std");
pub fn steps(b: *std.Build) void {
    const target = b.graph.host;
    const files = b.addWriteFiles();
    _ = files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const ids = b.createModule(.{ .root_source_file = files.add("share-ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const mods = @import("metal_targets.zig").metalEngineModules(b, target, .debug, ids);
    mods.sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = files.add("share-empty.zig", "pub const bytes: []const u8 = &.{};\n") });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_served.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "tensorfold", .module = mods.engine }, .{ .name = "engine_api", .module = api } } });
    b.step("test-qwen27-share-owner-host", "Native snapshot owner and lineage eligibility with inert assets").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host, .filters = &.{"qwen27 share:"} })).step);
    const root = b.createModule(.{ .root_source_file = b.path("zig/qwen27_speed_compile.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "tensorfold", .module = mods.engine }, .{ .name = "engine_api", .module = api }, .{ .name = "qwen27_served", .module = host } } });
    const object = b.addObject(.{ .name = "qwen27-share-native", .root_module = root });
    _ = object.getEmittedBin();
    b.step("check-qwen27-share-host", "Complete native cache adapter with inert assets and no execution").dependOn(&object.step);
}
