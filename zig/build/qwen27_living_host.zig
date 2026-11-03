//! Actual portable pocket loading, row dispatch and allocation ownership without a device.
const std = @import("std");
pub fn tests(b: *std.Build) void {
    const target = b.graph.host;
    const real = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target });
    const metal = b.createModule(.{ .root_source_file = b.path("zig/qwen27_pocket_recording.zig"), .target = target, .imports = &.{.{ .name = "real_metal", .module = real }} });
    const core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_living_core.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "metal", .module = metal }} });
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .link_libc = true });
    const sources = b.createModule(.{ .root_source_file = b.addWriteFiles().add("recording-pocket-source.zig", "pub const pocket = \"recorded, not compiled\";"), .target = target });
    const pocket = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/pocket.zig"), .target = target, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "qwen_runtime_sources", .module = sources } } });
    const storage = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/pocket_storage.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "living", .module = living } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/qwen27_living_host.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "living", .module = living }, .{ .name = "pocket", .module = pocket }, .{ .name = "storage", .module = storage } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    run.has_side_effects = true;
    b.step("test-qwen27-living-host", "Portable dense pocket I/O, actual dispatch and untouched stream rows without shaders").dependOn(&run.step);
    const linear_core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_linear_core.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "metal", .module = metal }} });
    const linear = b.createModule(.{ .root_source_file = b.path("zig/qwen27_linear_host.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = linear_core } } });
    b.step("test-qwen27-linear-host", "Actual mixed-width linear loader and original row views without a pocket or GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = linear, .filters = &.{"qwen27 linear:"} })).step);
}
