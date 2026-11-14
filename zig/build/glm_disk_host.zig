//! Mutable-state disk policy uses the actual GLM adapter without opening a device.
const std = @import("std");
pub fn steps(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/native/glm_disk.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{"GLM disk:"} }));
    all.dependOn(&run.step);
    b.step("test-glm-disk-host", "Actual GLM mutable snapshot disk policy without devices").dependOn(&run.step);
}
