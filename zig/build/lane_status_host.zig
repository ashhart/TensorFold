//! CPU check for issue 3200: a status read holds no lane-host lock, so a monitoring client cannot stall a decode round.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const lock = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_status_lock_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    b.step("test-lane-status-host", "A status read parks without the lane host's lock (issue 3200)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lock, .filters = &.{"status:"} })).step);
    _ = api;
}
