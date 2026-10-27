//! TF_Q27_PROFILE's host checks: the trace's holds, pauses and value, and replies with and without gpu_seconds.
const std = @import("std");

pub fn steps(b: *std.Build, server_tests: *std.Build.Module) void {
    const host = b.graph.host;
    const fake = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_metal.zig"), .target = host });
    const profile = b.createModule(.{ .root_source_file = b.path("zig/src/core/gpu_profile.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "metal", .module = fake }} });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const step = b.step("test-q27-profile-host", "TF_Q27_PROFILE's trace, its value and replies' gpu_seconds, with no GPU");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = profile })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"gpu_seconds:"} })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"gpu_seconds:"} })).step);
}
