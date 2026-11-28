//! The control vector's serve checks need no GPU, no model download and no checkpoint.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-glp-serve", "Control vector flags, refusals and the startup line");
    all.dependOn(step);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/server/glp_startup_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = api }} });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}
