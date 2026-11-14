//! The shared strict environment parser host step, without a driver or GPU.
const std = @import("std");

pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const environment = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host, .link_libc = true }) });
    const environment_step = b.step("test-environment-host", "Strict startup parsers, unchanged valid settings and named refusals without a driver");
    environment_step.dependOn(&b.addRunArtifact(environment).step);
    parent.dependOn(environment_step);
}
