//! The refusal for a model_type no native backend serves, pinned without a device: 0.6.6's hint only for its types.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.graph.host;
    const mod = b.createModule(.{ .root_source_file = b.path("zig/src/native/no_backend.zig"), .target = host, .optimize = .Debug });
    const step = b.step("test-no-backend-host", "The no-backend refusal: 0.6.6's install hint only for the types 0.6.6 serves");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    all.dependOn(step);
}
