//! The control vector's plan on the host: the layer lookup and the directions, with no GPU and no model.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-glp-steer", "The Qwen27 control vector's plan and its layer lookup, without a GPU");
    all.dependOn(step);
    // the recording device stands in for Metal: the plan's buffers live in host memory
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_metal.zig"), .target = target });
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen27/steer_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}
