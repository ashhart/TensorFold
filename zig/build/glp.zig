//! The GLP control vector reader's host checks need no GPU, no model download and no checkpoint.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-glp", "Control vector container, metadata rules and refusals");
    all.dependOn(step);
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/glp.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}
