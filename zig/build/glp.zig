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

    // The checker: the reader's view of a file, and a rewrite of it, for another implementation to compare against.
    const tool = b.addExecutable(.{ .name = "tf-glp-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/glp_check.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "glp", .module = module }} }) });
    b.step("tf-glp-check", "Show a control vector file as this reader sees it").dependOn(&b.addInstallArtifact(tool, .{}).step);
}
