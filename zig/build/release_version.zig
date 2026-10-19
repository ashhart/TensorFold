//! The source manifest and newest release notes must name the same version.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/build/release_version_test.zig"), .target = b.graph.host });
    root.addAnonymousImport("release_manifest", .{ .root_source_file = b.path("build.zig.zon") });
    root.addAnonymousImport("release_changelog", .{ .root_source_file = b.path("CHANGELOG.md") });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-release-version-host", "Source package version matches the newest changelog release").dependOn(&run.step);
    all.dependOn(&run.step);
}
