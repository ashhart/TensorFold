//! The source manifest, newest release notes and install examples must name the same version.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/build/release_version_test.zig"), .target = b.graph.host });
    root.addAnonymousImport("release_manifest", .{ .root_source_file = b.path("build.zig.zon") });
    root.addAnonymousImport("release_changelog", .{ .root_source_file = b.path("CHANGELOG.md") });
    root.addAnonymousImport("release_readme", .{ .root_source_file = b.path("README.md") });
    root.addAnonymousImport("release_runbook", .{ .root_source_file = b.path("RUNBOOK.md") });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-release-version-host", "Source package version matches the newest changelog release and the install examples").dependOn(&run.step);
    all.dependOn(&run.step);
}
