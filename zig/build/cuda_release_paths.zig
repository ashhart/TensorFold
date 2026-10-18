//! Host-only release privacy tools, shared by fatbin validation and archive staging.
const std = @import("std");
const contract = @import("kernel_contract.zig");
pub fn checker(b: *std.Build) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "check-release-paths", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/build/check_release_paths.zig"),
        .target = b.graph.host,
        .optimize = .fast,
        .strip = true,
        .imports = &.{.{ .name = "contract", .module = contract.module(b, b.graph.host, .fast) }},
    }) });
}
pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/build/release_paths_test.zig"),
        .target = b.graph.host,
        .imports = &.{.{ .name = "contract", .module = contract.module(b, b.graph.host, .debug) }},
    }), .filters = &.{"release paths:"} }));
    b.step("test-release-paths-host", "Planted path leaks in raw, wide and compressed release payloads").dependOn(&run.step);
    all.dependOn(&run.step);
    const tool = checker(b);
    b.step("compile-release-paths-host", "Build the host release checker without shader compilers").dependOn(&b.addInstallArtifact(tool, .{}).step);
}
pub fn directory(b: *std.Build, tool: *std.Build.Step.Compile, dir: std.Build.LazyPath) *std.Build.Step {
    const run = b.addSystemCommand(&.{"sh"});
    run.addFileArg(b.path("tools/release/check_paths.sh"));
    run.addArtifactArg(tool);
    run.addDirectoryArg(b.path("."));
    run.addDirectoryArg(dir);
    return &run.step;
}
