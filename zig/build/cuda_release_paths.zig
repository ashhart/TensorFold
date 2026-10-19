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
    const clean = b.addWriteFiles();
    const clean_file = clean.add("noise.bin", "\x01./.\x02");
    const bad = b.addWriteFiles();
    const root = absoluteRoot(b);
    _ = bad.add("leak.bin", b.fmt("prefix:{s}/assets/data.bin", .{root}));
    const directory_test = b.step("test-release-directory-host", "Actual release directory invocation accepts code fragments and refuses its absolute build root");
    directory_test.dependOn(directory(b, tool, clean.getDirectory()));
    const refuses = directory(b, tool, bad.getDirectory());
    refuses.cast(std.Build.Step.Run).?.expectExitCode(1);
    directory_test.dependOn(refuses);
    const command_test = b.step("test-release-paths-command-host", "Compiled privacy command refuses invalid roots before reading inputs");
    for ([_][]const u8{ "./.", "/", "//", "/.", "/../", "C:\\", "C:/", "C:relative", "relative/build", "\\relative", "C:\\..\\.", "\\\\server" }) |invalid| {
        const check = b.addRunArtifact(tool);
        check.addArg(invalid);
        check.addFileArg(clean_file);
        check.expectExitCode(2);
        check.expectStdErrEqual("release privacy: BadBuildRoot: use a named absolute build root\n");
        command_test.dependOn(&check.step);
    }
    const unreadable = b.addRunArtifact(tool);
    unreadable.addArgs(&.{ "./.", "." });
    unreadable.expectExitCode(2);
    unreadable.expectStdErrEqual("release privacy: BadBuildRoot: use a named absolute build root\n");
    command_test.dependOn(&unreadable.step);
    const valid = b.addRunArtifact(tool);
    valid.addArg(root);
    valid.addFileArg(clean_file);
    valid.expectExitCode(0);
    valid.expectStdErrEqual("");
    command_test.dependOn(&valid.step);
    for ([_][]const u8{ "C:\\release-root", "\\\\server\\share\\build" }) |absolute| {
        const check = b.addRunArtifact(tool);
        check.addArg(absolute);
        check.addFileArg(clean_file);
        check.expectExitCode(0);
        check.expectStdErrEqual("");
        command_test.dependOn(&check.step);
    }
}
pub fn directory(b: *std.Build, tool: *std.Build.Step.Compile, dir: std.Build.LazyPath) *std.Build.Step {
    const run = b.addSystemCommand(&.{"sh"});
    run.addFileArg(b.path("tools/release/check_paths.sh"));
    run.addArtifactArg(tool);
    run.addArg(absoluteRoot(b));
    run.addDirectoryArg(dir);
    return &run.step;
}

pub fn absoluteRoot(b: *std.Build) []const u8 {
    const cwd = std.process.currentPathAlloc(b.graph.io, b.allocator) catch @panic("cannot read release build directory");
    const root = std.fs.path.resolveAlloc(b.allocator, &.{ cwd, b.root.root_dir.path orelse ".", b.root.sub_path }) catch @panic("OOM");
    std.debug.assert(std.fs.path.isAbsolute(root));
    return root;
}
