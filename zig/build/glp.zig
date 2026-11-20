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

    // A command line that cannot run is refused with a usage line and status 2 before any file is opened; the files
    // named here do not exist, so a read of one first would fail with another status.
    const refused = [_]struct { args: []const []const u8, says: []const u8 }{
        .{ .args = &.{"read"}, .says = "read takes FILE OUT.json OUT.bin" },
        .{ .args = &.{ "read", "vector.gguf" }, .says = "read takes FILE OUT.json OUT.bin" },
        .{ .args = &.{ "read", "vector.gguf", "out.json" }, .says = "read takes FILE OUT.json OUT.bin" },
        .{ .args = &.{ "read", "vector.gguf", "out.json", "out.bin", "more" }, .says = "read takes FILE OUT.json OUT.bin" },
        .{ .args = &.{"rewrite"}, .says = "rewrite takes FILE OUT.gguf" },
        .{ .args = &.{ "rewrite", "vector.gguf" }, .says = "rewrite takes FILE OUT.gguf" },
        .{ .args = &.{ "rewrite", "vector.gguf", "out.gguf", "--repo" }, .says = "each take a value" },
        .{ .args = &.{ "rewrite", "vector.gguf", "out.gguf", "--other", "x" }, .says = "only --repo and --revision" },
        .{ .args = &.{ "convert", "vector.gguf" }, .says = "the command is read or rewrite" },
    };
    for (refused) |case| {
        const run = b.addRunArtifact(tool);
        run.addArgs(case.args);
        run.setName(b.fmt("tf-glp-check refuses: {s}", .{std.mem.join(b.allocator, " ", case.args) catch @panic("OOM")}));
        run.expectExitCode(2);
        run.expectStdErrMatch(case.says);
        step.dependOn(&run.step);
    }
    const usage = b.addRunArtifact(tool);
    usage.addArg("--help");
    usage.setName("tf-glp-check --help");
    usage.expectExitCode(0);
    usage.expectStdOutMatch("tf-glp-check read FILE OUT.json OUT.bin");
    step.dependOn(&usage.step);
}
