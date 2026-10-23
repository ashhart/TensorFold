//! `pull`'s own tests: `cli/hub.zig` reads the engine registry, so they need `native_engines` and EXLR8's import.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, engines: *std.Build.Module, core: *std.Build.Module, test_step: *std.Build.Step) void {
    if (target.result.os.tag == .windows) return; // the fake hub listens on a POSIX socket
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/pull_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "native_engines", .module = engines }, @import("exlr8.zig").cliImport(b, target, optimize, core) },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mod }));
    b.step("test-pull-host", "Checkpoint download paths: a nested shard folder, retries, resume and the whole-file fallback").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
