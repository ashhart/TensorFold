//! Session memory: portable bookkeeping tests everywhere, NVMe tier tests on Linux.

const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/sessions/sessions.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const tests = if (host.result.os.tag == .linux) module(b, host, .debug) else b.createModule(.{
        .root_source_file = b.path("zig/src/sessions/portable_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    b.step("test-sessions", "Session memory's host tests: pool, prefix trie, RAM tier, NVMe tier on Linux (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
