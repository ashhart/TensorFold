//! Session memory's build: the family-neutral `sessions` module (zig/src/sessions) and its host tests on any OS.

const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/sessions/sessions.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug) }));
    b.step("test-sessions", "Session memory's host tests: pool, prefix trie, RAM tier, NVMe tier on real files (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
