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
    if (host.result.os.tag != .linux) {
        const linux = b.resolveTargetQuery(.{ .cpu_arch = host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
        const full = b.addTest(.{ .root_module = module(b, linux, .debug) });
        b.step("check-sessions-linux", "Compile the full session/NVMe tests without executing Linux code")
            .dependOn(&full.step);
    }
    b.step("test-review-sessions", "Ownership rollback: actual RAM Store failure regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests, .filters = &.{"Ownership rollback:"} })).step);
    b.step("test-residency-review", "Residency rollback: rollback after committed disk policy side effects").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests, .filters = &.{"Residency rollback:"} })).step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    b.step("test-sessions", "Session memory's host tests: pool, prefix trie, RAM tier, NVMe tier on Linux (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
