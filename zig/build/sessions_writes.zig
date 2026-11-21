//! The sessions tier's write bench: the tool, its run and its host tests; no model, no GPU.
const std = @import("std");

fn testModule(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const imports: []const std.Build.Module.Import = &.{.{ .name = "sessions", .module = @import("sessions.zig").module(b, target, .ReleaseSafe) }};
    return b.createModule(.{ .root_source_file = b.path("zig/sessions_writes_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = imports });
}

pub fn steps(b: *std.Build) void {
    const target = b.graph.host;
    const imports: []const std.Build.Module.Import = &.{.{ .name = "sessions", .module = @import("sessions.zig").module(b, target, .ReleaseSafe) }};
    if (target.result.os.tag != .linux) {
        const linux = b.resolveTargetQuery(.{ .cpu_arch = target.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
        const full = b.addTest(.{ .root_module = testModule(b, linux) });
        _ = full.getEmittedBin();
        b.step("check-sessions-writes-linux", "Compile the write bench's real-file calibration without executing Linux code").dependOn(&full.step);
    }
    const tool = b.addExecutable(.{ .name = "tf-sessions-writes", .root_module = b.createModule(.{ .root_source_file = b.path("zig/sessions_writes.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = imports }) });
    b.step("compile-sessions-writes-host", "Compile the sessions tier's write bench").dependOn(&b.addInstallArtifact(tool, .{}).step);
    const run = b.addRunArtifact(tool);
    run.addPassthruArgs();
    run.has_side_effects = true;
    b.step("sessions-writes", "Bytes the sessions NVMe tier writes and reads back on synthetic agent, chat and mixed loads").dependOn(&run.step);
    const tests = b.addTest(.{ .root_module = testModule(b, target) });
    b.step("test-sessions-writes-host", "The write bench's accounting against Disk.plan, and on Linux against real files").dependOn(&b.addRunArtifact(tests).step);
}
