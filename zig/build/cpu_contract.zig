//! The CPU kernels' contract: tf-cpu-contract runs gates 1 to 4 and the controls on the host, no device; its tests.
const std = @import("std");

/// `cpu-contract` runs the program; `test-kernel-contract` runs it as a test, and `test` the kernels' own tests.
pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cpu_kernels.zig"), .target = host, .optimize = .ReleaseFast });
    const mixers = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cpu_mixers.zig"), .target = host, .optimize = .ReleaseFast });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cpu/contract_main.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .imports = &.{ .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, host, .ReleaseFast) }, .{ .name = "cpu_mixers", .module = mixers } },
    });
    const exe = b.addExecutable(.{ .name = "tf-cpu-contract", .root_module = root });
    b.step("tf-cpu-contract", "Build the CPU kernels' contract gates and controls (no run)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    b.step("cpu-contract", "The CPU projection, norm and the engine's rows through gates 1 to 4 on this host's cores, each control red").dependOn(&b.addRunArtifact(exe).step);
    const own = b.addRunArtifact(b.addTest(.{ .root_module = kernels }));
    b.step("test-cpu-kernels", "The CPU kernels' own tests: the same bytes at 1 to 16 threads and at column splits every 8 columns to 256 and every 256 beyond, on every host").dependOn(&own.step);
    test_step.dependOn(&own.step);
    const step = b.step("test-cpu-contract-host", "The CPU kernels through the contract's gates 1 to 4 as a test, every control red, no device");
    step.dependOn(&own.step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
    if (b.top_level_steps.get("test-kernel-contract")) |all| all.step.dependOn(step);
}
