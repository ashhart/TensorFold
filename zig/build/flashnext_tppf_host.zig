//! The TP prefill reduction launches' host cases through the real gates: oracles, stress and planted faults, no device.
const std = @import("std");

pub fn steps(b: *std.Build, core: *std.Build.Module, family: *std.Build.Module, all: []const *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_flashnext_exl3_tppf_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    for (all) |step| step.dependOn(&run.step);
    b.step("test-flashnext-tppf-contract-host", "The TP prefill launches' oracles, stress and planted faults through the gates").dependOn(&run.step);
}
