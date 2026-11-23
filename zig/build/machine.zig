//! The machine facts host step: chip bandwidth, GPU limits and the nvidia-smi and ioreg readers, without a GPU.
const std = @import("std");

pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const machine = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/platform/machine.zig"), .target = b.graph.host, .link_libc = true }) });
    const step = b.step("test-machine-host", "Chip bandwidth, GPU limits and the machine readers without a GPU");
    step.dependOn(&b.addRunArtifact(machine).step);
    parent.dependOn(step);
}
