//! test-benchmark-group-host: the loaded benchmark group's inventory and receipt consistency, on the CPU.
const std = @import("std");

pub fn steps(b: *std.Build, modules: []const *std.Build.Module) void {
    const about = "Loaded benchmark group inventory and receipt consistency on the CPU";
    const step = b.step("test-benchmark-group-host", about);
    for (modules) |m| {
        const t = b.addTest(.{ .root_module = m, .filters = &.{"benchmark group:"} });
        step.dependOn(&b.addRunArtifact(t).step);
    }
}
