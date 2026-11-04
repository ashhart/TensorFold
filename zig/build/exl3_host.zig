//! The EXL3 host tests of the Flash Next module, and its TP prefill reduction launches through the contract gates.
const std = @import("std");

pub fn steps(b: *std.Build, core: *std.Build.Module, flashnext: *std.Build.Module, all: []const *std.Build.Step) void {
    @import("flashnext_tppf_host.zig").steps(b, core, flashnext, all);
    b.step("test-flashnext-exl3-host", "Flash Next EXL3 admission, store, sizing and loader host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{"exl3: "} })).step);
}
