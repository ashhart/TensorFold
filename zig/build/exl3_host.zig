//! The EXL3 host tests of the Flash Next module.
const std = @import("std");

pub fn steps(b: *std.Build, flashnext: *std.Build.Module) void {
    b.step("test-flashnext-exl3-host", "Flash Next EXL3 admission, store, sizing and loader host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{"exl3: "} })).step);
}
