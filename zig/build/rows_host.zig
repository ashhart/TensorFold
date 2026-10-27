//! Disk spill lifecycle runs without shaders, checkpoints or devices.
const std = @import("std");
pub fn tests(b: *std.Build, api: *std.Build.Module, all: *std.Build.Step) void {
    const filter = b.option([]const u8, "rows-filter", "One spill lifecycle host regression") orelse "rows_";
    const run = b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{filter} }));
    b.step("test-rows-host", "Spill lifecycle and file exactness without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
