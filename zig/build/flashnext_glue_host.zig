//! The flashnext prompt contract's host checks: the served row range and the state alias, without a device.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, family: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_flashnext_glue_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    all.dependOn(&run.step);
    b.step("test-flashnext-glue-contract-host", "The flashnext prompt contract's served row range and its state alias").dependOn(&run.step);
}
