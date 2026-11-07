//! The flashnext prompt contract's host checks: the served row range and the state alias, without a device.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, family: *std.Build.Module, all: *std.Build.Step) void {
    const compose = b.step("test-flashnext-compose-host", "Dense heads and shared expert composition without a GPU");
    compose.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "split out lane", "head split", "dense split", "full heads", "half the heads", "shared expert composition" } })).step);
    const moe = b.step("test-moetile-host", "Shared down slot fixtures, launch dispatch and contract records without a GPU");
    all.dependOn(moe);
    moe.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"shared down stride"} })).step);
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/moetile_tests.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } } });
    moe.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fixtures })).step);
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_flashnext_glue_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    all.dependOn(&run.step);
    b.step("test-flashnext-glue-contract-host", "The flashnext prompt contract's served row range and its state alias").dependOn(&run.step);
}
