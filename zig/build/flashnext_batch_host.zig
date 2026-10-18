//! Source and lane regressions with a recording Metal boundary, no shader compiler or GPU work.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/flashnext_recording_metal.zig"), .target = b.graph.host });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const empty = b.addWriteFiles().add("flashnext-test-empty.zig", "pub const bytes = \"\";\n");
    sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = empty });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = b.graph.host, .link_libc = true });
    const root = b.createModule(.{ .root_source_file = b.path("zig/flashnext_batch_tests.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) } } });
    b.step("test-flashnext-vision-contracts-host", "Flash Next masked draws and vision contracts without device execution").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Flash Next contracts:"} })).step);
    b.step("test-flashnext-load-context-host", "Independent long and ordinary Flash Next load settings without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{ "Flash Next context:", "load sets no DeltaNet" } })).step);
    const tests_ = b.addTest(.{ .root_module = root, .filters = &.{"batch widths:"} });
    const run = b.addRunArtifact(tests_);
    run.has_side_effects = true;
    const step = b.step("test-flashnext-batch-host", "Actual shared lane verification at checkpoint widths, without Metal execution");
    step.dependOn(&run.step);
    all.dependOn(step);
}
