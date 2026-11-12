//! Host tests of kernel-contract's sharded run: the partition, options, tally merge and shard processes, no GPU code.
const std = @import("std");

pub fn host(b: *std.Build, core: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_shard_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "core", .module = core }},
    }) }));
    const step = b.step("test-contract-shard-host", "kernel-contract --jobs and --shard: partition, options, merge and shard processes without a GPU");
    step.dependOn(&run.step);
    all.dependOn(&run.step);
    if (b.graph.host.result.os.tag == .windows) return; // the shard processes are POSIX
    const processes = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_children_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    }) }));
    processes.has_side_effects = true;
    step.dependOn(&processes.step);
    all.dependOn(&processes.step);
}
