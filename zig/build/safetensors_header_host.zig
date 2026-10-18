//! Shared header validation through the core and cluster callers without GPU dependencies.
const std = @import("std");

pub fn steps(b: *std.Build, core: *std.Build.Module, all: *std.Build.Step) void {
    const checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/checkpoint.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    const root = b.createModule(.{ .root_source_file = b.path("zig/safetensors_header_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cluster_checkpoint", .module = checkpoint } } });
    const fuzz_run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"safetensors shared:"} }));
    b.step("test-safetensors-header-host", "Malformed header sweeps through both callers without shaders").dependOn(&fuzz_run.step);
    all.dependOn(&fuzz_run.step);
    const checkpoint_run = b.addRunArtifact(b.addTest(.{ .root_module = checkpoint }));
    b.step("test-cluster-checkpoint-host", "Cluster metadata and range controls without shaders").dependOn(&checkpoint_run.step);
    all.dependOn(&checkpoint_run.step);
    const loader = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/loader_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    b.step("test-cluster-loader-host", "Cluster local-file and fake-peer loading without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = loader })).step);
    const metal_core = b.createModule(.{ .root_source_file = b.path("zig/src/tensorfold.zig"), .target = b.graph.host, .link_libc = true });
    const metal_checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/checkpoint.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "core", .module = metal_core }} });
    const compile_root = b.createModule(.{ .root_source_file = b.path("zig/cluster_header_host_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = metal_core }, .{ .name = "cluster_checkpoint", .module = metal_checkpoint } } });
    const compile_object = b.addObject(.{ .name = "cluster-header-host-check", .root_module = compile_root });
    _ = compile_object.getEmittedBin();
    b.step("compile-cluster-header-host", "Compile the actual Metal core namespace and cluster header adapter without shaders").dependOn(&compile_object.step);
}
