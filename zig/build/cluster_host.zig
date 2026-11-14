//! Cluster membership, bring-up, and config validate on the build host with no GPU.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const host = b.graph.host;
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = host, .link_libc = true });
    fabric.addImport("verbs", @import("tp.zig").verbsModule(b, host, .debug));
    const canon = @import("cuda_engines.zig").canonModule(b, host, .debug);
    const bringup = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/bringup.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "fabric", .module = fabric }, .{ .name = "canon", .module = canon } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = bringup }));
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, host));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .link_libc = true }));
    const validate = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/validate.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "fabric", .module = fabric }, .{ .name = "canon", .module = canon }, .{ .name = "core", .module = core } } });
    const validate_run = b.addRunArtifact(b.addTest(.{ .root_module = validate, .filters = &.{"config validate:"} }));
    const step = b.step("test-cluster-membership-host", "Cluster membership, bring-up, and config validate, without a GPU");
    step.dependOn(&run.step);
    step.dependOn(&validate_run.step);
    const exe = b.addExecutable(.{ .name = "tf-config-validate", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/validate_main.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "fabric", .module = fabric }, .{ .name = "canon", .module = canon }, .{ .name = "core", .module = core } } }) });
    b.step("tf-config-validate", "Validate a cluster.json without planning or probes").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
