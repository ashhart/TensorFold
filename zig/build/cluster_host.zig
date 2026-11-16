//! The cluster's membership and bring-up over fake links, on the build host with no GPU.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const host = b.graph.host;
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = host, .link_libc = true });
    fabric.addImport("verbs", @import("tp.zig").verbsModule(b, host, .debug));
    const canon = @import("cuda_engines.zig").canonModule(b, host, .debug);
    const bringup = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/bringup.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "fabric", .module = fabric }, .{ .name = "canon", .module = canon } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = bringup }));
    b.step("test-cluster-membership-host", "Cluster membership and bring-up over fake links, without a GPU").dependOn(&run.step);
}
