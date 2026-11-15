//! Pair transport and handoff tests need neither a shader compiler nor a device.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const host = b.graph.host;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/flashnext_recording_metal.zig"), .target = host });
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = host, .link_libc = true });
    const verbs = @import("tp.zig").verbsModule(b, host, .debug);
    fabric.addImport("verbs", verbs);
    const root = b.createModule(.{ .root_source_file = b.path("zig/flashnext_pair_host.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "fabric", .module = fabric } } });
    root.addImport("pair_diagnostics", b.createModule(.{ .root_source_file = b.path("zig/flashnext_pair_diagnostics.zig"), .target = host }));
    const pair = b.addTest(.{ .root_module = root, .test_runner = .{ .path = b.path("zig/flashnext_pair_runner.zig"), .mode = .simple } });
    b.step("test-flashnext-pair-host", "Shared-event relay arrivals, terminal failures and encoded handoff order").dependOn(&b.addRunArtifact(pair).step);
    const endpoint = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/mcdma.zig"), .target = host, .link_libc = true, .imports = &.{.{ .name = "verbs", .module = verbs }} });
    b.step("test-mcdma-host", "Endpoint posting, deadlines and native flush ownership without a link").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = endpoint })).step);
}
