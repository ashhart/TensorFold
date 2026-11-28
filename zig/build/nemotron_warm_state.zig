//! Host-backed buffers drive the real Nemotron cache slots, snapshot copies and interrupted prompt pass.
const std = @import("std");
pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const host = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .link_libc = true });
    const memory = b.createModule(.{ .root_source_file = b.path("zig/nemotron_snapshot_memory.zig"), .target = host });
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/nemotron_warm_state_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{
        .{ .name = "lanes", .module = lanes },
        .{ .name = "metal", .module = memory },
        .{ .name = "living", .module = living },
        .{ .name = "kernel_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") }) },
    } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Nemotron warm state:"} }));
    run.has_side_effects = true;
    b.step("test-nemotron-warm-state-host", "Retain an unmarked landed Nemotron prefix on yield and restore it after release").dependOn(&run.step);
    parent.dependOn(&run.step);
}
