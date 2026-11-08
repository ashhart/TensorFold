//! A source-only Metal compiler gate, without a shader build dependency or GPU dispatch.
const std = @import("std");

pub fn steps(b: *std.Build, metal: *std.Build.Module, sources: *std.Build.Module) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/nemotron_prefill_sources_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    const tests = b.addTest(.{ .root_module = root, .filters = &.{"Metal source compile gate:"} });
    const run = b.addRunArtifact(tests);
    run.has_side_effects = true;
    b.step("test-nemotron-prefill-metal-sources", "Compile Nemotron base and masked prompt libraries; needs a device, no dispatch").dependOn(&run.step);
}
