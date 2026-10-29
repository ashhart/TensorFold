//! The shared affine reader's geometry checks, expert-plane conversion and exact cuts, without a GPU or model download.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const fast = mode == .fast;
        const name = if (fast) "test-quant-geometry-fast" else "test-quant-geometry-host";
        const about = "Affine geometry, routed and shared expert planes by their own dtype, and bit-exact cuts";
        const root = b.createModule(.{
            .root_source_file = b.path("zig/src/core/quant_geometry_host.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .link_libc = true,
        });
        const step = b.step(name, about);
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
        if (!fast) all.dependOn(step);
    }
}
