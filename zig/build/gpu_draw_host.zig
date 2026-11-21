//! test-gpu-draw-host: host GPU draws with selection against main's full sort; Debug for picks, ReleaseFast for times.
const std = @import("std");

pub fn step(b: *std.Build) void {
    const host = b.graph.host;
    const s = b.step("test-gpu-draw-host", "Host GPU draws equal main's full sort, and their time a row (no GPU)");
    for ([_]std.builtin.OptimizeMode{ .Debug, .ReleaseFast }) |mode| {
        const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = mode, .link_libc = true });
        const run = b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"gpu draw:"} }));
        run.has_side_effects = mode == .fast; // its times: run every time, never from the cache
        s.dependOn(&run.step);
    }
}
