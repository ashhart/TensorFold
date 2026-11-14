//! test-host-draw-timing: the Qwen host draw timed a 248,320-logit row, greedy, masked and sampled, ReleaseFast.
const std = @import("std");

pub fn step(b: *std.Build) void {
    const host = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .fast, .link_libc = true });
    const timing = b.createModule(.{
        .root_source_file = b.path("zig/src/host_draw_timing_test.zig"),
        .target = host,
        .optimize = .fast,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = timing, .filters = &.{"host draw timing:"} }));
    run.has_side_effects = true; // timings: run every time, never from the cache
    const about = "Time the Qwen host draw over 248,320-logit rows: greedy, masked and sampled (ReleaseFast, no GPU)";
    b.step("test-host-draw-timing", about).dependOn(&run.step);
}
