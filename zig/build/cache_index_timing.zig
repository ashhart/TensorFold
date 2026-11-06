//! test-cache-index-timing: the prompt cache's snapshot index timed at 16 to 1,024 kept states, ReleaseFast, host only.
const std = @import("std");

pub fn step(b: *std.Build) void {
    const host = b.graph.host;
    if (host.result.os.tag != .macos and host.result.os.tag != .linux) return;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .fast, .link_libc = true });
    const timing = b.createModule(.{
        .root_source_file = b.path("zig/src/core/prompt_cache_index_timing_test.zig"),
        .target = host,
        .optimize = .fast,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = timing, .filters = &.{"snapshot index timing:"} }));
    run.has_side_effects = true; // timings: run every time, never from the cache
    const about = "Time the prompt cache index's find and lookup at 16 to 1,024 kept states (ReleaseFast, no GPU)";
    b.step("test-cache-index-timing", about).dependOn(&run.step);
}
