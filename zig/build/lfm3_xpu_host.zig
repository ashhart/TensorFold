//! test-lfm3-image-host: #1502's LFM3 image tests (lfm3_image_tests.zig) here, until #1502's own step lands.
const std = @import("std");

pub fn steps(b: *std.Build, checks: *std.Build.Step) void {
    if (b.top_level_steps.contains("test-lfm3-image-host")) return; // #1502's lfm3_host.zig declares it
    const image = b.createModule(.{ .root_source_file = b.path("zig/src/lfm3_image_tests.zig"), .target = b.graph.host, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = image }));
    const step = b.step("test-lfm3-image-host", "LFM3 image geometry and shared decoder checks");
    step.dependOn(&run.step);
    checks.dependOn(step);
}
