//! `zig build test-device`: tensorfold device's scan parsers, support table, profile and summary, with no GPU.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/device/tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
    }) });
    const run = b.addRunArtifact(tests);
    run.setCwd(b.path("."));
    run.has_side_effects = true; // it writes a profile into a temporary folder and reads this host's own listing
    b.step("test-device", "tensorfold device: scan parsers, support table, profile and summary, no GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
