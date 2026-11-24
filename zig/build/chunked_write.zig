//! Registers the chunked-writer host tests as `test-chunked-write`.
const std = @import("std");

pub fn wire(b: *std.Build, parent: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/chunked_write_test_file.zig"), .target = b.graph.host }) });
    const step = b.step("test-chunked-write", "Bounded chunked output writes: cap respected, order preserved, the old one-shot path fails the enforcing sink");
    step.dependOn(&b.addRunArtifact(tests).step);
    parent.dependOn(step);
}
