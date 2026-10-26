//! CPU-only FSR ownership and file fixtures; no shader compiler or GPU driver.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/core/fsr/tests.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-fsr-host", "Exact FSR cache ownership and bounded reader fixtures without devices").dependOn(&run.step);
    all.dependOn(&run.step);
    const source = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/fsr_source_tests.zig"), .target = b.graph.host, .optimize = .debug }) }));
    b.step("test-fsr-source-host", "Recorded expert source address patch and refusal controls without shaders").dependOn(&source.step);
    all.dependOn(&source.step);
}
