const std = @import("std");
pub fn steps(b: *std.Build) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/copy_source.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const test_bin = b.addTest(.{ .root_module = root });
    b.step("test-cuda-copy-source", "CUDA copy-source ownership without a driver or GPU").dependOn(&b.addRunArtifact(test_bin).step);
}
