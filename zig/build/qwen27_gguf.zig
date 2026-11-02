//! Qwen35 GGUF-to-native host checks, including atomic prepared images and borrowed checkpoint tensors.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.step("test-qwen27-gguf-host", "Qwen35 native GGUF images and their cache without a GPU");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true });
        const root = @import("server.zig").plain(b, "zig/qwen27_gguf_host.zig", b.graph.host, mode, lanes);
        root.addImport("core", root);
        host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"GGUF Qwen:"} })).step);
    }
    all.dependOn(host);
}
