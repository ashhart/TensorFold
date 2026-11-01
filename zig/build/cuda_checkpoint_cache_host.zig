//! Checkpoint load cleanup and actual family load scopes, with no GPU dependency.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const step = b.step("test-cuda-load-cache-host", "Checkpoint cleanup and family load scopes without a GPU");
    for ([_][]const u8{ "zig/cuda_checkpoint_cache_host.zig", "zig/src/core/checkpoint_cache.zig" }) |path| {
        const m = b.createModule(.{ .root_source_file = b.path(path), .target = b.graph.host, .link_libc = true });
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
    all.dependOn(step);
}
