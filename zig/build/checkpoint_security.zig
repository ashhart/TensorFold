//! Checkpoint security regressions run without a GPU or model download.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-checkpoint-security", "Host checkpoint index, write containment and parser regressions");
    all.dependOn(step);
    const checkpoint = b.createModule(.{
        .root_source_file = b.path("zig/src/core/checkpoint_security_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = checkpoint }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
    const safetensors = b.createModule(.{
        .root_source_file = b.path("zig/src/core/safetensors.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    const kimi = b.createModule(.{
        .root_source_file = b.path("zig/src/families/kimi_k3/shard_header.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .imports = &.{.{ .name = "safetensors", .module = safetensors }},
    });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = kimi })).step);
    const npy = b.createModule(.{
        .root_source_file = b.path("zig/src/core/npy.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = npy })).step);
}
