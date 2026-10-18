//! Release-server refusal controls use the real admission and a private environment provider.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step, api: *std.Build.Module, nemotron: *std.Build.Module) void {
    const env = b.createModule(.{ .root_source_file = b.path("zig/cuda_release_refusal_env.zig"), .target = b.graph.host });
    const refusal = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_refusal.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "sys", .module = env } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/cuda_release_refusal_tests.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "release_refusal", .module = refusal }, .{ .name = "policy_env", .module = env } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"DeepSeek release:"} }));
    b.step("test-release-cuda-refusal-host", "Release DeepSeek refusal across controlled opt-in values without device work").dependOn(&run.step);
    all.dependOn(&run.step);
}
