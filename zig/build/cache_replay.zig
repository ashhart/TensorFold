//! Host-only prompt snapshot eviction measurement; it neither imports a device backend nor changes the engine.
const std = @import("std");
pub fn steps(b: *std.Build) void {
    const target = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const cache = b.createModule(.{ .root_source_file = b.path("zig/src/core/prompt_cache.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const imports: []const std.Build.Module.Import = &.{.{ .name = "cache", .module = cache }};
    const tool = b.addExecutable(.{ .name = "tf-cache-replay", .root_module = b.createModule(.{ .root_source_file = b.path("zig/cache_replay.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = imports }) });
    b.step("compile-cache-replay-host", "Compile the host-only snapshot eviction replay").dependOn(&b.addInstallArtifact(tool, .{}).step);
    const replay = b.addRunArtifact(tool);
    replay.addPassthruArgs();
    replay.has_side_effects = true;
    b.step("cache-replay", "Replay synthetic agent traces on main, saved-time-per-byte and recency eviction").dependOn(&replay.step);
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/cache_replay_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = imports }), .filters = &.{"cache replay:"} });
    b.step("test-cache-replay-host", "Replay equivalence, ownership, accounting and selector witnesses without devices").dependOn(&b.addRunArtifact(tests).step);
    const existing = b.addTest(.{ .root_module = cache });
    b.step("test-prompt-cache-host", "The snapshot store's existing host suite without a device backend").dependOn(&b.addRunArtifact(existing).step);
}
