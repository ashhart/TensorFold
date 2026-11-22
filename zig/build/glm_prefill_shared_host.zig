//! Narrow recording and strict environment regressions for the GLM prompt opt-in.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/glm_prefill_shared_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    const record = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"GLM shared prefill:"} }));
    b.step("test-glm-prefill-shared-host", "Actual shared expert and dense prompt launches, settings and rank agreement without devices").dependOn(&record.step);
    all.dependOn(&record.step);
    const env = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host });
    const strict = b.addRunArtifact(b.addTest(.{ .root_module = env }));
    b.step("test-glm-prefill-shared-contract-host", "Full strict environment registry and counts without devices").dependOn(&strict.step);
    all.dependOn(&strict.step);
}
