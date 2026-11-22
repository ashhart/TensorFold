//! Actual GLM dispatch and strict counted environment controls without a device.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/glm_hc_shared_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-glm-hc-shared-host", "Production HC dispatch ABI, byte fixture and corruption controls without CUDA").dependOn(&run.step);
    all.dependOn(&run.step);
    const env = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM strict env:"} }));
    env.has_side_effects = true;
    b.step("test-glm-hc-shared-env-host", "Every counted GLM environment row including the shared HC opt-in").dependOn(&env.step);
    all.dependOn(&env.step);
}
