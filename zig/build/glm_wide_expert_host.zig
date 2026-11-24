//! GLM wide expert production dispatch and strict counted environment checks without CUDA.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/glm_wide_expert_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-glm-wide-expert-host", "Actual expert down dispatch and byte contract fixtures without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const env = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{ "GLM strict env:", "GLM cost identity:" } }));
    env.has_side_effects = true;
    b.step("test-glm-wide-expert-env-host", "Count every GLM environment row and exercise the wide expert reader").dependOn(&env.step);
    all.dependOn(&env.step);
}
