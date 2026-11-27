//! Focused dense lean prompt recording, environment and compile checks.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "glm", .module = glm }, .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } };
    const root = b.createModule(.{ .root_source_file = b.path("zig/glm_dense_lean_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = imports });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-glm-dense-lean-host", "Actual dense prompt launch descriptors and per-element fragment ownership").dependOn(&run.step);
    all.dependOn(&run.step);
    const env = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM strict env:"} }));
    env.has_side_effects = true;
    b.step("test-glm-dense-lean-env-host", "Strict default-off dense prompt switch through startup and reader").dependOn(&env.step);
    const object = b.addObject(.{ .name = "glm-dense-lean-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_dense_lean_compile.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = imports }) });
    const emitted = b.addInstallFile(object.getEmittedBin(), "checks/glm-dense-lean.o");
    b.step("compile-glm-dense-lean-host", "Analyze dense byte contract and profiler without a device").dependOn(&emitted.step);
}
