//! GLM contract fixtures and loader controls that run without a GPU or kernel images.
const std = @import("std");

pub fn host(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_contract_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    b.step("test-glm-host-key-pages", "MTP host latent and device key strides through real dispatch without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests.root_module, .filters = &.{"GLM host key pages:"} })).step);
    const run = b.addRunArtifact(tests);
    const step = b.step("test-glm-contract-host", "GLM contract fixtures, frozen exports and clean-before-tamper controls");
    step.dependOn(&run.step);
    all.dependOn(step);
}
