//! Host-only persistence and policy checks for Living Weights.
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, server: *std.Build.Module, core: *std.Build.Module, living: *std.Build.Module, api: *std.Build.Module, nemotron: *std.Build.Module, all: *std.Build.Step) void {
    b.step("test-pocket-audit-host", "Read-only pocket audit probes, witnesses and CLI on the host").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "pocket audit", "audit request:", "audit CLI:", "audit probe bank:", "audit endpoint:" } })).step);
    const options = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_options.zig"), .target = host, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = api }} });
    b.step("test-pocket-options-host", "Native firing-count flags without device or model access").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = options, .filters = &.{"fact usage:"} })).step);
    const products = b.addRunArtifact(b.addTest(.{ .root_module = nemotron, .filters = &.{"pocket usage:"} }));
    const masks = b.addRunArtifact(b.addTest(.{ .root_module = api.import_table.get("lanes").?, .filters = &.{"pocket usage:"} }));
    const product_step = b.step("test-pocket-usage-host", "Existing CUDA pocket products and bounded masks without kernels");
    product_step.dependOn(&products.step);
    product_step.dependOn(&masks.step);
    all.dependOn(&masks.step);
    all.dependOn(&products.step);
    const policy = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"fact usage:"} }));
    b.step("test-living-policy-host", "Usage, pinning, full-pocket policy and exact deletion without devices").dependOn(&policy.step);
    const store = b.createModule(.{
        .root_source_file = b.path("zig/src/living_restart_store.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "living", .module = living }, .{ .name = "engine_api", .module = api }, .{ .name = "core", .module = core } },
    });
    const tests = b.createModule(.{ .root_source_file = b.path("zig/src/server/living_restart_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    var deps = server.import_table.iterator();
    while (deps.next()) |entry| tests.addImport(entry.key_ptr.*, entry.value_ptr.*);
    tests.addImport("restart_store", store);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests, .filters = &.{"restart case:"} }));
    b.step("test-living-restart-host", "Fresh learner teaches after reopening real affine shards, retained or empty graph").dependOn(&run.step);
    all.dependOn(&policy.step);
    all.dependOn(&run.step);
}
