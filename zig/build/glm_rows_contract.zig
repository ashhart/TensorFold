//! Host registration/fixture proofs and recording encoders for the GLM Metal window contract.
const std = @import("std");
pub fn host(b: *std.Build, sources: *std.Build.Module) *std.Build.Step {
    const filter = b.option([]const u8, "glm-rows-contract-filter", "One GLM window host regression") orelse "GLM window contract:";
    const fake = b.createModule(.{ .root_source_file = b.path("zig/glm_rows_recording_metal.zig"), .target = b.graph.host });
    const family = b.createModule(.{ .root_source_file = b.path("zig/glm_rows_contract_family.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "kernel_sources", .module = sources } } });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/glm_rows_contract_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "tensorfold", .module = family }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{filter} }));
    run.has_side_effects = true;
    const step = b.step("test-glm-window-contract-host", "Register and validate actual GLM replacement operators without a GPU");
    step.dependOn(&run.step);
    return step;
}
