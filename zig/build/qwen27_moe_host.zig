//! Served admission and the existing device contract references, without a device or shader compiler.
const std = @import("std");
pub fn tests(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_moe_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "metal", .module = metal },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"qwen35 Metal:"} }));
    run.has_side_effects = true;
    const step = b.step("test-qwen35-moe-metal-host", "35B geometry, attention source and routed FP64 references, no GPU");
    step.dependOn(&run.step);
}
