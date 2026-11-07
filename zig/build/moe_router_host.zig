//! Qwen router grid dispatch and FP64 cancellation witnesses without a CUDA device.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step, debug: *std.Build.Module, fast: *std.Build.Module, core: *std.Build.Module) void {
    const step = b.step("test-moe-router-host", "Expert-first router ABI, old-path switch and independent reference fixtures");
    for ([_]*std.Build.Module{ debug, fast }, [_]std.builtin.OptimizeMode{ .debug, .fast }) |cuda, mode| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"MoE router port:"} }));
        run.has_side_effects = true;
        step.dependOn(&run.step);
        const fixture = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/contract_moe_router_host_test.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
        }) }));
        fixture.has_side_effects = true;
        step.dependOn(&fixture.step);
    }
    all.dependOn(step);
}
