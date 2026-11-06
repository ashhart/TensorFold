//! Experimental DS4.1 weight-loading qualification targets.
const std = @import("std");

pub fn weights(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    core: *std.Build.Module,
) *std.Build.Step {
    const host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_weights_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "core", .module = core },
        },
    }) });
    const step = b.step("test-deepseek-v41-weights", "V4.1 configuration, Engram table plans and resident weight staging");
    step.dependOn(&b.addRunArtifact(host).step);
    return step;
}

pub fn checks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    sources: *std.Build.Module,
    engine: *std.Build.Module,
    core: *std.Build.Module,
) struct { device: *std.Build.Step, reference: *std.Build.Step } {
    const device = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "core", .module = core },
        },
    }) });
    const device_step = b.step("test-deepseek-v41", "Experimental DS4.1 EXL3, Engram, mHC and sparse attention on Metal");
    device_step.dependOn(&b.addRunArtifact(device).step);
    // A one-test slice of the same module: the index select's determinism probe, so a kernel-level
    // race can be iterated in seconds instead of a model boot.
    const probe = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "core", .module = core },
        },
    }), .filters = &.{"index one-shot select"} });
    const probe_run = b.addRunArtifact(probe);
    b.step("test-deepseek-v41-index-probe", "The index one-shot select's same-input determinism, one test").dependOn(&probe_run.step);
    const reference = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/deepseek_v41_reference_checks.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "tensorfold", .module = engine },
                .{ .name = "metal", .module = metal },
                .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, optimize) },
            },
        }),
        .filters = &.{"tests.deepseek_v41_reference."},
    });
    const reference_step = b.step("test-deepseek-v41-reference", "Independent CPU teacher-oracle arithmetic boundaries");
    reference_step.dependOn(&b.addRunArtifact(reference).step);
    return .{ .device = device_step, .reference = reference_step };
}
