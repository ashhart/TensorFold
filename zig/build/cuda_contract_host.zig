//! CPU references for the CUDA contracts: Flash Next writeback, utility, row-best and QSA, rows, windows and FP64.
const std = @import("std");

/// Each contract reference joins test-cuda-host and its own step; none needs a GPU.
pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, step: *std.Build.Step) *std.Build.Step {
    const glm_fact = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_nvfp4_glm_host.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) }));
    b.step("test-glm-fact-contract-host", "Actual factored-expert contract envelope at GLM shapes without devices").dependOn(&glm_fact.step);
    step.dependOn(&glm_fact.step);
    const writeback_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_flashnext_writeback.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) }));
    step.dependOn(&writeback_contract.step);
    @import("cuda_row_contracts.zig").host(b, host, cuda, mods.core, step);
    @import("cuda_window_host.zig").steps(b, host, step);
    const utility_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_utility_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&utility_contract.step);
    const flash_contract = b.step("test-flashnext-contract-host", "Flash Next contract stress and rounding references without a GPU");
    flash_contract.dependOn(&writeback_contract.step);
    flash_contract.dependOn(&utility_contract.step);
    const row_best_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_row_best_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&row_best_contract.step);
    flash_contract.dependOn(&row_best_contract.step);
    const qsa_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_qsa_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&qsa_contract.step);
    flash_contract.dependOn(&qsa_contract.step);
    @import("cuda_window_host.zig").contracts(b, host, cuda, mods.core, step);
    const reference_step = b.step("test-cuda-reference", "CUDA's FP64 comparison with CPU-only readback (no GPU)");
    const projection_step = b.step("test-cuda-projection-fixture", "Projection fixture dispatch controls with CPU-only CUDA calls");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const reference = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/quant_ref_host_tests.zig"),
            .target = host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
        }) }));
        step.dependOn(&reference.step);
        reference_step.dependOn(&reference.step);
        const projection = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/projection_f32_host_tests.zig"),
            .target = host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
        }) }));
        step.dependOn(&projection.step);
        projection_step.dependOn(&projection.step);
    }
    return flash_contract;
}
