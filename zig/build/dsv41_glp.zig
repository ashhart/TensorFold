//! Host-only steps for the DeepSeek-V4.1 GLPv3 additive CUDA primitives: the immutable plan and its policy words, the
//! launch checks and ABI of glp_ops.zig, the contract's host fixtures and models, and the registry rows. No nvcc,
//! no device: the kernels (zig/kernels/cuda/deepseek_v41/glp.cu, glp_controls.cu) are built by the family's
//! `fatbins-dsv41` / `dsv41-test` steps as the optional images `dsv41_glp_add` and `dsv41_glp_add_controls`
//! (zig/build/dsv41_units.zig) and proven on a device by `tf-dsv41-test glp <fatbin dir>` and
//! `tf-cuda-test kernel-contract glp`.
//!
//! Format and paired-site concept: Matt Suiche/weightless (spec b181a07d3980fc48abe39d95ddb044459dd0b868); the
//! additive convention is llama.cpp's; Arditi et al. (NeurIPS 2024) is background only.

const std = @import("std");
const family = @import("dsv41_cuda.zig");

/// The aggregate `test-dsv41-glp-host` step, returned so the caller can also hang it on a larger host step.
pub fn steps(b: *std.Build) *std.Build.Step {
    const step = b.step("test-dsv41-glp-host", "GLPv3 additive plan, launch checks, host models, contract fixtures and registry rows, no device");
    const plan = family.module(b, b.graph.host, "glp_plan.zig", false);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = plan })).step);
    // glp_ops.zig's launch checks and layout, among the kernel module's own tests
    const kernels = plan.import_table.get("dsv41_kernels").?;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = kernels, .filters = &.{"glp"} })).step);
    // the registered contract's host witnesses: fixtures, models and the planted faults' emulations
    const contract = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_glp.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
    });
    contract.addImport("cuda", plan.import_table.get("cuda").?);
    contract.addImport("core", plan.import_table.get("core").?);
    @import("cuda_contract_deepseek.zig").inert(b, contract);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract })).step);
    // the registry rows: unmeasured launches refuse qualification (the contract module's own tests, by name)
    const registry = @import("kernel_contract.zig").module(b, b.graph.host, .Debug);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = registry, .filters = &.{"glp records"} })).step);
    return step;
}
