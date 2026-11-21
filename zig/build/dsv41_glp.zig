//! Host-only steps for the GLPv3 additive CUDA primitives; the device proof is `tf-dsv41-test glp <fatbin dir>`.

// Format and paired site: Matt Suiche/weightless, spec b181a07d3980fc48abe39d95ddb044459dd0b868.
const std = @import("std");
// The additive convention is llama.cpp's; Arditi et al. (NeurIPS 2024) is background only.
const family = @import("dsv41_cuda.zig");

/// The aggregate `test-dsv41-glp-host` step, returned so the caller can also hang it on a larger host step.
pub fn steps(b: *std.Build) *std.Build.Step {
    const step = b.step("test-dsv41-glp-host", "GLPv3 additive plans, launch checks and contract models without device work");
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
    return step;
}
