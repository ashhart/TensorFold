//! BF16 prompt scratch envelopes and contract registrations without a GPU.
const std = @import("std");
const runtime = @import("cuda.zig").runtime;

pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const isolation = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qmmf_fused_split_host.zig"),
        .target = host,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    }) }));
    b.top_level_steps.get("test-flashnext-fused-split-host").?.step.dependOn(&isolation.step);
    const prompt_scratch = b.step("test-qmmf-prompt-scratch-host", "BF16 prompt scratch bounds in Debug and ReleaseFast without a GPU");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |optimize| {
        const tests = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/qmmf_prompt_tests.zig"),
            .target = host,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "cuda", .module = runtime(b, host, optimize, &.{}) }},
        });
        prompt_scratch.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests, .filters = &.{"qmmf prompt scratch"} })).step);
    }
    step.dependOn(prompt_scratch);
    const qmmf_prompt_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_qmmf_prompt.zig"),
        .target = host,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }), .filters = &.{"qmmf prompt contract"} }));
    b.step("test-qmmf-prompt-contract-host", "BF16 K64 and K32 prompt instances, controls and limits without a GPU").dependOn(&qmmf_prompt_contract.step);
    step.dependOn(&qmmf_prompt_contract.step);
    const dots_shapes = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qmmf_dots_prompt_test.zig"),
        .target = host,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    }) }));
    b.step("test-qmmf-dots-prompt-host", "The dots.mocr prompt shapes the dense engine serves stay prompt-eligible without a GPU").dependOn(&dots_shapes.step);
    step.dependOn(&dots_shapes.step);
}
