//! BF16 prompt scratch envelopes and contract registrations without a GPU.
const std = @import("std");
const runtime = @import("cuda.zig").runtime;

pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
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
    b.step("test-qmmf-prompt-contract-host", "Four BF16 prompt instances and their contract/limit coverage without a GPU").dependOn(&qmmf_prompt_contract.step);
    step.dependOn(&qmmf_prompt_contract.step);
}
