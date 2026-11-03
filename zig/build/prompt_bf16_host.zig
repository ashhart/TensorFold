//! CPU-only dispatch and contract registration checks for opt-in BF16 prompt dispatch.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step, debug_cuda: *std.Build.Module, fast_cuda: *std.Build.Module, core: *std.Build.Module) void {
    const step = b.step("test-prompt-bf16-host", "BF16 prompt tile dispatch and old-path controls without devices");
    for ([_]*std.Build.Module{ debug_cuda, fast_cuda }, [_]std.builtin.OptimizeMode{ .debug, .fast }) |cuda, mode| {
        const dispatch = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"BF16 prompt"} }));
        dispatch.has_side_effects = true;
        step.dependOn(&dispatch.step);
        const fixture = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/contract_prompt_bf16_host_test.zig"),
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
