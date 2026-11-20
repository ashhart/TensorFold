//! The checkpoint I/O regressions alone, Debug and ReleaseFast, for -Dtarget; `test` runs them in their full binaries.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const name = if (mode == .fast) "test-checkpoint-io-fast" else "test-checkpoint-io-host";
        const step = b.step(name, "Header-only checkpoint admission, Qwen shard release and header checks without a GPU");
        const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = mode, .link_libc = true });
        core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
        core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = mode, .link_libc = true }));
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"checkpoint admission:"} })).step);
        if (mode == .debug) b.step("test-modelopt-part-host", "The shared ModelOpt part reader (core/modelopt_part.zig) without a GPU")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"part:"} })).step);
        if (mode == .debug) b.step("test-fp8-block-host", "The shared block-FP8 reader and exact scale widening (core/quant/fp8.zig)")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"fp8 block:"} })).step);
        if (mode == .debug) b.step("test-gptq-reader-host", "The shared GPTQ reader on a synthetic checkpoint, against binary64 (core/quant/gptq.zig)")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"gptq:"} })).step);
        if (mode == .debug) b.step("test-artifact-resolution-host", "Pinned target/proposal tuple resolution before model replacement")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"artifact resolution"} })).step);
        if (mode == .debug) b.step("test-expert-stripes-host", "Multi-disk expert stripes, exact fallback and restart accounting without a GPU")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"expert stripe"} })).step);
        if (mode == .debug) b.step("test-deployment-fit-host", "Exact low-memory resident/SSD placement and bounded fit controls")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"deployment fit"} })).step);
        if (mode == .debug) b.step("test-low-memory-plan-host", "Measured-byte placement, exact fallback and transactional admission without devices")
            .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/low_memory_plan.zig"), .target = target, .optimize = mode }) })).step);
        const qwen = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/loader.zig"), .target = target, .optimize = mode, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen, .filters = &.{"Qwen shard integrity:"} })).step);
    }
}
