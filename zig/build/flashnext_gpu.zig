//! The Flash Next HIP GPU checks beside the model-free kernels in `hip-kernel-test`: no model server runs here.
const std = @import("std");
const hydrox = @import("hip.zig");

/// The draft batch's check: two streams' drafts in one head pass against the same two one stream at a time.
pub fn draftBatch(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, gpu: hydrox.Engine) void {
    const draft_gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/flashnext/hip/draft_batch_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = gpu.hip },
            .{ .name = "lanes", .module = gpu.lanes },
            .{ .name = "flashnext", .module = gpu.flashnext },
        },
    }) });
    const draft_run = b.addRunArtifact(draft_gpu);
    draft_run.has_side_effects = true; // the checkpoint TF_FLASHNEXT_DIR names is not a cached input
    b.step("draft-batch-gpu-build", "Compile the draft batch's GPU checks (TF_FLASHNEXT_DIR, TF_FLASHNEXT_PROMPTS)").dependOn(&draft_gpu.step);
    b.top_level_steps.get("hip-kernel-test").?.step.dependOn(&draft_run.step);
}
