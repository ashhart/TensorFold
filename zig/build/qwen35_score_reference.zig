//! Real native rows for the independent BF16/FP64 scoring reference, never a serving worker.
const std = @import("std");
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-qwen35-score-reference", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen35_score_reference.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen } },
    }) });
    b.step("qwen35-score-reference", "Capture native BF16 prompt/decode scoring rows for independent FP64 validation").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
