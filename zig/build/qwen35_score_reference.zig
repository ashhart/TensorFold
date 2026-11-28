//! Real native rows for the independent BF16/FP64 scoring reference, never a serving worker.
const std = @import("std");
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module) void {
    addTool(b, target, optimize, cuda, qwen, "score");
    addTool(b, target, optimize, cuda, qwen, "hidden");
}
fn addTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module, comptime kind: []const u8) void {
    const exe = b.addExecutable(.{ .name = "tf-qwen35-" ++ kind ++ "-reference", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen35_" ++ kind ++ "_reference.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen } },
    }) });
    b.step("qwen35-" ++ kind ++ "-reference", "Capture native BF16 " ++ kind ++ " rows for independent validation").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
