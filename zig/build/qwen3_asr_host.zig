//! Qwen3-ASR's host parts: config, tensors, mel rows and the float64 tower, no GPU.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.graph.host;
    const family = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_asr/qwen3_asr.zig"), .target = host, .link_libc = true });
    const checks = b.step("test-qwen3-asr-host", "Qwen3-ASR config, tensor names, mel rows and reference ops");
    checks.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
    all.dependOn(checks);
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = host, .link_libc = true });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .link_libc = true });
    const tool = b.addExecutable(.{ .name = "tf-qwen3-asr-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen3_asr_check.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{ .{ .name = "qwen3_asr", .module = family }, .{ .name = "safetensors", .module = st }, .{ .name = "tokenizer", .module = tokenizer } },
    }) });
    b.installArtifact(tool);
    b.step("tf-qwen3-asr-check", "Check a Qwen3-ASR checkpoint and its float64 tower against a tools/zig/qwen3_asr_case.py case")
        .dependOn(&b.addInstallArtifact(tool, .{}).step);
}
