//! tf-qwen3-asr and tf-qwen3-asr-forced: Qwen3-ASR's tower and language model on CUDA against a reference case.
const std = @import("std");

pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, tokenizer: *std.Build.Module, strip: ?bool) void {
    // the Qwen3 layers: qwen3's CUDA pass, the one the embedding engine runs
    const pass = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3/pass_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const family = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_asr/qwen3_asr_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "qwen3_pass", .module = pass } } });
    const main = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen3_asr_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen3_asr", .module = family }, .{ .name = "tokenizer", .module = tokenizer } } });
    const exe = b.addExecutable(.{ .name = "tf-qwen3-asr", .root_module = main });
    b.installArtifact(exe);
    b.step("tf-qwen3-asr", "Build tf-qwen3-asr, Qwen3-ASR's CUDA tower and language model on a clip").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const forced = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen3_asr_forced.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen3_asr", .module = family } } });
    const forced_exe = b.addExecutable(.{ .name = "tf-qwen3-asr-forced", .root_module = forced });
    b.installArtifact(forced_exe);
    b.step("tf-qwen3-asr-forced", "Build tf-qwen3-asr-forced, a clip teacher-forced on a case under the tie rule").dependOn(&b.addInstallArtifact(forced_exe, .{}).step);
}
