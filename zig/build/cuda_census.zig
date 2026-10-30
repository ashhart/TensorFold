//! The Linux compile census: the real checkers compiled for Linux, without images or execution.
const std = @import("std");

pub fn steps(b: *std.Build, cuda_target: std.Build.ResolvedTarget, check_cuda: *std.Build.Module, qwen35: *std.Build.Module, core: *std.Build.Module, flashnext: *std.Build.Module) void {
    const rounds_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen35_cuda_main.zig"), .target = cuda_target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = check_cuda }, .{ .name = "qwen35", .module = qwen35 }, .{ .name = "core", .module = core } } });
    const rounds_compile = b.addObject(.{ .name = "qwen35-rounds-checker-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen35_rounds_compile.zig"), .target = cuda_target, .link_libc = true, .imports = &.{.{ .name = "tool", .module = rounds_tool }} }) });
    _ = rounds_compile.getEmittedBin();
    b.step("compile-qwen35-rounds-host", "Compile the real CUDA 64-row checker for Linux, no images or execution").dependOn(&rounds_compile.step);
    const parity_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext_mtp_parity_main.zig"), .target = cuda_target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = check_cuda }, .{ .name = "flashnext", .module = flashnext } } });
    const parity_compile = b.addObject(.{ .name = "flashnext-mtp-parity-compile", .root_module = parity_tool });
    _ = parity_compile.getEmittedBin();
    b.step("compile-flashnext-mtp-parity-host", "Compile the real Flash Next MTP parity checker for Linux, no images or execution").dependOn(&parity_compile.step);
}
