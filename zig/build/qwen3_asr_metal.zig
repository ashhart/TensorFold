//! macOS: Qwen3-ASR's metallib, `zig build tf-qwen3-asr-metal` (tower and Qwen3 LM on a clip), its compile check.
const std = @import("std");

const embedded = @import("metal_targets.zig").embedded;

/// qwen3_asr.metal with safe math and precise fp32 functions.
pub fn metallib(b: *std.Build, metal_std: []const u8) std.Build.LazyPath {
    const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise", "-c" });
    cc.addFileArg(b.path("zig/kernels/metal/qwen3_asr/qwen3_asr.metal"));
    cc.addArg("-o");
    const air = cc.addOutputFileArg("qwen3_asr.air");
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const lib = link.addOutputFileArg("qwen3_asr.metallib");
    link.addFileArg(air);
    return lib;
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, lib: std.Build.LazyPath, no_compiler: ?*std.Build.Step) void {
    const bytes = embedded(b, lib, "qwen3_asr.metallib");
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    // the core kernels it shares: shared_attention and the Qwen-VL tower's GEMM and GELU, compiled at load
    const frags = b.createModule(.{ .root_source_file = b.path("zig/src/core/frags.zig"), .target = target, .optimize = optimize, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const attention = b.createModule(.{ .root_source_file = b.path("zig/src/core/shared_attention.zig"), .target = target, .optimize = optimize, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const family = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_asr/qwen3_asr_metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "frags", .module = frags }, .{ .name = "shared_attention", .module = attention } } });
    const main = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen3_asr_metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "qwen3_asr", .module = family }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "qwen3_asr_metallib", .module = bytes } } });
    const exe = b.addExecutable(.{ .name = "tf-qwen3-asr-metal", .root_module = main });
    const run = b.step("tf-qwen3-asr-metal", "Build tf-qwen3-asr-metal, Qwen3-ASR's Metal tower and language model on a clip");
    run.dependOn(&b.addInstallArtifact(exe, .{}).step);
    if (no_compiler) |fail| run.dependOn(fail);
    const check = b.addObject(.{ .name = "qwen3-asr-metal-host", .root_module = main });
    b.step("compile-qwen3-asr-metal-host", "Compile Qwen3-ASR's Metal binding and driver without running shaders").dependOn(&check.step);
}
