//! Qwen3 dense on CUDA in the build: the GPU check the box runs, and its GPU-free gates in `test`.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const runtime = @import("cuda_runtime.zig").runtime;
const options = @import("options.zig");

/// The prebuilt fatbins in `dir` (-Dfatbins); without it a check compiles with empty images and cannot run on a device.
fn images(b: *std.Build, dir: ?[]const u8) [catalog.kernels.len]?std.Build.LazyPath {
    var out: [catalog.kernels.len]?std.Build.LazyPath = @splat(null);
    if (dir) |d| {
        for (catalog.kernels, &out) |k, *image| image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ d, b.fmt("{s}.fatbin", .{k.name}) }));
    }
    return out;
}

/// `qwen3-cuda-gpu-build` and `qwen3-cuda-gpu-test`, and the checks no device is needed for.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, o: options.Cuda, all: *std.Build.Step) void {
    const cuda = runtime(b, target, optimize, &images(b, o.fatbins));
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3/cuda_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    const run = b.addRunArtifact(gpu);
    run.has_side_effects = true; // the checkpoints and oracle files named by TF_QWEN3_* are not cached inputs
    b.step("qwen3-cuda-gpu-build", "Compile the Qwen3 dense CUDA checks without running them").dependOn(&gpu.step);
    b.step("qwen3-cuda-gpu-test", "Qwen3 embedding vectors on CUDA against transformers (TF_QWEN3_EMBED_DIR, _REF, _BAR)").dependOn(&run.step);
    all.dependOn(&gpu.step);
    const host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3/checks.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true }) }));
    b.step("test-qwen3-checks-host", "Qwen3 checks' oracle pins, references and batching without a GPU").dependOn(&host.step);
    all.dependOn(&host.step);
}
