//! Keyed sampling on the GPU path: `test-dsv41-sampling` on in-process ranks, `tf-dsv41-samp` on one GPU (Linux).

const std = @import("std");
const tp_build = @import("tp.zig");
const dsv41_build = @import("dsv41.zig");
const server_build = @import("server.zig");

/// The CUDA runtime without the engine's kernels (tp's in-process ranks need only its types).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The family's serving module with its own `json` and `tokenizer`, over `lanes`.
fn serve(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lanes: *std.Build.Module) struct { serve: *std.Build.Module, json: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const s = server_build.family(b, target, optimize, tokenizer, lanes);
    return .{ .serve = s.serve, .json = s.json };
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const s = serve(b, host, .debug, lanes);
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/sampling_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tp", .module = tp_build.module(b, host, .debug, runtime(b, host, .debug)) },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "dsv41_serve", .module = s.serve },
            .{ .name = "json", .module = s.json },
        },
    });
    const step = b.step("test-dsv41-sampling", "DeepSeek-V4.1's keyed sampling pipeline on in-process ranks vs the Python engine's choices (no GPU)");
    // the pipeline over the golden (the serving module's own tests run in test-dsv41-serve)
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    test_step.dependOn(step);
    // tf-dsv41-samp compiles on any host; running it needs a GPU and the sampling image
    const rt = runtime(b, host, .debug);
    test_step.dependOn(&exe(b, host, .debug, rt, dsv41_build.kernels(b, host, .debug, rt)).step);
}

/// `tf-dsv41-samp` (Linux): the device steps on one GPU against the host reference, over cuda.zig's modules.
pub fn gpu(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, kernels: *std.Build.Module) void {
    b.installArtifact(exe(b, target, optimize, cuda, kernels));
}

fn exe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, kernels: *std.Build.Module) *std.Build.Step.Compile {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/sampling_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "dsv41_kernels", .module = kernels },
            .{ .name = "tp", .module = tp_build.module(b, target, optimize, cuda) },
            .{ .name = "dsv41_serve", .module = serve(b, target, optimize, lanes).serve },
        },
    });
    return b.addExecutable(.{ .name = "tf-dsv41-samp", .root_module = m });
}
