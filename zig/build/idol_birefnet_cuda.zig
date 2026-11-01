//! Idol's CUDA background remover in the build: its kernels, tf-idol-birefnet on CUDA and its host tests.
const std = @import("std");

/// The family's module over the runtime: BiRefNet's forward, its weights, and the shared host steps.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/idol_birefnet_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The launchers alone (idol_birefnet.cu's kernels), as tf-cuda-test's contract imports them.
fn launchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda_birefnet_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The launchers in tf-cuda-test's `runner` (idol-birefnet-contract), and tf-idol-birefnet on the first GPU.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, runner: *std.Build.Module) void {
    runner.addImport("idol_birefnet_cuda", launchers(b, target, optimize, cuda));
    const tool = b.createModule(.{ .root_source_file = b.path("zig/tests/idol_birefnet_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "idol_birefnet", .module = module(b, target, optimize, cuda) } } });
    const exe = b.addExecutable(.{ .name = "tf-idol-birefnet", .root_module = tool });
    b.installArtifact(exe);
    b.step("idol-birefnet-cuda", "Build tf-idol-birefnet: Idol's background removal on a reference case on CUDA").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The family's host tests (geometry, scratch, admission, launch geometry, host steps, contract references) in `all`.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda) }));
    const refs = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_birefnet_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol_birefnet_cuda", .module = launchers(b, host, .debug, cuda) } } });
    const ref_run = b.addRunArtifact(b.addTest(.{ .root_module = refs }));
    const step = b.step("test-idol-birefnet-cuda-host", "Idol's CUDA background remover host tests and contract references (no GPU work)");
    for ([_]*std.Build.Step{ &run.step, &ref_run.step }) |s| {
        step.dependOn(s);
        all.dependOn(s);
    }
}
