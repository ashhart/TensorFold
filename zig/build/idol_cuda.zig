//! Idol's CUDA denoiser in the build: its kernels for tf-cuda-test's contract, tf-idol on CUDA, its host tests.
const std = @import("std");

/// The family's module over the runtime and the core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// The launchers alone (idol.cu's kernels), as tf-cuda-test's contract imports them.
fn launchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The launchers in tf-cuda-test's `runner` (idol-contract), and tf-idol: the case driver on the first GPU.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, core: *std.Build.Module, runner: *std.Build.Module) void {
    runner.addImport("idol_cuda", launchers(b, target, optimize, cuda));
    const tool = b.createModule(.{ .root_source_file = b.path("zig/tests/idol_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol", .module = module(b, target, optimize, cuda, core) } } });
    const exe = b.addExecutable(.{ .name = "tf-idol", .root_module = tool });
    b.installArtifact(exe);
    b.step("idol-cuda", "Build tf-idol: Idol's denoiser on a reference case on CUDA").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The family's host tests (config, schedule, scratch, admission, launch geometry, contract ranges) in `all`.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core) }));
    const refs = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_idol_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol_cuda", .module = launchers(b, host, .debug, cuda) } } });
    const ref_run = b.addRunArtifact(b.addTest(.{ .root_module = refs }));
    const step = b.step("test-idol-cuda-host", "Idol's CUDA denoiser host tests and contract references (no GPU work)");
    for ([_]*std.Build.Step{ &run.step, &ref_run.step }) |s| {
        step.dependOn(s);
        all.dependOn(s);
    }
}
