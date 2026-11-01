//! Idol on CUDA in the build: its kernels for tf-cuda-test's contract, tf-idol and tf-idol-vae, its host tests.
const std = @import("std");

/// The family's module over the runtime and the core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// The launchers alone (idol.cu's kernels), as tf-cuda-test's contract imports them.
fn launchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The image encoder's launchers alone (idol_vae.cu's kernels), as tf-cuda-test's contract imports them.
fn vaeLaunchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda_vae_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The launchers in tf-cuda-test's `runner` (idol-contract, idol-vae-contract); tf-idol and tf-idol-vae on CUDA.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, core: *std.Build.Module, runner: *std.Build.Module) void {
    runner.addImport("idol_cuda", launchers(b, target, optimize, cuda));
    runner.addImport("idol_vae_cuda", vaeLaunchers(b, target, optimize, cuda));
    const family = module(b, target, optimize, cuda, core);
    const tools = [_]struct { []const u8, []const u8, []const u8, []const u8 }{
        .{ "tf-idol", "zig/tests/idol_cuda.zig", "idol-cuda", "Build tf-idol: Idol's denoiser on a reference case on CUDA" },
        .{ "tf-idol-vae", "zig/tests/idol_vae_cuda.zig", "idol-vae-cuda", "Build tf-idol-vae: Idol's image encoder on a reference case on CUDA" },
    };
    for (tools) |t| {
        const tool = b.createModule(.{ .root_source_file = b.path(t[1]), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol", .module = family } } });
        const exe = b.addExecutable(.{ .name = t[0], .root_module = tool });
        b.installArtifact(exe);
        b.step(t[2], t[3]).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}

/// The family's host tests (config, schedule, scratch, admission, launch geometry, contract ranges) in `all`.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core) }));
    const refs = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_idol_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol_cuda", .module = launchers(b, host, .debug, cuda) }, .{ .name = "idol_vae_cuda", .module = vaeLaunchers(b, host, .debug, cuda) } } });
    const ref_run = b.addRunArtifact(b.addTest(.{ .root_module = refs }));
    const step = b.step("test-idol-cuda-host", "Idol's CUDA denoiser and image encoder host tests and contract references (no GPU work)");
    for ([_]*std.Build.Step{ &run.step, &ref_run.step }) |s| {
        step.dependOn(s);
        all.dependOn(s);
    }
}
