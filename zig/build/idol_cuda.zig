//! Idol on CUDA in the build: denoiser and DINOv3 kernels for the contract, tf-idol and tf-idol-dino, host tests.
const std = @import("std");

/// The family's module over the runtime and the core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// The launchers alone (idol.cu's kernels), as tf-cuda-test's contract imports them.
fn launchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The DINOv3 encoder's launchers alone (idol_dino.cu's kernels), as tf-cuda-test's contract imports them.
fn dinoLaunchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/dino_cuda_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// The launchers in tf-cuda-test's `runner` (idol-contract, idol-dino-contract); tf-idol and tf-idol-dino on CUDA.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, core: *std.Build.Module, runner: *std.Build.Module) void {
    runner.addImport("idol_cuda", launchers(b, target, optimize, cuda));
    runner.addImport("idol_dino_cuda", dinoLaunchers(b, target, optimize, cuda));
    const family = module(b, target, optimize, cuda, core);
    const tools = [_]struct { []const u8, []const u8, []const u8, []const u8 }{
        .{ "zig/tests/idol_cuda.zig", "tf-idol", "idol-cuda", "Build tf-idol: Idol's denoiser on a reference case on CUDA" },
        .{ "zig/tests/idol_dino_cuda.zig", "tf-idol-dino", "idol-dino-cuda", "Build tf-idol-dino: Idol's DINOv3 encoder on a reference case on CUDA" },
    };
    for (tools) |t| {
        const tool = b.createModule(.{ .root_source_file = b.path(t[0]), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol", .module = family } } });
        const exe = b.addExecutable(.{ .name = t[1], .root_module = tool });
        b.installArtifact(exe);
        b.step(t[2], t[3]).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}

/// The family's host tests (config, scratch, admission, launch geometry, contract ranges; DINOv3's too) in `all`.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core) }));
    const refs = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_idol_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol_cuda", .module = launchers(b, host, .debug, cuda) }, .{ .name = "idol_dino_cuda", .module = dinoLaunchers(b, host, .debug, cuda) } } });
    const ref_run = b.addRunArtifact(b.addTest(.{ .root_module = refs }));
    const step = b.step("test-idol-cuda-host", "Idol's CUDA denoiser and DINOv3 host tests and contract references (no GPU work)");
    for ([_]*std.Build.Step{ &run.step, &ref_run.step }) |s| {
        step.dependOn(s);
        all.dependOn(s);
    }
}
