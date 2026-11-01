//! Idol on CUDA in the build: each component's kernels for the contract, its CUDA tool, the host tests.
const std = @import("std");

/// The family's module over the runtime and the core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/idol/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// Each component's launchers alone, by the import name tf-cuda-test's contract uses, and their file.
const launcher_files = [_]struct { []const u8, []const u8 }{
    .{ "idol_cuda", "cuda_kernels.zig" }, // idol.cu: the denoiser
    .{ "idol_dino_cuda", "dino_cuda_kernels.zig" }, // idol_dino.cu: the DINOv3 encoder
    .{ "idol_vae_cuda", "cuda_vae_kernels.zig" }, // idol_vae.cu: the VAE image encoder
};

fn launchersOf(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, file: []const u8) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/families/idol/{s}", .{file})), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

fn addLaunchers(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, m: *std.Build.Module) void {
    for (launcher_files) |l| m.addImport(l[0], launchersOf(b, target, optimize, cuda, l[1]));
}

/// The launchers in tf-cuda-test's `runner` (each component's contract); each component's tool on CUDA.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, core: *std.Build.Module, runner: *std.Build.Module) void {
    addLaunchers(b, target, optimize, cuda, runner);
    const family = module(b, target, optimize, cuda, core);
    const tools = [_]struct { []const u8, []const u8, []const u8, []const u8 }{
        .{ "tf-idol", "zig/tests/idol_cuda.zig", "idol-cuda", "Build tf-idol: Idol's denoiser on a reference case on CUDA" },
        .{ "tf-idol-dino", "zig/tests/idol_dino_cuda.zig", "idol-dino-cuda", "Build tf-idol-dino: Idol's DINOv3 encoder on a reference case on CUDA" },
        .{ "tf-idol-vae", "zig/tests/idol_vae_cuda.zig", "idol-vae-cuda", "Build tf-idol-vae: Idol's image encoder on a reference case on CUDA" },
    };
    for (tools) |t| {
        const tool = b.createModule(.{ .root_source_file = b.path(t[1]), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "idol", .module = family } } });
        const exe = b.addExecutable(.{ .name = t[0], .root_module = tool });
        b.installArtifact(exe);
        b.step(t[2], t[3]).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}

/// The family's host tests (config, scratch, admission, launch geometry, each component's contract ranges) in `all`.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core) }));
    const refs = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_idol_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    addLaunchers(b, host, .debug, cuda, refs);
    const ref_run = b.addRunArtifact(b.addTest(.{ .root_module = refs }));
    const step = b.step("test-idol-cuda-host", "Idol's CUDA components' host tests and contract references (no GPU work)");
    for ([_]*std.Build.Step{ &run.step, &ref_run.step }) |s| {
        step.dependOn(s);
        all.dependOn(s);
    }
}
