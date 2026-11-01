//! MiniMax-M3.1-Flash's CUDA backend in the build: its module, its reference-check tool and its host regressions.
const std = @import("std");

/// The MiniMax-M3.1-Flash CUDA family over the runtime, the core and the lane core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/minimax_m3/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    return m;
}

/// The family in tf-cuda-test's `runner`, and tf-minimax-cuda (the forward's rows and logits for the torch reference).
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, runner: *std.Build.Module) void {
    runner.addImport("minimax_m3_cuda", module(b, target, optimize, cuda, core, lanes));
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/minimax_m3/cuda_dump_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    m.addImport("cuda", cuda);
    m.addImport("minimax_m3_cuda", module(b, target, optimize, cuda, core, lanes));
    const exe = b.addExecutable(.{ .name = "tf-minimax-cuda", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-minimax-cuda", "MiniMax-M3.1-Flash's CUDA forward rows for the torch reference").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The family's host regressions (shares, EXL3 cuts and admission, config, kernel geometry, the link header) in `all`.
pub fn hostTests(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, lanes) }));
    b.step("test-minimax-cuda-host", "MiniMax-M3.1-Flash's CUDA backend host regressions (no GPU work)").dependOn(&run.step);
    all.dependOn(&run.step);
}
