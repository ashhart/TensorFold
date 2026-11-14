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
    // its shares as mesh declarations: a test-only module, so no binary roots canon or mesh twice
    const mesh = @import("mesh.zig").module(b, host, .debug, @import("cuda_engines.zig").canonModule(b, host, .debug));
    const decl = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/minimax_m3/cuda_shard_decl_test.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "mesh", .module = mesh }} }) }));
    run.step.dependOn(&decl.step);
    // the Metal backend's timed rounds as lane facts: no device code, so it runs beside the family's other host tests
    const costs = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/minimax_m3/round_costs.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "lanes", .module = lanes }} }) }));
    b.step("test-minimax-m3-costs-host", "MiniMax-M3 Metal's timed round widths as the lanes price them, no GPU").dependOn(&costs.step);
    all.dependOn(&costs.step);
}
