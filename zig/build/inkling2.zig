//! Inkling 2's CUDA family in the build: its module, its check tools and its host regressions.
const std = @import("std");

/// The Inkling 2 CUDA family over the runtime, the core and the lane core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/inkling2.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    return m;
}

/// The family in tf-cuda-test's `runner`; tf-inkling2-engine (the engine's graph and greedy checks).
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, runner: *std.Build.Module) void {
    runner.addImport("inkling2", module(b, target, optimize, cuda, core, lanes));
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/engine_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    m.addImport("cuda", cuda);
    m.addImport("inkling2", module(b, target, optimize, cuda, core, lanes));
    const exe = b.addExecutable(.{ .name = "tf-inkling2-engine", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-engine", "Inkling 2's engine checks: graph replay against eager rows, greedy decode").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The family's host regressions (config, chat wiring, reply split, admission, the attention plan) in `all`.
pub fn hostTests(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, lanes) }));
    b.step("test-inkling2-cuda-host", "Inkling 2's CUDA family host regressions (no GPU work)").dependOn(&run.step);
    all.dependOn(&run.step);
}
