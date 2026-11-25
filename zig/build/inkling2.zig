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

/// The family in tf-cuda-test's `runner`, where the kernel contract reaches its launchers.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, runner: *std.Build.Module) void {
    _ = strip;
    runner.addImport("inkling2", module(b, target, optimize, cuda, core, lanes));
}

/// The family's host regressions (config, chat wiring, reply split, admission, the attention plan) in `all`.
pub fn hostTests(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const fam = module(b, host, .debug, cuda, core, lanes);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = fam }));
    const step = b.step("test-inkling2-cuda-host", "Inkling 2's CUDA family host regressions (no GPU work)");
    step.dependOn(&run.step);
    all.dependOn(&run.step);
    // the projection groups' contract registrations: shapes, controls and launch hooks, type-checked without nvcc
    const group5 = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_inkling2_group5.zig"), .target = host, .optimize = .debug, .link_libc = true });
    for ([_][]const u8{ "cuda", "core", "inkling2" }, [_]*std.Build.Module{ cuda, core, fam }) |n, m| group5.addImport(n, m);
    const contract = b.addRunArtifact(b.addTest(.{ .root_module = group5, .filters = &.{"contract_inkling2_group5:"} }));
    step.dependOn(&contract.step);
    all.dependOn(&contract.step);
}
