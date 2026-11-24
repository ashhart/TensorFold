//! The leader step header (zig/src/fabric/step): a std-only module, its host tests, and compile checks for aarch64.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/fabric/step/step.zig"), .target = target, .optimize = optimize });
}

/// The tests' root: the same files, with libc for the golden fixtures' rewrite only (the module itself has none).
fn tests(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/fabric/step/step.zig"), .target = target, .optimize = .Debug, .link_libc = true });
}

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.graph.host;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests(b, host) }));
    run.has_side_effects = true; // TF_STEP_FIXTURES_OUT rewrites files, so a cached result must not skip it
    const step = b.step("test-fabric-step", "Step header: encode, decode, named refusals, golden fixtures");
    step.dependOn(&run.step);
    all.dependOn(step);
    const compile = b.step("test-fabric-step-compile", "Compile the step header's tests for Linux and macOS on aarch64");
    for ([_]std.Target.Os.Tag{ .linux, .macos }) |os| {
        const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = os });
        compile.dependOn(&b.addTest(.{ .root_module = tests(b, target) }).step);
    }
    all.dependOn(compile);
}
