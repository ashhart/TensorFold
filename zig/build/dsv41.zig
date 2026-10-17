//! DeepSeek-V4.1's build: the family's host side (zig/src/families/deepseek_v41) and its host tests on any OS.

const std = @import("std");

/// The backend-neutral core (the safetensors reader the pack headers use) for the family's host side.
fn core(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("tokenizer", tokenizer);
    return m;
}

/// The family's host side (config, pack headers, a rank's weight plan) as the `deepseek_v41` module.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/deepseek_v41.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("core", core(b, target, optimize));
    return m;
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug) }));
    b.step("test-dsv41", "DeepSeek-V4.1's host-side unit tests (no GPU work)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
