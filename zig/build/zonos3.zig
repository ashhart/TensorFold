//! ZONOS3 Mini's CUDA family module and its zonos3-generate tool.
const std = @import("std");

/// The family over a runtime, the core and the lane core (its keyed sampler).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/zonos3/zonos3.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

/// The family's host tests (config, prompt, sampler, generation, output stage, release layout) and its host's.
pub fn hostTests(b: *std.Build, step: *std.Build.Step, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module) void {
    const family = module(b, b.graph.host, .debug, cuda, core, lanes);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = family }));
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/zonos_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "zonos3", .module = family }, .{ .name = "engine_api", .module = api }, .{ .name = "cuda", .module = cuda } } });
    const run_host = b.addRunArtifact(b.addTest(.{ .root_module = host }));
    const own = b.step("test-zonos-host", "ZONOS3 Mini's prompt, sampler, generation, output and host tests without a GPU");
    for ([_]*std.Build.Step{ &run.step, &run_host.step }) |r| {
        step.dependOn(r);
        own.dependOn(r);
    }
}

pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const tool = b.createModule(.{ .root_source_file = b.path("zig/src/families/zonos3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda }, .{ .name = "zonos3", .module = module(b, target, optimize, cuda, core, lanes) } } });
    const exe = b.addExecutable(.{ .name = "zonos3-generate", .root_module = tool });
    b.step("zonos3-generate", "ZONOS3 Mini's backbone on CUDA against the reference's dumps").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
