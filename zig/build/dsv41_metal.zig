//! DeepSeek-V4.1's Metal backend package (families/deepseek_v41/metal) and its device tests, on macOS.
const std = @import("std");

/// The package over the family's neutral host module; `metal` and `sources` are the engine's own modules.
pub fn package(b: *std.Build, metal: *std.Build.Module, sources: *std.Build.Module, optimize: std.builtin.OptimizeMode) struct { pkg: *std.Build.Module, host: *std.Build.Module } {
    const host = @import("dsv41_shared.zig").moduleAt(b, b.graph.host, optimize, "neutral.zig");
    // the lanes and the engine API the backend's modules import
    @import("dsv41_draft.zig").addImportsAt(b, host, b.graph.host, optimize);
    const pkg = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/metal/metal.zig"),
        .target = b.graph.host,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "dsv41", .module = host } },
    });
    return .{ .pkg = pkg, .host = host };
}

/// `test-dsv41-metal`: package, layer and forward checks on this GPU; `test-dsv41-metal-real`: a TF_DSV41_REAL layer.
pub fn steps(b: *std.Build, metal: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    const p = package(b, metal, sources, .Debug);
    const tests = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_metal/layer_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = p.pkg }, .{ .name = "metal", .module = metal }, .{ .name = "dsv41", .module = p.host } },
    });
    const step = b.step("test-dsv41-metal", "DeepSeek-V4.1 Metal layer and forward: float64 stages, rows at once == one at a time");
    const pkg_run = &b.addRunArtifact(b.addTest(.{ .root_module = p.pkg })).step;
    step.dependOn(pkg_run);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    run.has_side_effects = true; // a device run: never cached
    step.dependOn(&run.step);
    test_step.dependOn(pkg_run); // the package's host checks only; the device run stays in its own step
    const fast = package(b, metal, sources, .ReleaseSafe);
    const real = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_metal/real_test.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = fast.pkg }, .{ .name = "metal", .module = metal }, .{ .name = "dsv41", .module = fast.host } },
    });
    const real_run = b.addRunArtifact(b.addTest(.{ .root_module = real }));
    real_run.has_side_effects = true;
    b.step("test-dsv41-metal-real", "DeepSeek-V4.1 Metal: one release layer (TF_DSV41_REAL) against float64").dependOn(&real_run.step);
    // the same layer's decode timed (GPU span, host encode, per-dispatch classes) at release speed
    const quick = package(b, metal, sources, .ReleaseFast);
    const timing = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_metal/timing_real.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = quick.pkg }, .{ .name = "metal", .module = metal }, .{ .name = "dsv41", .module = quick.host }, .{ .name = "kernel_sources", .module = sources } },
    });
    const timing_run = b.addRunArtifact(b.addTest(.{ .root_module = timing }));
    timing_run.has_side_effects = true;
    b.step("time-dsv41-metal-real", "DeepSeek-V4.1 Metal: one release layer's decode timed at 1 and 4 rows").dependOn(&timing_run.step);
    const ptime = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_metal/prompt_time.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = fast.pkg }, .{ .name = "metal", .module = metal }, .{ .name = "dsv41", .module = fast.host } },
    });
    const ptime_run = b.addRunArtifact(b.addTest(.{ .root_module = ptime }));
    ptime_run.has_side_effects = true;
    b.step("test-dsv41-metal-prompt-time", "DeepSeek-V4.1 Metal: ms a stage of one release layer's prompt chunks (TF_DSV41_REAL)").dependOn(&ptime_run.step);
}
