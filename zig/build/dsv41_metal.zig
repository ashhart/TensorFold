//! DeepSeek-V4.1's Metal backend package (families/deepseek_v41/metal) and its device tests, on macOS.
const std = @import("std");

/// The package over the family's neutral host module; `metal` and `sources` are the engine's own modules.
pub fn package(b: *std.Build, metal: *std.Build.Module, sources: *std.Build.Module, optimize: std.builtin.OptimizeMode) struct { pkg: *std.Build.Module, host: *std.Build.Module } {
    const host = @import("dsv41_shared.zig").moduleAt(b, b.graph.host, optimize, "neutral.zig");
    // the lanes, the engine API and the served engine (serve_lanes) the backend's target and pass plug into
    @import("dsv41_draft.zig").addImportsAt(b, host, b.graph.host, optimize);
    const pkg = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/metal/metal.zig"),
        .target = b.graph.host,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "dsv41", .module = host } },
    });
    pkg.addImport("core", host.import_table.get("core").?); // the picture decoder image input reads
    return .{ .pkg = pkg, .host = host };
}

/// The native server's engine over the package: metal/entry.zig, the family's own serving modules beside it.
pub fn engine(b: *std.Build, metal: *std.Build.Module, sources: *std.Build.Module, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const p = package(b, metal, sources, optimize);
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/metal/entry.zig"),
        .target = b.graph.host,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = p.pkg }, .{ .name = "dsv41", .module = p.host }, .{ .name = "metal", .module = metal } },
    });
    for ([_][]const u8{ "engine_api", "core", "tokenizer", "dsv41_serve", "lanes", "json" }) |name| m.addImport(name, p.host.import_table.get(name).?);
    return m;
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
    // the package's own checks run with the default test step; its device suites run from their named steps
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = p.pkg })).step);
    const step = b.step("test-dsv41-metal", "DeepSeek-V4.1 Metal layer and forward: float64 stages, rows at once == one at a time");
    // serve_lanes over the CPU twin, checked here so no CUDA step has to depend on the neutral served engine
    const lanes_test = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/serve_lanes_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = p.host.import_table.get("engine_api").? }, .{ .name = "lanes", .module = p.host.import_table.get("lanes").? }, .{ .name = "tokenizer", .module = p.host.import_table.get("tokenizer").? } },
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes_test })).step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    run.has_side_effects = true; // a device run: never cached
    step.dependOn(&run.step);
    const vis_run = b.addRunArtifact(b.addTest(.{ .root_module = tests, .filters = &.{"dsv41 metal vision"} }));
    vis_run.has_side_effects = true;
    const vision_step = b.step("test-dsv41-metal-vision", "DeepSeek-V4.1 Metal vision: the tower against float64, image rows in prompts");
    vision_step.dependOn(&vis_run.step);
    const prepare_run = b.addRunArtifact(b.addTest(.{ .root_module = p.pkg, .filters = &.{"prepare"} }));
    b.step("test-dsv41-metal-prepare", "DeepSeek-V4.1 Metal: image bytes to patches").dependOn(&prepare_run.step);
    // the served engine over the tiny model: LaneHost replies, drafts, the prompt cache, seeds, cancels
    const serve = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_metal/serve_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = p.pkg }, .{ .name = "metal", .module = metal }, .{ .name = "dsv41", .module = p.host } },
    });
    serve.addImport("dsv41_serve_fixtures", p.host.import_table.get("dsv41_serve").?.import_table.get("dsv41_serve_fixtures").?);
    const serve_run = b.addRunArtifact(b.addTest(.{ .root_module = serve }));
    serve_run.has_side_effects = true;
    step.dependOn(&serve_run.step);
    const serve_step = b.step("test-dsv41-metal-serve", "DeepSeek-V4.1 Metal served through LaneHost: concurrent, drafted, cached, seeded, cancelled");
    serve_step.dependOn(&serve_run.step);
    // the release-checkpoint steps (TF_DSV41_REAL) stay opt-in, so the umbrella holds the three default device suites
    const all = b.step("test-dsv41-metal-all", "DeepSeek-V4.1 Metal: every device test step on this GPU (layer, served, vision)");
    all.dependOn(step);
    all.dependOn(serve_step);
    all.dependOn(vision_step);
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
    // the release's own vision tower (TF_DSV41_REAL_VISION) against float64
    const vreal = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_metal/vision_real_test.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "dsv41_metal", .module = fast.pkg }, .{ .name = "metal", .module = metal }, .{ .name = "dsv41", .module = fast.host } },
    });
    const vreal_run = b.addRunArtifact(b.addTest(.{ .root_module = vreal }));
    vreal_run.has_side_effects = true;
    b.step("test-dsv41-metal-vision-real", "DeepSeek-V4.1 Metal: the release's vision tower (TF_DSV41_REAL_VISION) against float64").dependOn(&vreal_run.step);
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
    // tensorfold-dsv41 over Metal: the native server with this family's lane engine (the CUDA pair's macOS twin)
    const exe = @import("dsv41_draft.zig").serverOn(b, b.graph.host, .ReleaseSafe, engine(b, metal, sources, .ReleaseFast));
    b.step("dsv41-metal-serve", "Install tensorfold-dsv41 over Metal (DeepSeek-V4.1 release checkpoints on this Mac)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
