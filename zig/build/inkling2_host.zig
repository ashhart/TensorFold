//! Inkling 2 family host checks (no GPU).
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, root: []const u8) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true });
    const dir = "zig/src/core/";
    const tokenizer = b.createModule(.{ .root_source_file = b.path(dir ++ "tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("tokenizer", tokenizer);
    const core = b.createModule(.{ .root_source_file = b.path(dir ++ "root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    core.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    m.addImport("core", core);
    m.addImport("template", b.createModule(.{ .root_source_file = b.path(dir ++ "template/template.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize });
    if (target.result.os.tag == .macos) {
        metal.linkFramework("Metal", .{});
        metal.linkFramework("Foundation", .{});
        metal.linkSystemLibrary("objc", .{});
    }
    m.addImport("metal", metal);
    m.addImport("kernel_sources", b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = target, .optimize = optimize }));
    m.addImport("inkling2_metal", @import("inkling2_sources.zig").module(b, target, optimize, m.import_table.get("kernel_sources").?));
    const inkling2_metal_ref = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/ref.zig"), .target = target, .optimize = optimize });
    inkling2_metal_ref.addImport("metal", m.import_table.get("metal").?);
    m.addImport("inkling2_metal_ref", inkling2_metal_ref);
    m.addImport("inkling2_glue_plan", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/glue_plan.zig"), .target = target, .optimize = optimize }));
    m.addImport("inkling2_loop_plan", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/loop_plan.zig"), .target = target, .optimize = optimize }));
    m.addImport("inkling2_tree_plan", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/tree_plan.zig"), .target = target, .optimize = optimize }));
    m.addImport("inkling2_small_ops_plan", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/small_ops_plan.zig"), .target = target, .optimize = optimize }));
    const lanes = b.createModule(.{ .root_source_file = b.path(dir ++ "lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("lanes", lanes);
    m.addImport("engine_api", b.createModule(.{ .root_source_file = b.path(dir ++ "engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
    return m;
}

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const sources = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/metal_threadgroup_source_host.zig"),
        .target = target,
        .optimize = .Debug,
    }) }));
    b.step("test-metal-threadgroup-source-host", "Complete Inkling and masked-prompt Metal sources have one reservation per declaration").dependOn(&sources.step);
    test_step.dependOn(&sources.step);
    const owners = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .debug, "zig/inkling2_source_owners_host.zig") }));
    b.step("test-inkling2-source-owners-host", "Family and legacy Metal exports share one checked source owner without an SDK").dependOn(&owners.step);
    test_step.dependOn(&owners.step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/families/inkling2/host_test.zig") }));
    b.step("test-inkling2-host", "Inkling 2 family package: config, weight map, layer plan, chat template and tokenizer").dependOn(&run.step);
    test_step.dependOn(&run.step);

    // The metal module reaches Darwin-only std.c, so these tests run on macOS only, like main's Metal steps.
    if (target.result.os.tag == .macos) {
        const metal_run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/inkling2_metal_host_test.zig") }));
        b.step("test-inkling2-metal-host", "Inkling 2 Mini Metal launch plans and arithmetic references").dependOn(&metal_run.step);
        test_step.dependOn(&metal_run.step);
        const media_run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/inkling2_media_checks.zig") }));
        media_run.has_side_effects = true;
        b.step("test-inkling2-media-metal", "Inkling 2 Mini image tower on this GPU: kernels, tower fixtures and the original checkpoint").dependOn(&media_run.step);
        const tfq4_run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/inkling2_tfq4_metal_checks.zig") }));
        tfq4_run.has_side_effects = true;
        b.step("test-inkling2-tfq4-metal", "Inkling 2 Mini Metal TFQ4 dense (kind 0/2/3, 2b/225b) & hybrid TFPW MoE GPU parity").dependOn(&tfq4_run.step);
    }
    const native = b.addObject(.{ .name = "inkling2-native", .root_module = module(b, target, .Debug, "zig/inkling2_native_compile.zig") });
    _ = native.getEmittedBin();
    b.step("compile-inkling2-native-host", "Analyze the actual Inkling 2 Mini native opener without shaders or devices").dependOn(&native.step);
    test_step.dependOn(&native.step);
}
