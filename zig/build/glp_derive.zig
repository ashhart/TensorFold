//! The control vector producer: the derive tool, its host test and the modules they share.
const std = @import("std");
const metal_build = @import("metal_targets.zig");

/// The tool, its host test and the modules they share; macOS only, the family runs on Metal.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, test_step: *std.Build.Step) void {
    if (b.graph.host.result.os.tag != .macos) return;
    const mods = metal_build.metalEngineModules(b, target, .ReleaseSafe, draft_ids);
    const empty = b.addWriteFiles().add("glp-derive-compile-only.metallib", "");
    mods.sources.addImport("nemotron_packed_metallib", metal_build.embedded(b, empty, "nemotron_packed.metallib"));
    const metal = mods.metal;
    const sources = mods.sources;
    const shared = mods.qwen27;

    // The family's own modules, as the 27B's build makes them: the tool drives that same family.
    const core_row = b.createModule(.{ .root_source_file = b.path("zig/src/core/row_projection.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const core_lane = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_projection.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const qwen_core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_core.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true }) } } });
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "core", qwen_core }, .{ "core_lane", core_lane }, .{ "core_row", core_row }, .{ "bf16_topk", shared.bf16_topk }, .{ "bf16_topk_gpu", shared.bf16_topk_gpu }, .{ "tree_round", shared.tree_round }, .{ "tree_round_gpu", shared.tree_round_gpu }, .{ "tree_commit_gpu", shared.tree_commit }, .{ "draft_ops", shared.draft_ops }, .{ "kernel_sources", sources } }) |m| qwen_core.addImport(m[0], m[1]);
    const family = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/qwen27.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "core", .module = qwen_core }, .{ .name = "metal", .module = metal } } });
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "core_lane", core_lane }, .{ "qwen_runtime_sources", shared.runtime_sources }, .{ "core_delta", shared.delta }, .{ "qwen_gdn_source", shared.gdn_source } }) |m| family.addImport(m[0], m[1]);

    // The producer's own modules: the capture and the statistics, the prompt sets, the GLP file.
    const kernel_files = b.addWriteFiles();
    _ = kernel_files.addCopyFile(b.path("zig/kernels/metal/qwen27/capture.metal"), "capture.metal");
    const steer_sources = b.createModule(.{ .root_source_file = kernel_files.add("steer_sources.zig", "pub const capture = @embedFile(\"capture.metal\");\n") });
    const derive = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/steer_derive.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "steer_sources", .module = steer_sources } } });
    const prompt_set = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/steer_prompts.zig"), .target = target, .optimize = .ReleaseSafe });
    const glp = b.createModule(.{ .root_source_file = b.path("zig/src/core/glp.zig"), .target = target, .optimize = .ReleaseSafe });

    // The host test: the statistics alone, with no device and no model.
    const cases = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/steer_derive_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "steer_derive", .module = derive }, .{ .name = "metal", .module = metal }, .{ .name = "glp", .module = glp } } }) });
    const run = b.addRunArtifact(cases);
    b.step("test-steer-derive", "The control vector producer's statistics, with no GPU and no model").dependOn(&run.step);
    test_step.dependOn(&run.step);

    // The tool: the prompt sets through the native forward, then the GLP file.
    const tool = b.addExecutable(.{ .name = "tf-qwen27-steer-derive", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen27_steer_derive.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "qwen27", .module = family }, .{ .name = "core", .module = qwen_core }, .{ .name = "metal", .module = metal }, .{ .name = "steer_derive", .module = derive }, .{ .name = "steer_prompts", .module = prompt_set }, .{ .name = "glp", .module = glp } } }) });
    b.step("tf-qwen27-steer-derive", "Derive the control vectors from the model itself and write the GLP file").dependOn(&b.addInstallArtifact(tool, .{}).step);
}
