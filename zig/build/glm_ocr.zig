//! GLM-OCR's device-free checkpoint and processor checks.
const std = @import("std");

pub fn hostTests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .link_libc = true }));
    const options = b.addOptions();
    options.addOption(?[]const u8, "model_dir", b.option([]const u8, "glm-ocr-model-dir", "Existing GLM-OCR checkpoint for host validation"));
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const tests = b.createModule(.{
            .root_source_file = b.path("zig/src/families/glm_ocr/host_test.zig"),
            .target = target,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{.{ .name = "core", .module = core }},
        });
        tests.addOptions("glm_ocr_options", options);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
        run.has_side_effects = true;
        all.dependOn(&run.step);
        const name = if (mode == .debug) "test-glm-ocr-host" else "test-glm-ocr-fast";
        b.step(name, "GLM-OCR metadata, tensor geometry and image preparation without a GPU").dependOn(&run.step);
    }
}
