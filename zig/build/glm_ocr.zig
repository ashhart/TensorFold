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

pub fn hipTests(b: *std.Build, hip: *std.Build.Module, all: *std.Build.Step) void {
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_ocr/hip.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
    const weight_tests = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{"GLM-OCR HIP weights:"} }));
    all.dependOn(&weight_tests.step);
    b.step("test-glm-ocr-hip-host", "GLM-OCR HIP ownership and preallocation refusal without GPU execution").dependOn(&weight_tests.step);
    const upload = b.addTest(.{ .name = "tf-glm-ocr-weights-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/glm_ocr_weights_test.zig"),
        .target = b.graph.host,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "glm_ocr", .module = family } },
    }), .filters = &.{"GLM-OCR HIP device weights:"} });
    b.step("build-glm-ocr-weights-test", "Build real GLM-OCR upload checks; execution requires the device lease").dependOn(&b.addInstallArtifact(upload, .{}).step);
}
