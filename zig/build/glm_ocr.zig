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

pub fn hipTests(b: *std.Build, hip: *std.Build.Module, device_hip: *std.Build.Module, all: *std.Build.Step, options: @import("hip.zig").Options) void {
    const attention = b.addTest(.{ .name = "tf-aotriton-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/aotriton_test.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "core", .module = device_hip.import_table.get("core").? } },
    }), .filters = &.{"HIP AOTriton device:"} });
    b.step("build-aotriton-test", "Build native HIP attention checks against independent SDPA fixtures").dependOn(&b.addInstallArtifact(attention, .{}).step);
    const stub_files = b.addWriteFiles();
    const stub = b.createModule(.{ .root_source_file = stub_files.add("glm_ocr_stub.zig", "pub const arch = \"\"; pub const bytes align(8) = [_]u8{};\n") });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_ocr/hip.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
    family.addImport("glm_ocr_kernels", stub);
    const weight_tests = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "GLM-OCR HIP weights:", "GLM-OCR HIP ops:", "GLM-OCR HIP vision admission:", "GLM-OCR HIP decoder:", "GLM-OCR HIP input:", "GLM-OCR HIP snapshot:" } }));
    all.dependOn(&weight_tests.step);
    b.step("test-glm-ocr-hip-host", "GLM-OCR HIP ownership and preallocation refusal without GPU execution").dependOn(&weight_tests.step);
    const upload_family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_ocr/hip.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "core", .module = device_hip.import_table.get("core").? }, .{ .name = "glm_ocr_kernels", .module = stub } },
    });
    const upload = b.addTest(.{ .name = "tf-glm-ocr-weights-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/glm_ocr_weights_test.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "glm_ocr", .module = upload_family } },
    }), .filters = &.{"GLM-OCR HIP device weights:"} });
    b.step("build-glm-ocr-weights-test", "Build real GLM-OCR upload checks; execution requires device selection and admission").dependOn(&b.addInstallArtifact(upload, .{}).step);
    const cc = options.hipcc orelse "hipcc";
    const resolved = b.findProgram(.{ .names = &.{cc} }) orelse cc;
    const arch = options.hip_arch orelse "gfx1151";
    const files = b.addWriteFiles();
    const code = @import("hip.zig").compile(b, .{ .name = cc, .resolved = resolved, .device_lib = options.device_lib }, arch, b.path("zig/kernels/hip/glm_ocr/pointwise.hip"), "glm_ocr_pointwise.hsaco");
    _ = files.addCopyFile(code, "glm_ocr_pointwise.hsaco");
    const data = b.createModule(.{ .root_source_file = files.add("glm_ocr_kernels.zig", b.fmt("pub const arch = \"{s}\"; pub const bytes align(8) = @embedFile(\"glm_ocr_pointwise.hsaco\").*;\n", .{arch})) });
    const device_family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_ocr/hip.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "core", .module = device_hip.import_table.get("core").? }, .{ .name = "glm_ocr_kernels", .module = data } },
    });
    const patch = b.addTest(.{ .name = "tf-glm-ocr-patch-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/glm_ocr_patch_test.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "glm_ocr", .module = device_family }, .{ .name = "core", .module = device_hip.import_table.get("core").? } },
    }), .filters = &.{"GLM-OCR HIP patch:"} });
    b.step("build-glm-ocr-patch-test", "Build native patch projection comparison against the independent reference").dependOn(&b.addInstallArtifact(patch, .{}).step);
    const vision = b.addTest(.{ .name = "tf-glm-ocr-vision-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/glm_ocr_vision_test.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "glm_ocr", .module = device_family }, .{ .name = "core", .module = device_hip.import_table.get("core").? } },
    }), .filters = &.{"GLM-OCR HIP vision:"} });
    b.step("build-glm-ocr-vision-test", "Build native CogViT and connector reference comparison").dependOn(&b.addInstallArtifact(vision, .{}).step);
    const text = b.addTest(.{ .name = "tf-glm-ocr-text-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/glm_ocr_text_test.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "glm_ocr", .module = device_family }, .{ .name = "core", .module = device_hip.import_table.get("core").? } },
    }), .filters = &.{"GLM-OCR HIP text:"} });
    b.step("build-glm-ocr-text-test", "Build native four-norm decoder reference comparison").dependOn(&b.addInstallArtifact(text, .{}).step);
    const operations = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_ocr/hip_ops.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "glm_ocr_kernels", .module = data } },
    });
    const control_code = @import("hip.zig").compile(b, .{ .name = cc, .resolved = resolved, .device_lib = options.device_lib }, arch, b.path("zig/kernels/hip/contract_controls.hip"), "glm_ocr_controls.hsaco");
    const control_files = b.addWriteFiles();
    _ = control_files.addCopyFile(control_code, "controls.hsaco");
    const control_data = b.createModule(.{ .root_source_file = control_files.add("controls.zig", "pub const bytes align(8) = @embedFile(\"controls.hsaco\").*;\n") });
    const contracts = b.addTest(.{ .name = "tf-glm-ocr-contract-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/glm_ocr_contract_test.zig"),
        .target = device_hip.resolved_target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = device_hip }, .{ .name = "core", .module = device_hip.import_table.get("core").? }, .{ .name = "glm_ocr_ops", .module = operations }, .{ .name = "glm_ocr_kernels", .module = data }, .{ .name = "hip_controls", .module = control_data } },
    }), .filters = &.{"GLM-OCR HIP contract:"} });
    b.step("build-glm-ocr-contract-test", "Build common kernel gates, planted controls and launch audit for GLM-OCR").dependOn(&b.addInstallArtifact(contracts, .{}).step);
}
