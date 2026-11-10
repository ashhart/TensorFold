//! LFM3 checkpoint admission checks without Metal compilation or model execution.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const st = b.createModule(.{
        .root_source_file = b.path("zig/src/core/safetensors.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    const checks = b.step("test-lfm3-host", "LFM3 text and VL checkpoint config and tensor admission without a GPU");
    const rope = b.createModule(.{ .root_source_file = b.path("zig/src/core/rope_freq.zig"), .target = b.graph.host });
    for ([_][]const u8{ "config_test", "weights" }) |part| {
        const root = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/src/families/lfm3/{s}.zig", .{part})),
            .target = b.graph.host,
            .link_libc = true,
        });
        root.addImport("safetensors", st);
        root.addImport("rope_freq", rope);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
        checks.dependOn(&run.step);
    }
    const image = b.createModule(.{
        .root_source_file = b.path("zig/src/lfm3_image_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    const image_run = b.addRunArtifact(b.addTest(.{ .root_module = image }));
    const image_checks = b.step("test-lfm3-image-host", "LFM3 image geometry, processor, budget, pixel and vision tensor admission checks");
    image_checks.dependOn(&image_run.step);
    checks.dependOn(image_checks);
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host, .link_libc = true });
    const residency = b.createModule(.{
        .root_source_file = b.path("zig/src/lfm3_vision_residency_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    });
    const residency_run = b.addRunArtifact(b.addTest(.{ .root_module = residency }));
    b.step("test-lfm3-vision-residency-host", "LFM3 vision tower residency refusal without a GPU").dependOn(&residency_run.step);
    checks.dependOn(&residency_run.step);
    const memory = b.createModule(.{ .root_source_file = b.path("zig/src/gpu_memory.zig"), .target = b.graph.host });
    const contract = b.createModule(.{
        .root_source_file = b.path("zig/src/core/contract.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu_memory", .module = memory }, .{ .name = "safetensors", .module = st }, .{ .name = "rope_freq", .module = rope } },
    });
    const arithmetic = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/lfm3_math.zig"),
        .target = b.graph.host,
        .imports = &.{.{ .name = "contract", .module = contract }},
    });
    checks.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = arithmetic })).step);
    const prefill = b.createModule(.{ .root_source_file = b.path("zig/src/lfm3_prefill_tests.zig"), .target = b.graph.host, .link_libc = true });
    const prefill_run = b.addRunArtifact(b.addTest(.{ .root_module = prefill, .filters = &.{"LFM3 prefill"} }));
    b.step("test-lfm3-prefill-host", "LFM3 prompt cancellation, yield and progress without a GPU").dependOn(&prefill_run.step);
    checks.dependOn(&prefill_run.step);
    all.dependOn(checks);
    const tool = b.addExecutable(.{ .name = "tf-lfm3-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/lfm3_check.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{.{ .name = "safetensors", .module = st }},
    }) });
    b.installArtifact(tool);
    b.step("tf-lfm3-check", "Validate saved LFM3 tensor-header metadata without model data or a GPU")
        .dependOn(&b.addInstallArtifact(tool, .{}).step);
}
