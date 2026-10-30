//! Actual tower tensor loading through small synthetic safetensors, without a driver or GPU library.
const std = @import("std");

pub fn host(b: *std.Build, core: *std.Build.Module, all: *std.Build.Step) void {
    const diagnostics = b.createModule(.{
        .root_source_file = b.path("zig/src/vision/qwen_cuda_weights_logs.zig"),
        .target = b.graph.host,
    });
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/src/vision/qwen_cuda_weights_test.zig"),
            .target = b.graph.host,
            .link_libc = true,
            .imports = &.{
                .{ .name = "core", .module = core },
                .{ .name = "qwen_weights_diagnostics", .module = diagnostics },
            },
        }),
        .filters = &.{"Qwen tower weights:"},
        .test_runner = .{ .path = b.path("zig/qwen_vision_weights_runner.zig"), .mode = .simple },
    });
    const run = b.addRunArtifact(tests);
    b.step("test-qwen-vision-weights-host", "Both checkpoint names and byte layouts without a device").dependOn(&run.step);
    all.dependOn(&run.step);
}
