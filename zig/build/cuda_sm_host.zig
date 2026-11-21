//! The real release metadata and runtime, with inert images and a recording driver in place of a GPU.
const std = @import("std");
pub fn steps(b: *std.Build, api: *std.Build.Module, host_tests: *std.Build.Step) void {
    const catalog = @import("cuda.zig");
    const empty = b.addWriteFiles().add("inert-sm-host.fatbin", "");
    const images: [catalog.kernels.len]?std.Build.LazyPath = @splat(empty);
    const runtime = @import("cuda_runtime.zig").runtimeWithTargets(b, b.graph.host, .debug, &images, null, catalog.release_sms);
    const logs = b.createModule(.{ .root_source_file = b.path("zig/cuda_sm_logs.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/cuda_sm_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = runtime }, .{ .name = "engine_api", .module = api }, .{ .name = "qwen_weights_diagnostics", .module = logs } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"CUDA release SM:"}, .test_runner = .{ .path = b.path("zig/qwen_vision_weights_runner.zig"), .mode = .simple } }));
    b.step("test-cuda-sm-host", "Release SM admission and capability JSON with inert images and no GPU").dependOn(&run.step);
    host_tests.dependOn(&run.step);
}
