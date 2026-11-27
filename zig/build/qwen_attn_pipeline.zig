//! Focused attention-pipeline checks; neither host step loads CUDA images or a device.
const std = @import("std");
pub fn host(b: *std.Build, cuda: *std.Build.Module, qwen: *std.Build.Module, core: *std.Build.Module) void {
    const tests = b.createModule(.{ .root_source_file = b.path("zig/qwen_attn_pipeline_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen }, .{ .name = "core", .module = core } } });
    b.step("test-qwen-attn-pipeline-host", "Actual prompt dispatch, resources, unchanged launch and option defaults").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests })).step);
    const schedule = b.addExecutable(.{ .name = "attn-pipe-schedule", .root_module = b.createModule(.{ .target = b.graph.host, .optimize = .debug, .link_libcpp = true }) });
    schedule.root_module.addCSourceFile(.{ .file = b.path("zig/tests/cuda/attn_pipe_schedule_host.cpp"), .flags = &.{ "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror" } });
    schedule.root_module.addIncludePath(b.path("zig/kernels/cuda"));
    b.step("test-qwen-attn-pipeline-schedule-host", "Actual scheduling helper, slot ownership, waits and planted fence controls").dependOn(&b.addRunArtifact(schedule).step);
    const records = b.createModule(.{ .root_source_file = b.path("zig/qwen_attn_pipeline_records_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    b.step("test-qwen-attn-pipeline-records-host", "Pending resource records refuse hardware qualification").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = records })).step);
    const contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_qwen_pipeline.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen }, .{ .name = "core", .module = core } } });
    b.step("test-qwen-attn-pipeline-fixture-host", "Actual generic nonfinite gate reaches both products' launch boundary").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract, .filters = &.{"Qwen attention pipeline fixture:"} })).step);
    const object = b.addObject(.{ .name = "qwen-attn-pipeline-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen_attn_pipeline_compile.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "contract", .module = contract }, .{ .name = "qwen35", .module = qwen } } }) });
    _ = object.getEmittedBin();
    b.step("compile-qwen-attn-pipeline-host", "Actual attention replacement contract and engine startup without devices").dependOn(&object.step);
}
