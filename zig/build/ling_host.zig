//! Ling family host checks: config, weight map, fixtures, synth JSON and the CPU GGUF reader. No GPU.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/ling/host_test.zig"),
        .target = target,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-ling-host", "Ling family package: config, names, tiny fixtures, synth JSON, grouped routing, KDA mixer, MLA RoPE and mixer").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const weights = b.createModule(.{ .root_source_file = b.path("zig/src/core/cpu_weights.zig"), .target = target });
    const cpu = b.createModule(.{
        .root_source_file = b.path("zig/src/families/ling/cpu.zig"),
        .target = target,
        .imports = &.{.{ .name = "core", .module = weights }},
    });
    const cpu_run = b.addRunArtifact(b.addTest(.{ .root_module = cpu }));
    b.step("test-ling-cpu-host", "Ling CPU GGUF reader: names, views, metadata refusals, exact Q4_K/Q6_K/Q5_0 decode and projections").dependOn(&cpu_run.step);
    test_step.dependOn(&cpu_run.step);
}
