//! Source checks before any device build: literal device lookups have a source token; tf_mla keeps GLM's tokens.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/kernel_symbols.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    }) });
    const run = b.addRunArtifact(cases);
    run.setCwd(b.path("."));
    run.has_side_effects = true;
    const step = b.step("test-kernel-symbols-host", "Every literal CUDA/HIP lookup has a source token, without a device");
    step.dependOn(&run.step);
    all.dependOn(step);
    const twins = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/mla_twins.zig"), .target = b.graph.host, .optimize = .debug }) }));
    twins.setCwd(b.path("."));
    twins.has_side_effects = true;
    const twin_step = b.step("test-mla-twins-host", "tf_mla's absorb_q, expand_v and merge_part keep GLM's kernels' tokens at GLM's widths");
    twin_step.dependOn(&twins.step);
    all.dependOn(twin_step);
}
