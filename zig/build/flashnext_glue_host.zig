//! The flashnext prompt contract's host checks: the served row range and the state alias, without a device.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, family: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_flashnext_glue_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    all.dependOn(&run.step);
    b.step("test-flashnext-glue-contract-host", "The flashnext prompt contract's served row range and its state alias").dependOn(&run.step);
    const fused = b.step("test-flashnext-hc-fused-host", "HC fusion dispatch guards and FP32 stress witnesses without a GPU");
    all.dependOn(fused);
    const checks = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_hc_fused.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } } });
    fused.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = checks })).step);
    fused.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{"HC prompt fusions"} })).step);
    const mix = b.step("test-flashnext-hc-mix-host", "HC vector mix stress witnesses and launch records without a GPU");
    all.dependOn(mix);
    const mix_checks = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_hc_mix.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    mix.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mix_checks, .filters = &.{"vector mix"} })).step);
    const mix_records = b.createModule(.{ .root_source_file = b.path("zig/src/core/contract/flashnext_hc_mix_vec_records.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{.{ .name = "gpu_memory", .module = core.import_table.get("gpu_memory").? }} });
    mix.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mix_records, .filters = &.{"HC vector mix"} })).step);
}
