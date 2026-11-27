//! Persistence identities and disk-open options under recording drivers, without shader producers.
const std = @import("std");
pub fn steps(b: *std.Build, native: *std.Build.Module, all: *std.Build.Step) void {
    const open = b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"CUDA disk open:"} }));
    b.step("test-cuda-disk-open-host", "CUDA's disk store takes --spill-gib's mode and --checkpoint-slots").dependOn(&open.step);
    all.dependOn(&open.step);
    const kept = b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"CUDA kept:"} }));
    b.step("test-cuda-kept-host", "The shared kept image over host memory, stale keys through the real store").dependOn(&kept.step);
    all.dependOn(&kept.step);
    const identity = b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"disk identity:"} }));
    b.step("test-disk-identity-host", "Actual CUDA identity, resolved profile values and stale disk misses").dependOn(&identity.step);
    all.dependOn(&identity.step);
    const witnessed = b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"execution witness"} }));
    witnessed.has_side_effects = true;
    b.step("test-execution-witness-cuda-host", "The witness's first CUDA path: hook, every weight byte, stamps, reloads").dependOn(&witnessed.step);
    all.dependOn(&witnessed.step);
}

/// Compile the actual four Metal identity bodies without a shader producer, link or execution.
pub fn metal(b: *std.Build, metal_module: *std.Build.Module, engine: *std.Build.Module, api: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/disk_identity_metal_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal_module }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const object = b.addObject(.{ .name = "disk-identity-metal-host", .root_module = root });
    _ = object.getEmittedBin();
    b.step("compile-disk-identity-metal-host", "Compile four production Metal persistence identities without shaders").dependOn(&object.step);
    const witness = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal_witness.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal_module }, .{ .name = "tensorfold", .module = engine }, .{ .name = "core", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const witnessed = b.addRunArtifact(b.addTest(.{ .root_module = witness, .filters = &.{"execution witness"} }));
    witnessed.has_side_effects = true;
    if (b.top_level_steps.get("test")) |all| all.step.dependOn(&witnessed.step);
    b.step("test-execution-witness-metal-host", "The witness's Metal path: every weight byte, kernel sources, this Mac's facts, reloads").dependOn(&witnessed.step);
    const qwen27 = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_witness.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal_module }, .{ .name = "tensorfold", .module = engine }, .{ .name = "core", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const qwen27_witnessed = b.addRunArtifact(b.addTest(.{ .root_module = qwen27, .filters = &.{"execution witness"} }));
    qwen27_witnessed.has_side_effects = true;
    if (b.top_level_steps.get("test")) |all| all.step.dependOn(&qwen27_witnessed.step);
    b.step("test-execution-witness-qwen27-host", "The witness's Qwen3.8-27B-class Metal path: geometries, the load's builds, the digest").dependOn(&qwen27_witnessed.step);
}
