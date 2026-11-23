//! Execute the real family timing policies without loading Metal devices or shaders.
const std = @import("std");
pub fn steps(b: *std.Build, lanes: *std.Build.Module) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    inline for (.{ "nemotron", "glm" }) |family| {
        const m = b.createModule(.{ .root_source_file = b.path("zig/" ++ family ++ "_metal_timing_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "metal", .module = metal } } });
        b.step("test-" ++ family ++ "-metal-timing-host", "Actual " ++ family ++ " timing-controller host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
}

pub fn compile(b: *std.Build, draft_ids: *std.Build.Module) void {
    const target = if (b.graph.host.result.os.tag == .macos) b.graph.host else b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos });
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .link_libc = true });
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .link_libc = true });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "verbs", .module = @import("tp.zig").verbsModule(b, target, .debug) }} });
    const engine = b.createModule(.{ .root_source_file = b.path("zig/src/tensorfold.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "lanes", .module = lanes }, .{ .name = "living", .module = living }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "fabric", .module = fabric }, .{ .name = "nemotron_draft_ids", .module = draft_ids } } });
    _ = @import("qwen27.zig").engineModules(b, target, metal, sources, engine);
    engine.addImport("engram_hash_core", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target }));
    const empty = b.addWriteFiles().add("cost-compile-only.metallib", "");
    sources.addImport("nemotron_packed_metallib", @import("metal_targets.zig").embedded(b, empty, "nemotron_packed.metallib"));
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const root = b.createModule(.{ .root_source_file = b.path("zig/metal_cost_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const object = b.addObject(.{ .name = "metal-cost-adapters", .root_module = root });
    _ = object.getEmittedBin();
    b.step("check-metal-cost-adapters-host", "Compile actual Metal cost adapter bodies as a Darwin object without linking or GPU execution").dependOn(&object.step);
}
