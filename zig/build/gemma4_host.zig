//! Safe host targets. Recording fixtures execute no GPU call; compile objects use only embedded source strings.
const std = @import("std");
pub fn tests(b: *std.Build, sources: *std.Build.Module) void {
    const config_run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/config.zig"), .target = b.graph.host, .link_libc = true }) }));
    config_run.has_side_effects = true;
    b.step("test-gemma4-config", "Real Gemma parser fixtures without GPU dependencies").dependOn(&config_run.step);
    const fake = b.createModule(.{ .root_source_file = b.path("zig/t52_metal_fake.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/t52_affine_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "kernel_sources", .module = sources } } });
    const filter = b.option([]const u8, "gemma4-host-filter", "One portable shared affine regression");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{filter orelse "T52"} }));
    run.has_side_effects = true;
    b.step("test-gemma4-affine-host", "Record actual shared dispatch without devices or shader compilation").dependOn(&run.step);
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/gemma4.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/t52_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "host", .module = host }, .{ .name = "tensorfold", .module = engine } } });
    const object = b.addObject(.{ .name = "gemma4-host-compile", .root_module = root });
    _ = object.getEmittedBin();
    b.step("compile-gemma4-host", "Compile actual startup and restored GLM/LFM2 callers without a metallib").dependOn(&object.step);
}
