//! Qwen2B prompt retention uses host buffers, and its served startup compiles without shaders or device execution.
const std = @import("std");

pub fn tests(b: *std.Build) void {
    const fake = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_metal.zig"), .target = b.graph.host });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const files = b.addWriteFiles();
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/kv_pages.h"), "kv_pages.h");
    const sources = b.createModule(.{ .root_source_file = files.add("sources.zig", "pub const kv_pages = @embedFile(\"kv_pages.h\");\n"), .target = b.graph.host });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/qwen35_cache_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "lanes", .module = lanes }, .{ .name = "kernel_sources", .module = sources } },
    });
    b.step("test-qwen35-cache-host", "Qwen2B retained bytes, paged images and resumed host-reference tokens").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Qwen retained:"} })).step);
    const pure = &.{ "admit the affine 2B", "admit untied heads", "text_config parses", "a top-level config", "the geometry refuses", "the Python loader's refusals", "candidate selection preserves", "a masked draw", "a stream's cache bytes" };
    b.step("test-qwen35-metal-host", "Qwen2B geometry, keyed draws and state sizing on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = pure })).step);
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen35.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const object = b.addObject(.{ .name = "qwen35-cache-compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen35_cache_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "host", .module = host }},
    }) });
    _ = object.getEmittedBin();
    b.step("check-qwen35-cache-host", "Compile Qwen2B served startup with a test-only empty packed image").dependOn(&object.step);
}
