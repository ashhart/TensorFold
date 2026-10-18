//! Residency accounting and owner inventories without shader compilation or device execution.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host, .link_libc = true });
    if (b.graph.host.result.os.tag == .macos) metal.linkSystemLibrary("objc", .{});
    const run = b.addRunArtifact(b.addTest(.{ .root_module = metal, .filters = &.{ "residency", "a set is built", "buffers made later", "a join whose", "an operator load ceiling" } }));
    b.step("test-metal-residency-host", "Unique residency allocations, limits, flags and lifetimes without devices").dependOn(&run.step);
    const recording = b.createModule(.{ .root_source_file = b.path("zig/residency_recording_metal.zig"), .target = b.graph.host });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = b.graph.host, .link_libc = true });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = b.addWriteFiles().add("residency-empty.zig", "pub const bytes = \"\";\n") });
    const owners = b.createModule(.{ .root_source_file = b.path("zig/residency_owner_tests.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = recording }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "kernel_sources", .module = sources } } });
    _ = @import("qwen27.zig").engineModules(b, b.graph.host, recording, sources, owners);
    owners.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = b.graph.host }));
    b.step("test-metal-residency-owners-host", "Actual family allocations against the recorded residency inventory").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = owners, .filters = &.{"residency owner"} })).step);
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const bytes = b.addWriteFiles().add("residency-drama-empty.zig", "pub const bytes: []const u8 = &.{};\n");
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .link_libc = true });
    const rounds = b.createModule(.{ .root_source_file = b.path("zig/src/core/frame_rounds.zig"), .target = b.graph.host });
    const shim = b.addWriteFiles().add("residency-drama-core.zig", "pub const safetensors = @import(\"tensorfold\").safetensors;\npub const tokenizer = @import(\"tokenizer\");\npub const frame_rounds = @import(\"frame_rounds\");\n");
    const core = b.createModule(.{ .root_source_file = shim, .target = b.graph.host, .imports = &.{ .{ .name = "tensorfold", .module = engine }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "frame_rounds", .module = rounds } } });
    const drama = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/drama3_metal.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "drama3_metallib", .module = b.createModule(.{ .root_source_file = bytes }) } } });
    host.addImport("core", core);
    host.addImport("tokenizer", tokenizer);
    host.addImport("drama3", drama);
    const root = b.createModule(.{ .root_source_file = b.path("zig/residency_native_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "host", .module = host }} });
    const object = b.addObject(.{ .name = "metal-residency-native", .root_module = root });
    _ = object.getEmittedBin();
    b.step("compile-metal-residency-host", "Compile all native Metal open routes without shaders or execution").dependOn(&object.step);
}
