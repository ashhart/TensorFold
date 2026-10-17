//! Portable H3 geometry and schedule checks, with no Metal compiler or GPU dependency.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/geometry_test.zig"), .target = target, .imports = &.{.{ .name = "safetensors", .module = st }} });
    const filter = b.option([]const u8, "h3-filter", "Run one H3 host regression");
    const cases = b.addTest(.{ .root_module = root, .filters = if (filter) |f| &.{f} else &.{} });
    const run = b.addRunArtifact(cases);
    b.step("test-h3-geometry", "H3 host admission and schedule regression fixtures").dependOn(&run.step);
    test_step.dependOn(&run.step);
    if (target.result.os.tag != .macos) return;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") });
    const model = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/h3.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = st }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "h3_kernels", .module = kernels } } });
    const tool = b.addExecutable(.{ .name = "tf-h3-host-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_dit.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "h3", .module = model }, .{ .name = "metal", .module = metal } } }) });
    b.step("compile-h3-host", "Compile the actual H3 tool without compiling or executing shaders").dependOn(&tool.step);
}
