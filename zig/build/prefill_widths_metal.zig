//! Metal prompt observations tested at their real completion boundaries; no shaders or model execution.
const std = @import("std");

pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/prompt_flight_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Nemotron prefill widths:"} }));
    run.has_side_effects = true;
    b.step("test-nemotron-prefill-widths-host", "Actual Nemotron cut and flight completion with recording buffers").dependOn(&run.step);
    parent.dependOn(&run.step);
}

pub fn compileHost(b: *std.Build, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/metal_prefill_widths_compile.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "tensorfold", .module = engine }, .{ .name = "lanes", .module = lanes } } });
    const object = b.addObject(.{ .name = "metal-prefill-widths-compile", .root_module = root });
    _ = object.getEmittedBin();
    b.step("check-nemotron-prefill-widths-host", "Compile actual Metal prefill, read, land and release without shaders").dependOn(&object.step);
}
