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

pub fn rest(b: *std.Build, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const metal = engine.import_table.get("metal").?;
    const bodies = b.createModule(.{ .root_source_file = b.path("zig/src/metal_prefill_rest_compile.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "tensorfold", .module = engine }, .{ .name = "lanes", .module = lanes }, .{ .name = "kimi", .module = @import("metal_targets.zig").kimiModule(b, b.graph.host, .debug, metal, engine, lanes) } } });
    const object = b.addObject(.{ .name = "metal-prefill-rest-compile", .root_module = bodies });
    _ = object.getEmittedBin();
    const check = b.step("check-metal-prefill-rest-host", "Actual remaining Metal prompt callback bodies without shaders or execution");
    check.dependOn(&object.step);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen4_metal.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) } } });
    @import("qwen4.zig").metalNative(b, b.graph.host, host);
    const qwen = b.addObject(.{ .name = "qwen4-prefill-widths-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/qwen4_prefill_widths_compile.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "host", .module = host }} }) });
    _ = qwen.getEmittedBin();
    check.dependOn(&qwen.step);
}
