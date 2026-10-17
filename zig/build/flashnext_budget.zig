//! Pure startup budgeting fixtures have no Metal compiler or device dependencies.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const exe = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/flashnext_budget_tests.zig"), .target = b.graph.host, .optimize = .Debug }) });
    const run = b.addRunArtifact(exe);
    run.has_side_effects = true;
    b.step("test-flashnext-budget", "Prompt cache reservation before shared-session sizing and startup budgets").dependOn(&run.step);
    all.dependOn(&run.step);
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/flashnext_batch_host.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const legacy = b.createModule(.{ .root_source_file = b.path("zig/src/native/flashnext_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const legacy_object = b.addObject(.{ .name = "flashnext-legacy-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/flashnext_legacy_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "host", .module = legacy }} }) });
    _ = legacy_object.getEmittedBin();
    b.step("compile-flashnext-legacy-host", "Compile the existing Flash Next legacy host with no shaders or execution").dependOn(&legacy_object.step);
    const native_window = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const window_object = b.addObject(.{ .name = "flashnext-window-open", .root_module = b.createModule(.{ .root_source_file = b.path("zig/flashnext_window_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "native", .module = native_window }} }) });
    _ = window_object.getEmittedBin();
    b.step("compile-flashnext-window-host", "Compile the actual Flash Next context entry with empty embedded data").dependOn(&window_object.step);
    const object = b.addObject(.{ .name = "flashnext-budget-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/native_budget_compile.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "batch_host", .module = host }} }) });
    _ = object.getEmittedBin();
    b.step("compile-flashnext-budget-host", "Compile the actual shared host against an empty test-only packed image").dependOn(&object.step);
}
