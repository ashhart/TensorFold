//! Pure startup budgeting fixtures have no Metal compiler or device dependencies.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const constrained = b.step("test-flashnext-constraint-host", "Single-slot grammar routing, masks, drafts and cleanup on a fake target");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true });
        const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
        // the constraint run reads lanes through `tensorfold`, as native_engines' other files do; here that is lanes alone
        const tf = b.createModule(.{ .root_source_file = b.addWriteFiles().add("flashnext-constraint-tf.zig", "pub const lanes = @import(\"lanes\");\n"), .target = b.graph.host, .optimize = mode, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
        const tests = b.createModule(.{ .root_source_file = b.path("zig/src/native/flashnext_constraint_test.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = tf } } });
        constrained.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests })).step);
    }
    all.dependOn(constrained);
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
    legacy.addImport("lanes", lanes);
    b.step("test-flashnext-driver-host", "Single-driver completion and drain regressions without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = legacy, .filters = &.{ "status after a reply finishes", "a drained Flash Next host" } })).step);
    host.addImport("lanes", lanes);
    const legacy_object = b.addObject(.{ .name = "flashnext-legacy-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/flashnext_legacy_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "host", .module = legacy }} }) });
    _ = legacy_object.getEmittedBin();
    b.step("compile-flashnext-legacy-host", "Compile the existing Flash Next legacy host with no shaders or execution").dependOn(&legacy_object.step);
    const native_window = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    native_window.addImport("lanes", lanes);
    const window_object = b.addObject(.{ .name = "flashnext-window-open", .root_module = b.createModule(.{ .root_source_file = b.path("zig/flashnext_window_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "native", .module = native_window }} }) });
    _ = window_object.getEmittedBin();
    b.step("compile-flashnext-context-entry-host", "Compile the actual Flash Next context entry with empty embedded data").dependOn(&window_object.step);
    const object = b.addObject(.{ .name = "flashnext-budget-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/native_budget_compile.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "batch_host", .module = host }} }) });
    _ = object.getEmittedBin();
    b.step("compile-flashnext-budget-host", "Compile the actual shared host against an empty test-only packed image").dependOn(&object.step);
}
