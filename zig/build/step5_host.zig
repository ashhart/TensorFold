//! Step-5 host checks share the actual family and native callback, using no GPU work or shader compiler.
const std = @import("std");
pub fn steps(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/step5.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const filter = b.option([]const u8, "step5-filter", "One Step-5 host regression");
    const check = b.step("test-step5-host", "Step-5 geometry and owned prompt-cache callback fixtures without GPU work");
    const family = b.createModule(.{ .root_source_file = b.path("zig/step5_family_tests.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = engine.import_table.get("kernel_sources").? } } });
    for ([_]*std.Build.Module{ family, host }) |module| {
        const tests = b.addTest(.{ .root_module = module, .filters = &.{ "T50 imports", filter orelse "T50" } });
        const run = b.addRunArtifact(tests);
        run.has_side_effects = true;
        check.dependOn(&run.step);
    }
    const object = b.addObject(.{ .name = "step5-host-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/step5_host_compile.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "host", .module = host }} }) });
    _ = object.getEmittedBin();
    b.step("compile-step5-host", "Compile actual Step-5 startup and family with an empty test-only packed image").dependOn(&object.step);
}
