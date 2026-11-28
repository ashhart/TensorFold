//! HIP tail graph refresh, staging ownership and actual readback body checks without a GPU.
const std = @import("std");
pub fn steps(b: *std.Build, hip: *std.Build.Module, family: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/hip/tail_graph_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "hip", .module = hip }} });
    const host = b.step("test-flashnext-hip-tail-host", "Exact HIP tail sequence refresh, rollback and staging fences without a GPU");
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"HIP tail:"} })).step);
    const environment = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host, .link_libc = true });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = environment, .filters = &.{"Flash Next strict env:"} })).step);
    all.dependOn(host);
    const body = b.step("test-flashnext-hip-tail-body-host", "Production HIP tail readback and original graph/pool regressions without a GPU");
    body.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "HIP tail body:", "a round's graph is chosen for the pool's caches only" } })).step);
    body.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = hip, .filters = &.{ "a held replay is the shape's", "kernel nodes refuse" } })).step);
    all.dependOn(body);
    const runner = b.addExecutable(.{ .name = "flashnext-hip-tail-compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_hip/main.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "flashnext", .module = family }, .{ .name = "tokenizer", .module = hip.import_table.get("core").?.import_table.get("tokenizer").? } },
    }) });
    const callbacks = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/hip_tail_compile.zig"), .target = b.graph.host, .link_libc = true });
    var imports = family.import_table.iterator();
    while (imports.next()) |entry| callbacks.addImport(entry.key_ptr.*, entry.value_ptr.*);
    const object = b.addObject(.{ .name = "flashnext-hip-native-tail", .root_module = callbacks });
    const compile_step = b.step("compile-flashnext-hip-tail", "Compile actual HIP family runner and native tail callbacks with inert images, no device execution");
    compile_step.dependOn(&runner.step);
    compile_step.dependOn(&object.step);
}
