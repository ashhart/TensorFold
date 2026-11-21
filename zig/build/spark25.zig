//! Spark2.5's CUDA family over the maintained runtime, the core and the lane core; its host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/spark25/spark25.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-spark25-host", "Spark2.5 config admission, RoPE tables, cache sizes, snapshot runs and refusals without a GPU").dependOn(&run.step);
    const core = m.import_table.get("core").?;
    const lanes = m.import_table.get("lanes").?;
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const prefill_root = b.createModule(.{ .root_source_file = b.path("zig/spark25_prefill_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = m.import_table.get("cuda").? }, .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "spark25", .module = m } } });
    const prefill_run = b.addRunArtifact(b.addTest(.{ .root_module = prefill_root, .filters = &.{"Spark25 prefill:"} }));
    b.step("test-spark25-prefill-host", "Spark2.5 cancelled-prefill snapshots and exact resumed rows without a GPU").dependOn(&prefill_run.step);
    all.dependOn(&run.step);
    all.dependOn(&prefill_run.step);
}
