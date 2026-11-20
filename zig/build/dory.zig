//! Dory family construction and its host tests (config, weight map, schedule, memory) without a GPU.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/dory/dory.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-dory-host", "Dory config, weight map, sublayer schedule and memory without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const loader = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"Dory loader:"} }));
    b.step("test-dory-loader-host", "Dory load refusals and device ownership under host allocation failure").dependOn(&loader.step);
    const upload = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"Dory queued upload:"} }));
    const core = m.import_table.get("core").?;
    const lanes = m.import_table.get("lanes").?;
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const prefill_root = b.createModule(.{ .root_source_file = b.path("zig/dory_prefill_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = m.import_table.get("cuda").? }, .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "dory", .module = m } } });
    const prefill_run = b.addRunArtifact(b.addTest(.{ .root_module = prefill_root, .filters = &.{"Dory prefill:"} }));
    b.step("test-dory-prefill-host", "Dory cancelled-prefill snapshots and exact resumed rows without a GPU").dependOn(&prefill_run.step);
    all.dependOn(&prefill_run.step);
    b.step("test-dory-upload-host", "Queued Dory inputs outlive caller mutation without GPU work").dependOn(&upload.step);
}
