//! Real native prompt adapter, pool and Gemma state copies under an inert CUDA driver.
const std = @import("std");
pub fn host(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, gemma: *std.Build.Module, all: *std.Build.Step) void {
    const adapter = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_prompt_pages.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/gemma4_shared_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "gemma4", .module = gemma }, .{ .name = "prompt_pages", .module = adapter } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Gemma shared:"} }));
    b.step("test-gemma4-shared-host", "Gemma paged resumes, shared streams and MTP cache bindings without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
