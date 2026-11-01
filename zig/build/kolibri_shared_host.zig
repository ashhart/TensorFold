//! Real native prompt adapter, pool and Kolibri state copies under an inert CUDA driver.
const std = @import("std");
pub fn host(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, kolibri: *std.Build.Module, all: *std.Build.Step) void {
    const adapter = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_prompt_pages.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/kolibri_shared_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "kolibri1", .module = kolibri }, .{ .name = "prompt_pages", .module = adapter } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Kolibri shared:"} }));
    b.step("test-kolibri-shared-host", "Kolibri paged resumes, shared streams and copy-draft cache bindings without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
