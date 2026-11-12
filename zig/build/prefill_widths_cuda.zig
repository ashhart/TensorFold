//! Completed prompt observations at the three served CUDA families' existing waits, without devices.
const std = @import("std");
pub fn tests(b: *std.Build, cuda: *std.Build.Module, mods: anytype, all: *std.Build.Step) void {
    const host = b.graph.host;
    const m = b.createModule(.{ .root_source_file = b.path("zig/cuda_prefill_widths_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{
        .{ .name = "qwen35", .module = mods.qwen35 },
        .{ .name = "nemotron", .module = mods.nemotron },
        .{ .name = "glm", .module = @import("cuda_engines.zig").glmCuda(b, host, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) },
        .{ .name = "cuda", .module = cuda },
        .{ .name = "lanes", .module = mods.lanes },
    } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"CUDA prefill widths:"} }));
    run.has_side_effects = true;
    b.step("test-cuda-prefill-widths-host", "Three CUDA prompt adapters at existing completion waits without devices").dependOn(&run.step);
    all.dependOn(&run.step);
}
