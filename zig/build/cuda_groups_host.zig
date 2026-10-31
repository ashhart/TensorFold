//! CUDA family grouped launch recording tests, without loading a driver or checkpoint.
const std = @import("std");
pub fn host(b: *std.Build, cuda: *std.Build.Module, mods: anytype, all: *std.Build.Step) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/cuda_family_groups_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "qwen35", .module = mods.qwen35 },
        .{ .name = "gemma4", .module = @import("gemma4_cuda.zig").module(b, b.graph.host, .debug, cuda, mods.core, mods.lanes) },
    } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"GB10 family grouping:"} }));
    b.step("test-cuda-family-groups-host", "GB10 family grouping through actual adapters without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
