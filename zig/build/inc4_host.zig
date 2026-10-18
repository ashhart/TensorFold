//! Host controls for the incremented operator contracts; empty CUDA images and no device execution.
const std = @import("std");
pub fn round(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_round_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    b.step("test-inc4-round-host", "Actual round fixture layout and pending launch controls").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}
