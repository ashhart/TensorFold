//! Legacy proposal table, row and launch comparisons with no device initialization or shader producer.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, family: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/flashnext_proposal_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "flashnext", .module = family } } });
    b.step("test-flashnext-proposal-host", "Original Flash Next IDs, stored rows and native launches on common owners").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Flash Next proposal:"} })).step);
    const pages = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_fn_pages.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "flashnext", .module = family } },
    }) }));
    b.step("test-flashnext-pages-contract-host", "Paged media fixtures, address controls and production wrappers without a GPU").dependOn(&pages.step);
    b.top_level_steps.get("test-cuda-host").?.step.dependOn(&pages.step);
    const pack = b.createModule(.{ .root_source_file = b.path("zig/flashnext_proposal_pack_host.zig"), .target = b.graph.host, .link_libc = true });
    b.step("test-flashnext-proposal-pack-host", "Original Flash Next padded support, pack bytes and fingerprints").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pack, .filters = &.{"Flash Next proposal:"} })).step);
}
