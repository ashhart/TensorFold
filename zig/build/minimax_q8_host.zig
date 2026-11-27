//! Registration and fixture checks for MiniMax's Q8 replacement contract without a GPU.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = b.graph.host });
    sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = b.addWriteFiles().add("minimax-q8-empty.zig", "pub const bytes = \"\";\n") });
    const metal = b.createModule(.{ .root_source_file = b.path("zig/residency_recording_metal.zig"), .target = b.graph.host });
    const family = b.createModule(.{ .root_source_file = b.path("zig/minimax_q8_contract_family.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/minimax_q8_contract_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "tensorfold", .module = family }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"MiniMax Q8 contract:"} }));
    b.step("test-minimax-q8-contract-host", "Real Q8 replacement, wrong-expert control and independent baseline buffers").dependOn(&run.step);
}
