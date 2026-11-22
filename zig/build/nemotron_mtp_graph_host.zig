//! Two focused host groups for the optional MTP graph and its shared conditional driver boundary.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, mods: anytype, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/nemotron_mtp_graph_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "nemotron", .module = mods.nemotron }, .{ .name = "core", .module = mods.core } } });
    root.addIncludePath(b.path("zig/kernels/cuda"));
    root.addCSourceFile(.{ .file = b.path("zig/tests/nemotron_mtp_controller.cpp"), .flags = &.{ "-std=c++17", "-fno-fast-math", "-ffp-contract=off" } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    run.has_side_effects = true;
    b.step("test-nemotron-mtp-graph-host", "Compiled device controller versus actual DepthRule and strict opt-in/lifecycle boundaries").dependOn(&run.step);
    all.dependOn(&run.step);
    const boundary = b.createModule(.{ .root_source_file = b.path("zig/cuda_conditional_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const second = b.addRunArtifact(b.addTest(.{ .root_module = boundary }));
    second.has_side_effects = true;
    b.step("test-cuda-conditional-host", "Production conditional-node ABI and driver refusal without devices").dependOn(&second.step);
    all.dependOn(&second.step);
}
