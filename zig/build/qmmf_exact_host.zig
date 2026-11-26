//! The exact NVFP4 wide tile's actual launcher and contract registration without CUDA hardware.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qmmf_exact_tile_host.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    root.addIncludePath(b.path("zig/kernels/cuda"));
    root.addCSourceFile(.{
        .file = b.path("zig/tests/cuda/qmmf_exact_tile_layout.cpp"),
        .flags = &.{ "-std=c++17", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off" },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"qmmf exact tile:"} }));
    b.step("test-qmmf-exact-tile-host", "Fixed-split NVFP4 16x128 launch, layout and contract checks without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const registration = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qmmf_exact_registration_host.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    const registered = b.addRunArtifact(b.addTest(.{ .root_module = registration, .filters = &.{"qmmf exact registration:"} }));
    b.step("test-qmmf-exact-contract-host", "Actual exact-tile backend constructor, list and pending resource records").dependOn(&registered.step);
    const panels = b.addRunArtifact(b.addTest(.{ .root_module = registration, .filters = &.{"qmmf exact panels:"} }));
    b.step("test-qmmf-exact-panels-host", "Pinned production panels and directed exceptional-value fixtures without a GPU").dependOn(&panels.step);
    all.dependOn(&registered.step);
}
