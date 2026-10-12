//! Flash Next's device checks on synthetic weights: programs that run on the GPU, each under its own step.

const std = @import("std");

/// tf-flashnext-widths: every product class at every oQ width against float64, every row's bits in every window.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-flashnext-widths", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_widths.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    const about = "Flash Next's products at every oQ width against float64 (worst column), every row's bits in every window";
    b.step("tf-flashnext-widths", about).dependOn(&b.addInstallArtifact(exe, .{}).step);
}
