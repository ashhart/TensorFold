//! The core's device checks on synthetic weights: programs that run on the GPU, each under its own step.

const std = @import("std");

/// tf-row-widths-check: the row projection at every MLX width and group against float64, windows, strides, stock bytes.
pub fn rowWidths(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, row_mod: *std.Build.Module) void {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    const exe = b.addExecutable(.{ .name = "tf-row-widths-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/row_widths_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "row_projection", .module = row_mod }, .{ .name = "core", .module = core } },
    }) });
    const about = "The row projection at every MLX width and group: fp64 reference, windows, strides, stock bytes";
    b.step("tf-row-widths-check", about).dependOn(&b.addInstallArtifact(exe, .{}).step);
}
