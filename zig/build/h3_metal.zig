//! MiniMax H3's Metal tools: FastH3's transformer (tf-h3-dit) and MiniMax-H3-Image's (tf-h3-image).
const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module) void {
    // MiniMax H3 / FastH3's transformer: the packed rows' denoising forwards against a case file (M5 tensor units).
    const h3_kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") });
    const h3_core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const h3_mod = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/h3.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = h3_core }, .{ .name = "h3_kernels", .module = h3_kernels }, .{ .name = "kernel_sources", .module = sources } } });
    const h3_dit = b.addExecutable(.{ .name = "tf-h3-dit", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/h3_dit.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3", .module = h3_mod } },
    }) });
    b.step("tf-h3-dit", "FastH3's Metal transformer against a case file: velocities, every step, timings").dependOn(&b.addInstallArtifact(h3_dit, .{}).step);
    // MiniMax-H3-Image: the same transformer on a picture case, its six distilled steps gated against a reference
    const h3_image = b.addExecutable(.{ .name = "tf-h3-image", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/h3_image.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3", .module = h3_mod }, .{ .name = "core", .module = h3_core } },
    }) });
    b.step("tf-h3-image", "MiniMax-H3-Image's Metal transformer on a picture case: six distilled steps, teacher-forced and gated").dependOn(&b.addInstallArtifact(h3_image, .{}).step);
    @import("h3_image_encoder.zig").tool(b, target, optimize, metal);
}
