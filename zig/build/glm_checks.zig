//! GLM-5.3-Flash's device checks on synthetic weights: programs that run on the GPU, each under its own step.

const std = @import("std");

/// tf-glm-rows-check: the products at every MLX width against float64, every row's bits in every window and launch.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module, sources: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-glm-rows-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_rows_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "kernel_sources", .module = sources } },
    }) });
    b.installArtifact(exe);
    const about = "GLM-5.3-Flash's products at every MLX width on synthetic weights against float64, each row's bits in every launch";
    b.step("tf-glm-rows-check", about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    const speed = b.addExecutable(.{ .name = "tf-glm-rows-speed", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_rows_speed.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "kernel_sources", .module = sources } },
    }) });
    b.step("tf-glm-rows-speed", "GLM-5.3-Flash one-row decode products at the checkpoints' shapes: GPU time a launch, GB/s, a token by class").dependOn(&b.addInstallArtifact(speed, .{}).step);
}
