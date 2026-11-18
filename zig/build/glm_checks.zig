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
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "kernel_sources", .module = sources },
        },
    }) });
    b.step(
        "tf-glm-rows-speed",
        "GLM-5.3-Flash one-row decode products at the checkpoints' shapes: GPU time a launch, GB/s, a token by class",
    ).dependOn(&b.addInstallArtifact(speed, .{}).step);
    const index = b.addExecutable(.{ .name = "tf-glm-index-speed", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_index_speed.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    b.step("tf-glm-index-speed", "GLM-5.3-Flash's decode indexer and sparse attention at long positions: GPU time a launch").dependOn(&b.addInstallArtifact(index, .{}).step);
    const kda = b.addExecutable(.{ .name = "tf-glm-kda-speed", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_kda_speed.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    const mrows = b.addExecutable(.{ .name = "tf-glm-moe-rows", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_moe_rows.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    b.step("tf-glm-moe-rows", "GLM-5.3-Flash's decode experts of 1-16 rows at two and four outputs a simdgroup, bit for bit, GPU time").dependOn(&b.addInstallArtifact(mrows, .{}).step);
    const arows = b.addExecutable(.{ .name = "tf-glm-attn-rows", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_attn_rows.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    b.step("tf-glm-attn-rows", "GLM-5.3-Flash's dense latent attention of a window's rows in three launches against each row's own, bit for bit, GPU time").dependOn(&b.addInstallArtifact(arows, .{}).step);
    b.step("tf-glm-kda-speed", "GLM-5.3-Flash's fused KDA decode step: the window kernel against kda_rows bit for bit, GPU time").dependOn(&b.addInstallArtifact(kda, .{}).step);
}

/// tf-glm-sample-check and test-glm-sampling: the real GLM sampler against its host reference, no checkpoint.
pub fn sampler(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    lanes: *std.Build.Module,
    sources: *std.Build.Module,
) void {
    // the gate compiles its shader source at runtime: no offline metallib
    const glm_draw = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm/draw_dispatch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "lanes", .module = lanes } },
    });
    const exe = b.addExecutable(.{ .name = "tf-glm-sample-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_sample_check.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "draw_dispatch", .module = glm_draw },
            .{ .name = "kernel_sources", .module = sources },
        },
    }) });
    b.step("tf-glm-sample-check", "Build the checkpoint-free GLM sampler gate").dependOn(&b.addInstallArtifact(exe, .{}).step);
    b.step("test-glm-sampling", "Run the real GLM sampler against its host reference").dependOn(&b.addRunArtifact(exe).step);
}
