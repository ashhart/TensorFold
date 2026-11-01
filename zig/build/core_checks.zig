//! The core's device checks on synthetic weights: programs that run on the GPU, each under its own step.

const std = @import("std");

/// tf-row-widths-check: the row projection at every MLX width and group against float64, windows, strides, stock bytes.
pub fn rowWidths(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, row_mod: *std.Build.Module) void {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
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

/// tf-affine-ref: the prompt products (affine_mm, simd_qmm) against float64, every row's bits in every window.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-affine-ref", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/affine_ref.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    b.installArtifact(exe);
    const about = "The prompt products (affine_mm, simd_qmm) on synthetic weights against float64, every row's bits in every window";
    b.step("tf-affine-ref", about).dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// tf-tile-check: the prompt tiles and their decode twin on synthetic matrices (any Mac's GPU), and the host test.
pub fn tileQmm(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    sources: *std.Build.Module,
    test_step: *std.Build.Step,
) void {
    const tile_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/core/tile_qmm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    const am_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/core/affine_mm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    const tile_check = b.addExecutable(.{ .name = "tf-tile-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/tile_qmm_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "tile_qmm", .module = tile_mod },
            .{ .name = "affine_mm", .module = am_mod },
        },
    }) });
    const about = "The prompt tiles on synthetic 4-, 6- and 8-bit matrices: rows, twin and grouped bit-equal, " ++
        "a float64 reference";
    b.step("tf-tile-check", about).dependOn(&b.addInstallArtifact(tile_check, .{}).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tile_mod })).step);
}

/// The core KV pool's pages on Metal: an arena and the streams' tables, host checks and a device round trip.
pub fn kvPages(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    const pages = b.createModule(.{
        .root_source_file = b.path("zig/src/core/metal_kv_pages.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = pages }));
    // Qwen3.5-2B's paged kernels against its contiguous ones, the same bits through shuffled pages
    const qwen = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/paged_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    const qwen_run = b.addRunArtifact(b.addTest(.{ .root_module = qwen }));
    // Nemotron's seven paged K/V kernels against main's, built as its loaders build them on this GPU
    const frags = b.createModule(.{
        .root_source_file = b.path("zig/src/core/frags.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    const nemotron = b.createModule(.{
        .root_source_file = b.path("zig/src/families/nemotron/paged_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "frags", .module = frags } },
    });
    const nemotron_run = b.addRunArtifact(b.addTest(.{ .root_module = nemotron }));
    const step = b.step("test-metal-kv-pages", "The KV pool's pages on Metal: leases into tables, a full pool, an idle pool, scattered pages on the GPU, Qwen3.5's and Nemotron's paged kernels");
    step.dependOn(&run.step);
    step.dependOn(&qwen_run.step);
    step.dependOn(&nemotron_run.step);
    test_step.dependOn(step);
}
