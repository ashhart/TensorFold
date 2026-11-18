//! Local bf16 Ling forward diagnostics over existing registered Metal kernels.
const std = @import("std");
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal_mod: *std.Build.Module, sources: *std.Build.Module, kimi_metallib: *std.Build.Module) void {
    const ling_ck = b.createModule(.{
        .root_source_file = b.path("zig/src/families/ling/checkpoint.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "safetensors", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target }) },
        },
    });
    const ling_kernels = b.createModule(.{ .root_source_file = b.path("zig/src/families/kimi_k3/kernels.zig"), .target = target, .imports = &.{.{ .name = "metal", .module = metal_mod }} });
    const ling_rows = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/metal_rows.zig"), .target = target, .imports = &.{
        .{ .name = "metal", .module = metal_mod },
        .{ .name = "kernel_sources", .module = sources },
        .{ .name = "ling_checkpoint", .module = ling_ck },
        .{ .name = "kimi_kernels", .module = ling_kernels },
        .{ .name = "kimi_metallib", .module = kimi_metallib },
    } });
    const ling_kda = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/metal_kda.zig"), .target = target, .imports = &.{
        .{ .name = "metal", .module = metal_mod },
        .{ .name = "ling_checkpoint", .module = ling_ck },
        .{ .name = "kimi_kernels", .module = ling_kernels },
        .{ .name = "ling_rows", .module = ling_rows },
    } });
    const ling_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/ling/sources.zig") });
    const ling_mla = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/metal_mla.zig"), .target = target, .imports = &.{
        .{ .name = "metal", .module = metal_mod },           .{ .name = "ling_checkpoint", .module = ling_ck },
        .{ .name = "kimi_kernels", .module = ling_kernels }, .{ .name = "ling_rows", .module = ling_rows },
        .{ .name = "ling_sources", .module = ling_sources },
    } });
    const ling_moe = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/metal_moe.zig"), .target = target, .imports = &.{
        .{ .name = "metal", .module = metal_mod },           .{ .name = "ling_checkpoint", .module = ling_ck },
        .{ .name = "kimi_kernels", .module = ling_kernels }, .{ .name = "ling_rows", .module = ling_rows },
        .{ .name = "ling_sources", .module = ling_sources },
    } });
    const ling_forward = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/metal_forward.zig"), .target = target, .imports = &.{
        .{ .name = "metal", .module = metal_mod },     .{ .name = "ling_checkpoint", .module = ling_ck },
        .{ .name = "ling_rows", .module = ling_rows }, .{ .name = "ling_kda", .module = ling_kda },
        .{ .name = "ling_mla", .module = ling_mla },   .{ .name = "ling_moe", .module = ling_moe },
    } });
    const ling_forward_tool = b.addExecutable(.{ .name = "ling-forward", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/ling_forward.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal_mod }, .{ .name = "ling_checkpoint", .module = ling_ck }, .{ .name = "ling_rows", .module = ling_rows }, .{ .name = "ling_forward", .module = ling_forward }, .{ .name = "ling_mla", .module = ling_mla } },
    }) });
    b.step("tf-ling-forward", "Build the native Ling bf16 full-forward diagnostic").dependOn(&b.addInstallArtifact(ling_forward_tool, .{}).step);
    const python = b.graph.environ_map.get("TF_LING_PYTHON") orelse "python3";
    const full_check = b.addSystemCommand(&.{ python, "tools/zig/ling_forward_check.py", "--binary" });
    full_check.addArtifactArg(ling_forward_tool);
    b.step("test-ling-forward", "Optional real-checkpoint Ling exactness and FP64 comparison receipts").dependOn(&full_check.step);
    const ling_projection = b.addExecutable(.{ .name = "ling-projection", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/ling_projection.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "ling_checkpoint", .module = ling_ck },
            .{ .name = "kimi_metallib", .module = kimi_metallib },
            .{ .name = "kimi_kernels", .module = ling_kernels },
            .{ .name = "ling_rows", .module = ling_rows },
            .{ .name = "ling_kda", .module = ling_kda },
            .{ .name = "ling_mla", .module = ling_mla },
            .{ .name = "ling_moe", .module = ling_moe },
        },
    }) });
    b.step("tf-ling-projection", "Build Ling's real bf16 projection diagnostic").dependOn(&b.addInstallArtifact(ling_projection, .{}).step);
    const ling_checkpoint = b.addExecutable(.{ .name = "ling-checkpoint", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/ling_checkpoint.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal_mod }, .{ .name = "ling_checkpoint", .module = ling_ck } },
    }) });
    b.step("tf-ling-checkpoint", "Build the native bf16 Ling checkpoint validator").dependOn(&b.addInstallArtifact(ling_checkpoint, .{}).step);
}
