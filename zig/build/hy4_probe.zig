//! tf-hy4-speed: the Hy4-flash Metal graph's greedy decode speed and per-stage GPU profile on a real checkpoint.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, real_metal: *std.Build.Module, sources: *std.Build.Module) void {
    const metal = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/split_metal.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "real_metal", .module = real_metal }},
    });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    const with_metal: []const std.Build.Module.Import = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } };
    const row = b.createModule(.{ .root_source_file = b.path("zig/src/core/row_projection.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = with_metal });
    const lane = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_projection.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = with_metal });
    const route = b.createModule(.{ .root_source_file = b.path("zig/src/core/moe_route.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = with_metal });
    const draft_ops = b.createModule(.{ .root_source_file = b.path("zig/src/core/draft_ops.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "metal", .module = metal },
        .{ .name = "draft_ops_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/draft_ops_sources.zig") }) },
    } });
    const hash = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target, .optimize = optimize });
    const exe = b.addExecutable(.{ .name = "tf-hy4-speed", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/speed_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "core", .module = core },
            .{ .name = "row_projection", .module = row },
            .{ .name = "lane_projection", .module = lane },
            .{ .name = "moe_route", .module = route },
            .{ .name = "draft_ops", .module = draft_ops },
            .{ .name = "engram_hash_core", .module = hash },
        },
    }) });
    b.step("tf-hy4-speed", "Hy4-flash Metal greedy decode: IDs, hash, ms a token, per-stage GPU profile").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
