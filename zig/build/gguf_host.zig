//! GGUF byte-reader and repack checks, independent of GPU modules and compiled assets.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.step("test-gguf-host", "Checked GGUF containers and typed metadata without a GPU");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const m = b.createModule(.{ .root_source_file = b.path("zig/src/core/gguf.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true });
        host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
        if (mode == .debug) {
            const points = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"GGUF repack: deterministic"} }));
            points.has_side_effects = true;
            b.step("test-gguf-repack-points", "Original GGUF values against compact affine blocks without a GPU").dependOn(&points.step);
        }
    }
    all.dependOn(host);
    const family = b.step("test-qwen27-gguf-host", "Qwen27 GGUF metadata, names and exact affine preparation without a GPU");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_gguf_core.zig"), .target = b.graph.host, .optimize = mode });
        const hub = b.createModule(.{ .root_source_file = b.path("zig/src/server/hub.zig"), .target = b.graph.host, .optimize = mode });
        const root = b.createModule(.{ .root_source_file = b.path("zig/qwen27_gguf_host.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "server_hub", .module = hub } } });
        family.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Qwen27 GGUF:"} })).step);
    }
    all.dependOn(family);
    const linux = b.createModule(.{ .root_source_file = b.path("zig/src/core/gguf.zig"), .target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu }), .optimize = .Debug, .link_libc = true });
    b.step("test-gguf-linux-compile", "Compile GGUF host checks for Linux aarch64").dependOn(&b.addTest(.{ .root_module = linux }).step);
}
