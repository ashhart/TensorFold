//! GGUF byte-reader and repack checks, independent of GPU modules and compiled assets.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const admission = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/weight_only.zig"), .target = b.graph.host, .optimize = .debug }) }));
    b.step("test-weight-only-admission-host", "Generic/CUDA group refusals and the isolated Metal group16 admission").dependOn(&admission.step);
    all.dependOn(&admission.step);
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
    const proof_core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_gguf_core.zig"), .target = b.graph.host, .optimize = .ReleaseFast });
    const proof_family = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/gguf_projection.zig"), .target = b.graph.host, .optimize = .ReleaseFast, .imports = &.{.{ .name = "core", .module = proof_core }} });
    const proof_root = b.createModule(.{ .root_source_file = b.path("zig/tools/qwen27_gguf_check.zig"), .target = b.graph.host, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{ .{ .name = "core", .module = proof_core }, .{ .name = "preparation", .module = proof_family } } });
    const proof = b.addExecutable(.{ .name = "tf-qwen27-gguf-check", .root_module = proof_root });
    b.step("tf-qwen27-gguf-check", "Check decoded first-layer GGUF projections against binary64 without a GPU").dependOn(&b.addInstallArtifact(proof, .{}).step);
    if (b.graph.host.result.os.tag == .macos) {
        const reference = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/gguf_cpu_reference.zig"), .target = b.graph.host, .optimize = .ReleaseFast, .imports = &.{.{ .name = "core", .module = proof_core }} });
        const reference_root = b.createModule(.{ .root_source_file = b.path("zig/tools/qwen27_gguf_reference.zig"), .target = b.graph.host, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{.{ .name = "reference", .module = reference }} });
        reference_root.linkFramework("Accelerate", .{});
        reference.linkFramework("Accelerate", .{});
        const ref_test = b.addTest(.{ .root_module = reference });
        const reference_checks = b.addRunArtifact(ref_test);
        b.step("test-qwen27-gguf-reference-host", "Binary64 CPU projection and activation checks without a GPU").dependOn(&reference_checks.step);
        all.dependOn(&reference_checks.step);
        const ref_exe = b.addExecutable(.{ .name = "tf-qwen27-gguf-reference", .root_module = reference_root });
        b.step("tf-qwen27-gguf-reference", "Binary64 CPU reference over decoded GGUF weights and fixed prompt ids").dependOn(&b.addInstallArtifact(ref_exe, .{}).step);
    }
    const family = b.step("test-qwen27-gguf-host", "Qwen27 GGUF metadata, names and exact affine preparation without a GPU");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_gguf_core.zig"), .target = b.graph.host, .optimize = mode });
        const hub = b.createModule(.{ .root_source_file = b.path("zig/src/server/hub.zig"), .target = b.graph.host, .optimize = mode });
        const root = b.createModule(.{ .root_source_file = b.path("zig/qwen27_gguf_host.zig"), .target = b.graph.host, .optimize = mode, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "server_hub", .module = hub } } });
        family.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Qwen27 GGUF:"} })).step);
    }
    all.dependOn(family);
    const text_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .fast, .link_libc = true });
    const compare_root = @import("server.zig").plain(b, "zig/gguf_tokenizer_compare.zig", b.graph.host, .fast, text_lanes);
    compare_root.addImport("core", compare_root);
    const compare = b.addExecutable(.{ .name = "tf-gguf-tokenizer-compare", .root_module = compare_root });
    b.step("gguf-tokenizer-compare-host", "Compare actual GGUF and MLX vocabulary, encoding, decoding, stops and rendered chat without a GPU").dependOn(&b.addInstallArtifact(compare, .{}).step);
    const linux = b.createModule(.{ .root_source_file = b.path("zig/src/core/gguf.zig"), .target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu }), .optimize = .Debug, .link_libc = true });
    b.step("test-gguf-linux-compile", "Compile GGUF host checks for Linux aarch64").dependOn(&b.addTest(.{ .root_module = linux }).step);
}
