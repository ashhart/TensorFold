//! Existing GLM family recording controls on the CPU runtime.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    const host = b.graph.host;
    const kv8_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM FP8 KV:"} }));
    b.step("test-glm-fp8-kv-host", "GLM latent cache layout, mode agreement and recorded CUDA launches without devices").dependOn(&kv8_run.step);
    step.dependOn(&kv8_run.step);
    const profile_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM prompt profile:"} }));
    b.step("test-glm-prompt-profile-host", "Paired prompt-stage event timing, overlap, failure and disabled controls without devices").dependOn(&profile_run.step);
    step.dependOn(&profile_run.step);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const admission = b.createModule(.{ .root_source_file = b.path("zig/glm_admission_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "engine_api", .module = api } } });
    const admission_run = b.addRunArtifact(b.addTest(.{ .root_module = admission, .filters = &.{"GLM CUDA admission:"} }));
    b.step("test-glm-admission-host", "Real GLM admission capabilities through the lane worker without GPU work").dependOn(&admission_run.step);
    step.dependOn(&admission_run.step);
    const copies = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_snaps.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "engine_api", .module = api }} });
    const snapshots = b.createModule(.{ .root_source_file = b.path("zig/glm_snapshot_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "engine_api", .module = api }, .{ .name = "copies", .module = copies } } });
    b.step("test-glm-snapshots-host", "Prompt-sized GLM state copies, page budgets and rank ordering without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = snapshots })).step);
    b.step("test-glm-prompt-settings-host", "Actual opt-in settings and rank agreement on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"prompt K experiment"} })).step);
    const mm_record = b.createModule(.{ .root_source_file = b.path("zig/test_support/glm_mm_cuda.zig"), .target = host, .optimize = .debug });
    const mm_actual = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_mm.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = mm_record }} });
    const mm_policy = b.createModule(.{ .root_source_file = b.path("zig/glm_mm_policy_host.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = mm_record }, .{ .name = "mm", .module = mm_actual } } });
    b.step("test-glm-mm-policy-host", "Actual prompt K policy dispatch without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mm_policy })).step);
    const drafter_host = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-dflash-host", "DFlash rank buffers, partial loader ownership and pending records on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = drafter_host, .filters = &.{"glm dflash:"} })).step);
    const packed_host = b.createModule(.{ .root_source_file = b.path("zig/glm_packed_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-packed-host", "Packed GLM launch records, cap parsing and native loader on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = packed_host, .filters = &.{"glm packed:"} })).step);
    const ring_host = b.createModule(.{ .root_source_file = b.path("zig/glm_ring_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    b.step("test-glm-ring-host", "Actual GLM MTP rollback pool through recording CUDA copies").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ring_host, .filters = &.{"glm ring:"} })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    const prompt_host = b.createModule(.{ .root_source_file = b.path("zig/exl3_prompt_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes } } });
    const prompt_run = b.addRunArtifact(b.addTest(.{ .root_module = prompt_host, .filters = &.{"exl3 prompt:"} }));
    prompt_run.has_side_effects = true;
    b.step("test-exl3-prompt-host", "Actual common prompt dispatch and scratch with recording CUDA calls").dependOn(&prompt_run.step);
}

pub fn compile(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, comptime makeGlm: anytype) void {
    const fp8_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_ref_fp8.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    const fp8_object = b.addObject(.{ .name = "glm-fp8-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_fp8_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = fp8_gate }} }) });
    _ = fp8_object.getEmittedBin();
    b.step("compile-glm-fp8-gate-host", "Compile synthetic FP8 write/read and FP64 device checks with inert images, never execute").dependOn(&fp8_object.step);
    const glm_open = b.addObject(.{ .name = "glm-open-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_open_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }} }) });
    _ = glm_open.getEmittedBin();
    b.step("compile-glm-native-open", "Compile the actual GLM Linux open body with empty GPU images").dependOn(&glm_open.step);
    const prompt_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_mm_fixed_tests.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }, .{ .name = "cuda", .module = cuda } } });
    const prompt_gate_compile = b.addObject(.{ .name = "glm-prompt-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = prompt_gate }} }) });
    _ = prompt_gate_compile.getEmittedBin();
    b.step("compile-glm-prompt-gate-host", "Compile the actual device prompt comparison gate with inert images").dependOn(&prompt_gate_compile.step);
}
