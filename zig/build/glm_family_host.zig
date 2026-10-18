//! Existing GLM family recording controls on the CPU runtime.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    const host = b.graph.host;
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
