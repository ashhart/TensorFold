//! GigaChat 4 family host checks (no GPU).
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/gigachat4/host_test.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    m.addImport("safetensors", b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = .Debug, .link_libc = true }));
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    run.has_side_effects = true; // the files TF_GIGACHAT4_HEADERS and TF_GIGACHAT4_CONFIG_37B name are not cached inputs
    b.step("test-gigachat4-host", "GigaChat 4 family package: config, weight map, layer plan, norm sites and memory floors").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
