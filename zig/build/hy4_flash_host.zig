//! Hy4-flash family host checks: config, layer plan, weight map, refusals and chat template. No GPU.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/host_test.zig"),
        .target = target,
        .link_libc = true,
    });
    root.addImport("template", b.createModule(.{
        .root_source_file = b.path("zig/src/core/template/template.zig"),
        .target = target,
        .link_libc = true,
    }));
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-hy4-flash-host", "Hy4-flash family package: config, layer plan, weight map, refusals and chat template").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
