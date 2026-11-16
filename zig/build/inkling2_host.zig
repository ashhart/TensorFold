//! Inkling 2 family host checks (no GPU).
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, root: []const u8) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true });
}

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/families/inkling2/host_test.zig") }));
    b.step("test-inkling2-host", "Inkling 2 family package: config parsing and its refusals, safetensors headers, weight map and layer plan").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
