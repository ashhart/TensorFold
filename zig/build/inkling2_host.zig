//! Inkling 2 family host checks (no GPU).
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, root: []const u8) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true });
    const dir = "zig/src/core/";
    m.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path(dir ++ "tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    m.addImport("template", b.createModule(.{ .root_source_file = b.path(dir ++ "template/template.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return m;
}

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/families/inkling2/host_test.zig") }));
    b.step("test-inkling2-host", "Inkling 2 family package: config, weight map, layer plan, refusals, chat template and tokenizer").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
