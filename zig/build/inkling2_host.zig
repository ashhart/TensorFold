//! Inkling 2 family host checks (no GPU).
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, root: []const u8) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true });
    const dir = "zig/src/core/";
    const tokenizer = b.createModule(.{ .root_source_file = b.path(dir ++ "tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("tokenizer", tokenizer);
    const core = b.createModule(.{ .root_source_file = b.path(dir ++ "root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    core.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    m.addImport("core", core);
    m.addImport("template", b.createModule(.{ .root_source_file = b.path(dir ++ "template/template.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return m;
}

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, .Debug, "zig/src/families/inkling2/host_test.zig") }));
    b.step("test-inkling2-host", "Inkling 2 family package: config, weight map, layer plan, chat template and tokenizer").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
