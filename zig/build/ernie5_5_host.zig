//! ERNIE 5.5 family host checks (no GPU).
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, root: []const u8) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = .Debug, .link_libc = true });
    const dir = "zig/src/core/";
    m.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path(dir ++ "tokenizer/tokenizer.zig"), .target = target, .optimize = .Debug, .link_libc = true }));
    m.addImport("safetensors", b.createModule(.{ .root_source_file = b.path(dir ++ "safetensors.zig"), .target = target, .optimize = .Debug, .link_libc = true }));
    return m;
}

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, target, "zig/src/families/ernie5_5/host_test.zig") }));
    b.step("test-ernie5_5-host", "ERNIE 5.5 family package: config, weight map, layer plan, chat encoding and tokenizer").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
