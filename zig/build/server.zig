//! The server's `json` module and a family's serving module beside it: `test-dsv41-serve` and `tf-dsv41-tokbench`.
const std = @import("std");

/// The server's JSON values as a module (zig/src/server/json.zig), so a family's serving code shares them.
pub fn json(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize });
}

/// The family's serving module (tokenizer, encoding, DSML tools, sampling) over the modules the server shares.
pub fn serveModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, json_mod: *std.Build.Module, tokenizer: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/fixtures/serve/fixtures.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/serve/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "json", .module = json_mod },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "dsv41_serve_fixtures", .module = fixtures },
        },
    });
}

/// The serving module with its own `json` and `tokenizer` over `lanes` (family tests and tools beside no server).
pub fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tokenizer: *std.Build.Module, lanes: *std.Build.Module) struct { serve: *std.Build.Module, json: *std.Build.Module } {
    const j = json(b, target, optimize);
    return .{ .serve = serveModule(b, target, optimize, j, tokenizer, lanes), .json = j };
}

/// A server module's `json` and `dsv41_serve` imports over its `tokenizer`; returns `json` (its tests run apart).
pub fn addTo(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tokenizer: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const s = family(b, target, optimize, tokenizer, lanes);
    m.addImport("json", s.json);
    m.addImport("dsv41_serve", s.serve);
    return s.json;
}

/// `test-dsv41-serve` (in `test`) and `tf-dsv41-tokbench` on the host.
pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const step = b.step("test-dsv41-serve", "DeepSeek-V4.1's serving side: tokenizer, encoding, DSML tools, sampling (goldens: TF_DSV41_MODEL, TF_DSV41_GOLDEN)");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .Debug) })).step);
    test_step.dependOn(step);
    const bench = b.addExecutable(.{ .name = "tf-dsv41-tokbench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/dsv41/tokbench.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{.{ .name = "dsv41_serve", .module = module(b, host, .ReleaseFast) }},
    }) });
    b.step("tf-dsv41-tokbench", "The DeepSeek-V4.1 tokenizer's throughput over gen_tokenizer.py's corpus").dependOn(&b.addInstallArtifact(bench, .{}).step);
}

/// The serving module alone, over its own tokenizer and lanes.
fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    return family(b, target, optimize, tokenizer, lanes).serve;
}
