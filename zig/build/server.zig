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

/// The native server (zig/src/server/main.zig) as `name` over `engine` (its engine_api and tokenizer too).
pub fn withEngine(b: *std.Build, target: std.Build.ResolvedTarget, engine: *std.Build.Module, build_options: *std.Build.Step.Options, name: []const u8) *std.Build.Step.Compile {
    const api = engine.import_table.get("engine_api").?;
    const tokenizer = engine.import_table.get("tokenizer").?;
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = engine }} });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engine }, .{ .name = "checkpoint_cli", .module = cli } },
    });
    m.addOptions("build_options", build_options);
    // the server's json and the family's text side share the engine's serving module (one `dsv41_serve`)
    m.addImport("json", engine.import_table.get("dsv41_serve").?.import_table.get("json").?);
    m.addImport("dsv41_serve", engine.import_table.get("dsv41_serve").?);
    return b.addExecutable(.{ .name = name, .root_module = m });
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
