const std = @import("std");
pub fn addImports(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    addImportsAt(b, m, target, .Debug);
}
pub fn addImportsAt(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const core = m.import_table.get("core").?;
    const tokenizer = core.import_table.get("tokenizer").?;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/fixtures/serve/fixtures.zig"), .target = target, .optimize = optimize });
    const serve = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/serve/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "json", .module = json }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "lanes", .module = lanes }, .{ .name = "dsv41_serve_fixtures", .module = fixtures } } });
    m.addImport("lanes", lanes);
    m.addImport("json", json);
    m.addImport("tokenizer", tokenizer);
    m.addImport("dsv41_serve", serve);
    m.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
}
pub fn steps(b: *std.Build) void {
    const source = @import("dsv41_cuda.zig").module(b, b.graph.host, "serve_engine.zig", false);
    b.step("test-draft-dsv41-host", "DeepSeek draft source metadata without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = source, .filters = &.{"draft sources:"} })).step);
    inline for (.{ false, true }) |cross| {
        const target = if (cross) b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) else b.graph.host;
        const root = @import("dsv41_cuda.zig").module(b, target, "device_host_test.zig", cross);
        const tests = b.addTest(.{ .root_module = root });
        if (cross) {
            _ = tests.getEmittedBin();
            b.step("compile-dsv41-device-host", "Emit Linux model, drafting, rows and session adapters").dependOn(&tests.step);
            const cli = b.addExecutable(.{ .name = "tf-dsv41", .root_module = @import("dsv41_cuda.zig").module(b, target, "tool_main.zig", true) });
            _ = cli.getEmittedBin();
            b.step("compile-dsv41-tool", "Emit actual Linux DeepSeek host and GPU-check CLI, never execute").dependOn(&cli.step);
            b.step("compile-dsv41-server", "Emit actual Linux server with the DeepSeek lane engine, never execute").dependOn(&server(b, target).step);
            const sampler = b.addExecutable(.{ .name = "tf-dsv41-samp", .root_module = @import("dsv41_cuda.zig").module(b, target, "sampling_check.zig", true) });
            _ = sampler.getEmittedBin();
            b.step("compile-dsv41-sampler", "Emit the actual Linux sampler check, never execute").dependOn(&sampler.step);
        } else b.step("test-dsv41-device-host", "Host checks of actual row and device plans, no device work").dependOn(&b.addRunArtifact(tests).step);
    }
    inline for (.{ false, true }) |cross| {
        const target = if (cross) b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) else b.graph.host;
        const m = @import("dsv41_cuda.zig").module(b, target, "draft_host_test.zig", cross);
        const test_bin = b.addTest(.{ .root_module = m });
        if (cross) {
            _ = test_bin.getEmittedBin();
            b.step("compile-dsv41-draft", "Emit Linux sampling and host drafting tests").dependOn(&test_bin.step);
        } else b.step("test-dsv41-draft", "Keyed sampling, DSpark and shared lane core host tests").dependOn(&b.addRunArtifact(test_bin).step);
    }
}

fn server(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    return serverOn(b, target, .ReleaseSafe, @import("dsv41_cuda.zig").module(b, target, "serve_engine.zig", true));
}

/// tensorfold-dsv41, the native server over the DeepSeek lane `engine`, its own modules at `optimize`.
pub fn serverOn(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, engine: *std.Build.Module) *std.Build.Step.Compile {
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", @import("../../build.zig.zon").version);
    build_options.addOption(bool, "slide_test_hooks", false);
    const name = "tensorfold-dsv41";
    const api = engine.import_table.get("engine_api").?;
    const tokenizer = engine.import_table.get("tokenizer").?;
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const exlr8 = @import("exlr8.zig").cliImport(b, target, optimize, engine.import_table.get("core").?);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = engine }, .{ .name = "core", .module = engine.import_table.get("core").? }, exlr8 } });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = engine.import_table.get("core").? }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engine }, .{ .name = "checkpoint_cli", .module = cli } },
    });
    m.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    m.addOptions("build_options", build_options);
    // the server's json and the family's text side share the engine's serving module (one `dsv41_serve`)
    m.addImport("json", engine.import_table.get("dsv41_serve").?.import_table.get("json").?);
    m.addImport("dsv41_serve", engine.import_table.get("dsv41_serve").?);
    m.addImport("dsv41_chat", engine.import_table.get("dsv41_serve").?);
    // the server's Drama voice catalog reads the prompt builder; this engine set has no drama3 family to share
    m.addImport("drama_prompt", b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/prompt.zig"), .target = target, .optimize = optimize }));
    const exe = b.addExecutable(.{ .name = name, .root_module = m });
    _ = exe.getEmittedBin();
    return exe;
}
