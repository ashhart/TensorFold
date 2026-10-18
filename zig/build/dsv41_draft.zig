const std = @import("std");
pub fn addImports(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    const core = m.import_table.get("core").?;
    const tokenizer = core.import_table.get("tokenizer").?;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/fixtures/serve/fixtures.zig"), .target = target, .optimize = .Debug });
    const serve = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/serve/root.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "json", .module = json }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "lanes", .module = lanes }, .{ .name = "dsv41_serve_fixtures", .module = fixtures } } });
    m.addImport("lanes", lanes);
    m.addImport("json", json);
    m.addImport("tokenizer", tokenizer);
    m.addImport("dsv41_serve", serve);
    m.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
}
pub fn steps(b: *std.Build) void {
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
