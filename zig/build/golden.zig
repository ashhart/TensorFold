//! The frozen server parity goldens: a Zig check drives fake_serve over recorded answers, on macOS and Linux hosts.

const std = @import("std");

/// `zig build test-golden` over the server tests' module and the imports it was built with.
pub fn step(b: *std.Build, target: std.Build.ResolvedTarget, server: *std.Build.Module, api: *std.Build.Module, tokenizer: *std.Build.Module, template: *std.Build.Module) void {
    const fake_serve = b.addExecutable(.{ .name = "fake_serve", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/fake_serve.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "server", .module = server }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } },
    }) });
    // golden_check calls getpid for its scratch folder, so it links libc (macOS links it anyway)
    const golden_check = b.addExecutable(.{ .name = "golden_check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/golden_check.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const all = b.step("test-golden", "Frozen parity answers and native HTTP security goldens");
    for ([_][]const u8{ "cases.json", "security-cases.json" }) |file| {
        const run = b.addRunArtifact(golden_check);
        run.addArtifactArg(fake_serve);
        run.addFileArg(b.path(b.fmt("zig/tests/server/golden/{s}", .{file})));
        run.addDirectoryArg(b.path("zig/tests/server/golden"));
        run.addDirectoryArg(b.path("zig/tests/server/fixtures"));
        run.setCwd(b.path("."));
        all.dependOn(&run.step);
        if (std.mem.eql(u8, file, "security-cases.json")) b.step("test-security-golden-host", "Native HTTP key, host and origin security with the local fake server").dependOn(&run.step);
    }
    const security = b.step("test-server-security", "API authentication and hostile HTTP framing on the CPU");
    const wire = b.addSystemCommand(&.{"python3"});
    wire.addFileArg(b.path("zig/tests/server/security_check.py"));
    wire.addArtifactArg(fake_serve);
    wire.addArg("--fixtures");
    wire.addDirectoryArg(b.path("zig/tests/server/fixtures"));
    wire.setCwd(b.path("."));
    security.dependOn(&wire.step);
    security.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"security:"} })).step);
    const pull = b.step("test-pull-security", "Checkpoint download paths, byte limits and metadata refusal on the CPU");
    for ([_][]const u8{ "zig/src/cli/hub.zig", "zig/src/cli/pull_test.zig" }) |path| {
        const checks = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(path),
            .target = target,
            .optimize = .Debug,
            .link_libc = true,
        }), .filters = &.{"security:"} });
        const run = b.addRunArtifact(checks);
        run.setCwd(b.path("."));
        pull.dependOn(&run.step);
    }
    const core = b.createModule(.{
        .root_source_file = b.path("zig/src/core/root.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "tokenizer", .module = tokenizer }},
    });
    const oracle = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/cuda_checks.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }), .filters = &.{"security:"} });
    const oracle_run = b.addRunArtifact(oracle);
    oracle_run.setCwd(b.path("."));
    pull.dependOn(&oracle_run.step);
}
