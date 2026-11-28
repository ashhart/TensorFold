//! Rank-secret path, diagnostics and partial liveness ownership checks without a network or device.
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, tp: *std.Build.Module) void {
    const root = tp.import_table.get("cuda").?;
    const config_module = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/config.zig"), .target = host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = root }} });
    const config = b.addRunArtifact(b.addTest(.{ .root_module = config_module, .filters = &.{"rank secret settings:"} }));
    const lines = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"rank liveness cleanup:"} }));
    const probe = b.addExecutable(.{ .name = "rank-secret-diagnostic", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/rank_secret_diagnostic.zig"), .target = host, .link_libc = true, .imports = &.{.{ .name = "tp", .module = config_module }} }) });
    const run = b.addRunArtifact(probe);
    run.expectExitCode(0);
    const stderr = run.captureStdErr(.{});
    const diagnostic = b.addSystemCommand(&.{ "python3", "-c", "import pathlib,sys; data=pathlib.Path(sys.argv[1]).read_text(); assert 'diagnostic-private' not in data, 'rank key path leaked'; assert 'TF_RANK_SECRET_FILE' in data; print('rank secret diagnostics: setting named, path omitted')" });
    diagnostic.addFileArg(stderr);
    const owner = b.step("test-rank-secret-settings-host", "Validated secret paths, redacted production diagnostics and partial liveness cleanup");
    owner.dependOn(&config.step);
    owner.dependOn(&lines.step);
    owner.dependOn(&probe.step);
    owner.dependOn(&diagnostic.step);
}

/// Production-mode root discovery is required for the CLI's @import("root").serves capability table.
pub fn cliTests(b: *std.Build, host: std.Build.ResolvedTarget, server: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/server/rank_secret_config_probe.zig"), .target = host, .link_libc = true });
    for (server.import_table.keys(), server.import_table.values()) |name, module| root.addImport(name, module);
    const config = b.addRunArtifact(b.addExecutable(.{ .name = "rank-secret-config", .root_module = root }));
    config.expectExitCode(0);
    const capabilities = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"capability audit: environment declares every variable serve reads"} }));
    const owner = b.step("test-rank-secret-cli-host", "Rank-capable CLI metadata and configuration-relative key paths");
    owner.dependOn(&capabilities.step);
    owner.dependOn(&config.step);
}
