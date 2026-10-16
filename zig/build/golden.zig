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
    const golden_run = b.addRunArtifact(golden_check);
    golden_run.addArtifactArg(fake_serve);
    golden_run.addFileArg(b.path("zig/tests/server/golden/cases.json"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/golden"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/fixtures"));
    golden_run.setCwd(b.path("."));
    b.step("test-golden", "The frozen server parity goldens checked by a Zig step driving fake_serve").dependOn(&golden_run.step);
}
