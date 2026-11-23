//! `pull`'s and `rm`'s own tests: `cli/hub.zig` reads the engine registry (`native_engines`); pull also needs EXLR8's.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, engines: *std.Build.Module, core: *std.Build.Module, test_step: *std.Build.Step) void {
    const route = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/pull_route.zig"), .target = target, .optimize = optimize, .link_libc = true }) }));
    b.step("test-pull-route-host", "Proxy configuration, refusal names and routing rules without a network").dependOn(&route.step);
    test_step.dependOn(&route.step);
    // rm's tests make their own cache of links, hard links and copies, so they run on every OS
    const rm = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/rm_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "native_engines", .module = engines }},
    }) }));
    b.step("test-rm-host", "Checkpoint removal in a temporary cache: shared blobs, refs, dry runs, refusals and links out").dependOn(&rm.step);
    test_step.dependOn(&rm.step);
    if (target.result.os.tag != .windows) { // the journal fixture listens on a POSIX socket
        const journal_log = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/journal_log.zig"), .target = target, .optimize = optimize }) }));
        b.step("test-journal-log-host", "tensorfold log over a fixture request journal: list, show, --json and follow across a rotation").dependOn(&journal_log.step);
        test_step.dependOn(&journal_log.step);
    }
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/pull_host_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "native_engines", .module = engines }, @import("exlr8.zig").cliImport(b, target, optimize, core) },
    });
    const lock_test = b.addRunArtifact(b.addTest(.{ .root_module = mod, .filters = &.{"pull refuses cache mutation"} }));
    b.step("test-pull-cache-lock-host", "Cache mutation refusal before any HTTP request").dependOn(&lock_test.step);
    test_step.dependOn(&lock_test.step);
    const tests = b.addTest(.{ .name = "tf-pull-host-tests", .root_module = mod });
    const run = b.addRunArtifact(tests);
    b.step("test-pull-host", "Checkpoint download paths: a nested shard folder, retries, resume and the whole-file fallback").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const security = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/pull_security_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "native_engines", .module = engines }, @import("exlr8.zig").cliImport(b, target, optimize, core) },
    }) }));
    test_step.dependOn(&security.step);
}

pub fn compile(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module) void {
    // Host checks always use b.graph.host; compile this separate artifact and all its imports for -Dtarget.
    const cuda = @import("cuda.zig");
    const runtime = cuda.runtime(b, target, .debug, &.{});
    const mods = cuda.family(b, target, .debug, runtime, draft_ids);
    const native = @import("cuda_engines.zig").engines(b, target, .debug, runtime, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    const tests = b.addTest(.{ .name = "tf-pull-host-tests", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/pull_host_test.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "native_engines", .module = native.engines }, @import("exlr8.zig").cliImport(b, target, .debug, mods.core) },
    }) });
    const install = b.addInstallArtifact(tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } });
    b.step("compile-pull-host", "Compile and install the pull host tests for -Dtarget without executing them").dependOn(&install.step);
}
