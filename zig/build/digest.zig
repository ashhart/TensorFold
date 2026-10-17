//! The per-file digest cache's host checks and Linux compile gate never need GPU images or compilers.
const std = @import("std");
pub fn steps(b: *std.Build, host_tests: *std.Build.Step, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
    const server = @import("cuda.zig").distServer(b, target, draft_ids, build_options, null);
    _ = server.getEmittedBin();
    b.step("compile-checkpoint-consumers", "Compile actual Linux server and family checkpoint consumers without GPU images").dependOn(&server.step);
    inline for (.{ false, true }) |cross| {
        const test_target = if (cross) b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) else b.graph.host;
        const root = b.createModule(.{ .root_source_file = b.path("zig/src/core/digest_cache_test.zig"), .target = test_target, .optimize = .Debug, .link_libc = true });
        const writer = b.addExecutable(.{ .name = if (cross) "digest-writer-linux" else "digest-writer", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/digest_cache_probe.zig"), .target = test_target, .optimize = .Debug, .link_libc = true }) });
        const options = b.addOptions();
        options.addOptionPath("writer", writer.getEmittedBin());
        root.addOptions("digest_test_options", options);
        const identities = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_identity.zig"), .target = test_target, .optimize = .Debug, .link_libc = true }) });
        if (cross) {
            _ = identities.getEmittedBin();
            b.step("compile-checkpoint-identity", "Compile Linux whole and projected checkpoint readers").dependOn(&identities.step);
        } else {
            const identity_run = b.addRunArtifact(identities);
            b.step("test-checkpoint-identity", "Cached whole and split checkpoint identity checks").dependOn(&identity_run.step);
            host_tests.dependOn(&identity_run.step);
        }
        const filter = b.option([]const u8, if (cross) "digest-linux-filter" else "digest-filter", "One digest regression");
        const exe = b.addTest(.{ .root_module = root, .filters = if (filter) |f| &.{f} else &.{} });
        if (cross) {
            _ = exe.getEmittedBin();
            b.step("compile-digest-cache", "Compile Linux descriptor and cache tests without executing").dependOn(&exe.step);
        } else {
            const run = b.addRunArtifact(exe);
            b.step("test-digest-cache", "Digest persistence, invalidation, corruption and competing writers").dependOn(&run.step);
            host_tests.dependOn(&run.step);
        }
    }
}
