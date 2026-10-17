//! The per-file digest cache's host checks and Linux compile gate never need GPU images or compilers.
const std = @import("std");
pub fn steps(b: *std.Build, host_tests: *std.Build.Step) void {
    inline for (.{ false, true }) |cross| {
        const target = if (cross) b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) else b.graph.host;
        const root = b.createModule(.{ .root_source_file = b.path("zig/src/core/digest_cache_test.zig"), .target = target, .optimize = .Debug, .link_libc = true });
        const writer = b.addExecutable(.{ .name = if (cross) "digest-writer-linux" else "digest-writer", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/digest_cache_probe.zig"), .target = target, .optimize = .Debug, .link_libc = true }) });
        const options = b.addOptions();
        options.addOptionPath("writer", writer.getEmittedBin());
        root.addOptions("digest_test_options", options);
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
