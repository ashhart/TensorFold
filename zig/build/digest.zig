//! The per-file digest cache's host checks and Linux compile gate never need GPU images or compilers.
const std = @import("std");
pub fn steps(b: *std.Build, host_tests: *std.Build.Step, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    var baseline_query: std.Target.Query = .{ .cpu_arch = b.graph.host.result.cpu.arch, .os_tag = b.graph.host.result.os.tag, .abi = b.graph.host.result.abi, .cpu_model = .baseline };
    if (b.graph.host.result.cpu.arch == .aarch64) baseline_query.cpu_features_sub.addFeature(@backingInt(std.Target.aarch64.Feature.sha2));
    if (b.graph.host.result.cpu.arch == .x86_64) baseline_query.cpu_features_sub.addFeature(@backingInt(std.Target.x86.Feature.sha));
    const baseline = b.resolveTargetQuery(baseline_query);
    const sha = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/runtime_sha256_test.zig"), .target = baseline, .optimize = .fast, .link_libc = true }) });
    const sha_run = b.addRunArtifact(sha);
    host_tests.dependOn(&sha_run.step);
    b.step("test-runtime-sha256-host", "Runtime SHA256 vectors on the baseline CPU without device dependencies").dependOn(&sha_run.step);
    const bench = b.addExecutable(.{ .name = "digest-throughput", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/digest_throughput.zig"), .target = baseline, .optimize = .fast, .link_libc = true }) });
    b.step("compile-digest-throughput-host", "Build the CPU-only digest throughput probe without running it").dependOn(&b.addInstallArtifact(bench, .{}).step);
    const cross_sha = b.step("compile-runtime-sha256-linux", "Compile baseline aarch64 and x86 SHA256 dispatch without execution");
    inline for (.{ .aarch64, .x86_64 }) |arch| {
        const cross = b.resolveTargetQuery(.{ .cpu_arch = arch, .os_tag = .linux, .abi = .gnu, .cpu_model = .baseline });
        const check = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/runtime_sha256_test.zig"), .target = cross, .optimize = .fast, .link_libc = true }) });
        _ = check.getEmittedBin();
        cross_sha.dependOn(&check.step);
    }
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
