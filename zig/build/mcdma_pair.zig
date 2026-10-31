//! tf-mcdma-pair (zig/tests/mcdma_pair.zig), the #1594 gate's harness over MCDMA: for this target and each rank's host.

const std = @import("std");

/// Linked against glibc 2.35 (Ubuntu 22.04; DGX OS 7 on 24.04 has 2.39); dlopen is in libc.so.6 itself.
const glibc: std.SemanticVersion = .{ .major = 2, .minor = 35, .patch = 0 };

/// ReleaseSafe whatever -Doptimize says: a gate keeps its safety checks; `sys` comes from sys.zig's step-graph walk.
fn exe(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    return b.addExecutable(.{ .name = "tf-mcdma-pair", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/mcdma_pair.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{.{ .name = "fabric", .module = fabric }},
    }) });
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget) void {
    b.step("tf-mcdma-pair", "The #1594 gate over MCDMA, one rank per process (ConnectX or Thunderbolt), for this target").dependOn(&b.addInstallArtifact(exe(b, target), .{}).step);
    const dist = b.step("tf-mcdma-pair-dist", "tf-mcdma-pair for every rank's host: aarch64-macos, aarch64-linux-gnu (GB10) and x86_64-linux-gnu (glibc 2.34 or later)");
    const hosts = [_]struct { std.Target.Query, []const u8 }{
        .{ .{ .cpu_arch = .aarch64, .os_tag = .macos, .os_version_min = .{ .semver = .{ .major = 26, .minor = 0, .patch = 0 } } }, "aarch64-macos" }, // any Apple Silicon Mac
        .{ .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu, .glibc_version = glibc }, "aarch64-linux-gnu" },
        .{ .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu, .glibc_version = glibc }, "x86_64-linux-gnu" },
    };
    for (hosts) |h| {
        const install = b.addInstallArtifact(exe(b, b.resolveTargetQuery(h[0])), .{ .dest_dir = .{ .override = .{ .custom = b.fmt("bin/mcdma-pair/{s}", .{h[1]}) } } });
        dist.dependOn(&install.step);
    }
}
