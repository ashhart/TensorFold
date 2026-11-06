//! The documented TF_* environment defaults, pinned without a GPU: one step over the readers' own test roots.
const std = @import("std");
const cuda_build = @import("cuda.zig");
const drama3_host = @import("drama3_host.zig");

pub fn steps(b: *std.Build, draft_ids: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = cuda_build.runtime(b, host, .ReleaseSafe, &.{});
    const mods = cuda_build.family(b, host, .ReleaseSafe, cuda, draft_ids);
    const step = b.step("test-env-defaults-host", "Documented TF_* environment defaults without devices");
    const native_test = b.createModule(.{
        .root_source_file = b.path("zig/src/native/env_defaults_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = mods.api },
            .{ .name = "drama3", .module = drama3_host.module(b, host, .debug, cuda, mods.core) },
            .{ .name = "core", .module = mods.core },
            .{ .name = "cuda", .module = cuda },
        },
    });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_test })).step);
    const glm_test = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm/env_defaults_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "heat", .module = mods.heat },
        },
    });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm_test })).step);
    all.dependOn(step);
}
