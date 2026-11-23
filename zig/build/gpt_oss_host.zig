//! gpt-oss slice 1: pure host metadata and arithmetic, with the server's existing Harmony splitter.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const st = b.createModule(.{
            .root_source_file = b.path("zig/src/core/safetensors.zig"),
            .target = b.graph.host,
            .link_libc = true,
        });
        const mx = b.createModule(.{
            .root_source_file = b.path("zig/src/core/mxfp4.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .imports = &.{.{ .name = "safetensors", .module = st }},
        });
        const reply = b.createModule(.{
            .root_source_file = b.path("zig/src/server/reply_text.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .imports = &.{.{ .name = "json", .module = @import("server.zig").json(b, b.graph.host, mode) }},
        });
        const root = b.createModule(.{
            .root_source_file = b.path("zig/src/families/gpt_oss/host_test.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "safetensors", .module = st }, .{ .name = "mxfp4", .module = mx }, .{ .name = "reply_text", .module = reply } },
        });
        const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
        const step = b.step(if (mode == .fast) "test-gpt-oss-fast" else "test-gpt-oss-host", "gpt-oss config, real headers, sinks, routing, exact MXFP4 and shared Harmony; no GPU");
        step.dependOn(&run.step);
        all.dependOn(step);
    }
}
