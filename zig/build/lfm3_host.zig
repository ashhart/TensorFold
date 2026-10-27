//! LFM3 checkpoint admission checks without Metal compilation or model execution.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const st = b.createModule(.{
        .root_source_file = b.path("zig/src/core/safetensors.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    const checks = b.step("test-lfm3-host", "LFM3 text and VL checkpoint config and tensor admission without a GPU");
    for ([_][]const u8{ "config_test", "weights" }) |part| {
        const root = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/src/families/lfm3/{s}.zig", .{part})),
            .target = b.graph.host,
            .link_libc = true,
        });
        root.addImport("safetensors", st);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
        checks.dependOn(&run.step);
    }
    all.dependOn(checks);
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/lfm3/lfm3.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{.{ .name = "safetensors", .module = st }},
    });
    const tool = b.addExecutable(.{ .name = "tf-lfm3-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/lfm3_check.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "lfm3", .module = family }, .{ .name = "safetensors", .module = st } },
    }) });
    b.step("tf-lfm3-check", "Validate saved LFM3 tensor-header metadata without model data or a GPU")
        .dependOn(&b.addInstallArtifact(tool, .{}).step);
}
