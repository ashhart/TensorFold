//! CPU controls compile the same nullable-address helper that the CUDA backup kernels consume.
const std = @import("std");

pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const module = b.createModule(.{ .target = b.graph.host, .optimize = .debug, .link_libc = true });
    module.addCSourceFile(.{ .file = b.path("zig/tests/cuda/qwen_before_host.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    const exe = b.addExecutable(.{ .name = "qwen-before-host", .root_module = module });
    const step = b.step("test-qwen35-before-host", "Actual nullable backup resolver and planted controls without a device");
    step.dependOn(&b.addRunArtifact(exe).step);
    for ([_][]const u8{ "--plant-null-offset", "--plant-legacy-presence" }) |control| {
        const red = b.addRunArtifact(exe);
        red.addArg(control);
        red.expectExitCode(1);
        step.dependOn(&red.step);
    }
    parent.dependOn(step);
}
