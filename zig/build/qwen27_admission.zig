//! Qwen27 Metal config, tied head widths and routing on every host, without shaders or a device.
const std = @import("std");

pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_admission_host.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    root.addImport("core", root);
    root.addImport("tensorfold", root);
    const run = b.addRunArtifact(b.addTest(.{
        .root_module = root,
        .filters = &.{ "config_test.", "a qwen3_5 checkpoint opens" },
    }));
    b.step("test-qwen27-metal-config-host", "Qwen27 Metal config, tied embedding widths and engine routing without a GPU").dependOn(&run.step);
    parent.dependOn(&run.step);
}
