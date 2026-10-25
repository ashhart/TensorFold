//! Only the named device step executes Metal; its compile-only peer emits the same complete executable.
const std = @import("std");
pub fn metal(b: *std.Build) void {
    if (b.graph.host.result.os.tag != .macos) return;
    const target = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const core = b.createModule(.{ .root_source_file = b.path("zig/qwen4_core_host.zig"), .target = target, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api } } });
    const device = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .link_libc = true });
    device.linkFramework("Metal", .{});
    device.linkFramework("Foundation", .{});
    device.linkSystemLibrary("objc", .{});
    const root = b.createModule(.{ .root_source_file = b.path("zig/qwen4_metal_device.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "qwen4", .module = @import("qwen4.zig").module(b, target, .Debug, core) }, .{ .name = "lanes", .module = lanes }, .{ .name = "metal", .module = device } } });
    const exe = b.addExecutable(.{ .name = "qwen4-metal-device", .root_module = root });
    b.step("compile-qwen4-metal-device", "Compile the real Qwen4 Metal device gate without executing it").dependOn(&exe.step);
    const run = b.addRunArtifact(exe);
    run.has_side_effects = true;
    b.step("test-qwen4-metal", "Run nonzero Qwen4 native Metal logits, picks and state exactness on a GPU").dependOn(&run.step);
}
