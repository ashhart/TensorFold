//! Execute the real family timing policies without loading Metal devices or shaders.
const std = @import("std");
pub fn steps(b: *std.Build, lanes: *std.Build.Module) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    inline for (.{ "nemotron", "glm" }) |family| {
        const m = b.createModule(.{ .root_source_file = b.path("zig/" ++ family ++ "_metal_timing_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "metal", .module = metal } } });
        b.step("test-" ++ family ++ "-metal-timing-host", "Actual " ++ family ++ " timing-controller host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
}
