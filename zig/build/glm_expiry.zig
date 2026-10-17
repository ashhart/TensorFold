//! Check snapshot ownership and peer ordering on host memory, with no Metal frameworks or checkpoint.
const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step, lanes: *std.Build.Module) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const fabric = b.createModule(.{
        .root_source_file = b.path("zig/src/fabric/fabric.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    const tests = b.addTest(.{ .filters = &.{"GLM expiry"}, .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_expiry_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "fabric", .module = fabric },
        },
    }) });
    const run = b.addRunArtifact(tests);
    const compile = b.addObject(.{ .name = "glm-restore-host", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_restore_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric } },
    }) });
    b.step("check-glm-restore-host", "Compile the real GLM Metal prefill and restore bodies without shaders or execution").dependOn(&compile.step);
    test_step.dependOn(&run.step);
    b.step("test-glm-expiry", "CPU GLM expiry, borrowed snapshots and peer command ordering").dependOn(&run.step);
}
