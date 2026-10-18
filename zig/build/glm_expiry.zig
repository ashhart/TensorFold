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
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = b.graph.host });
    const gathers = b.addTest(.{ .filters = &.{"GLM gathers"}, .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_gather_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    }) });
    b.step("test-glm-gathers", "CPU GLM gather formats, mixed projections and encoder parameters").dependOn(&b.addRunArtifact(gathers).step);
    const cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_gather_cases.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    }) });
    b.step("test-glm-gather-cases", "CPU coverage of the GPU gather gate's routing distributions").dependOn(&b.addRunArtifact(cases).step);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/glm_gather_api.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const gpu_check = b.addObject(.{ .name = "glm-gather-gpu-compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm_gather_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "tensorfold", .module = api } },
    }) });
    b.step("check-glm-gather-gpu-host", "Compile the GPU gather equality gate without compiling or running shaders").dependOn(&gpu_check.step);
    const schedule = b.addTest(.{ .filters = &.{ "glm prefill:", "fill aging:" }, .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_schedule_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "kernel_sources", .module = sources } },
    }) });
    if (b.graph.host.result.os.tag == .macos) schedule.root_module.linkSystemLibrary("objc", .{});
    b.step("test-glm-scheduling-host", "Actual GLM prefill chunks and lane-host decode fairness using a fake engine").dependOn(&b.addRunArtifact(schedule).step);
    if (b.graph.host.result.os.tag == .macos) {
        const compile = b.addObject(.{ .name = "glm-restore-host", .root_module = b.createModule(.{
            .root_source_file = b.path("zig/src/glm_restore_compile.zig"),
            .target = b.graph.host,
            .optimize = .debug,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric } },
        }) });
        b.step("check-glm-restore-host", "Compile the real GLM Metal prefill and restore bodies without shaders or execution").dependOn(&compile.step);
    }
    test_step.dependOn(&run.step);
    b.step("test-glm-expiry", "CPU GLM expiry, borrowed snapshots and peer command ordering").dependOn(&run.step);
}
