const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    const imports = [_]std.Build.Module.Import{
        .{ .name = "metal", .module = metal },
        .{ .name = "kernel_sources", .module = sources },
    };
    const exe = b.addExecutable(.{ .name = "tf-metal-counters", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/metal_counters.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &imports,
    }) });
    b.step("tf-metal-counters", "Model-free Metal counter capability and dependent operator probe").dependOn(
        &b.addInstallArtifact(exe, .{}).step,
    );
    const host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/gpu_counters.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    }) });
    const host_run = b.addRunArtifact(host);
    b.step("test-metal-counters-host", "CPU calibration and counter error boundaries").dependOn(&host_run.step);
    test_step.dependOn(&host_run.step);
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/metal_counter_consumer.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
        .imports = &imports,
    }) });
    b.step("test-metal-counters-gpu", "Metal counter lifecycle and dependent operator bit identity").dependOn(
        &b.addRunArtifact(gpu).step,
    );
}
