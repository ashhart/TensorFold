const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, host_tests: *std.Build.Step) void {
    const files = b.addWriteFiles();
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection.metal"), "exl3_projection.metal");
    const sources = b.createModule(.{
        .root_source_file = files.add("sources.zig", "pub const core_exl3_projection = @embedFile(\"exl3_projection.metal\");\n"),
    });
    const imports: []const std.Build.Module.Import =
        &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } };
    const projection = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exl3_projection.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = imports,
    });
    const bench = b.addExecutable(.{ .name = "tf-exl3-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/exl3_projection_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "exl3_projection", .module = projection } },
    }) });
    b.step("tf-exl3-bench", "Standalone deterministic EXL3 FP64 oracle and relative operator benchmark")
        .dependOn(&b.addInstallArtifact(bench, .{}).step);
    const host = b.addRunArtifact(b.addTest(.{ .root_module = projection }));
    host.setCwd(b.path("."));
    const oracle_tests = b.addRunArtifact(b.addTest(.{ .root_module = bench.root_module }));
    oracle_tests.setCwd(b.path("."));
    const host_step = b.step("test-exl3", "EXL3 extraction, codebook, shape, storage and oracle host contracts");
    host_step.dependOn(&host.step);
    host_step.dependOn(&oracle_tests.step);
    host_tests.dependOn(host_step);
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exl3_projection_gpu_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = imports,
    }) });
    const gpu_run = b.addRunArtifact(gpu);
    gpu_run.setCwd(b.path("."));
    b.step("test-exl3-gpu", "EXL3 Metal arithmetic, row windows, routes and validation regressions")
        .dependOn(&gpu_run.step);
}
