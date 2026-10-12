const std = @import("std");

/// EXLR8's host tests: the reader and decoder against the reference loader's fixtures (TF_EXLR8_DIR adds the layers).
pub fn hostTests(b: *std.Build, target: std.Build.ResolvedTarget, host_tests: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exlr8_oracle_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    }) });
    const run = b.addRunArtifact(tests);
    run.setCwd(b.path("."));
    b.step("test-exlr8", "EXLR8 reader and decoder against the reference loader's layer fixtures").dependOn(&run.step);
    host_tests.dependOn(&run.step);
}

/// The Metal projections' synthetic check program and its host tests.
pub fn metalTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, host_tests: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/exlr8.zig"), .target = target, .optimize = optimize });
    const projection = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exlr8_projection.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "exlr8", .module = core } },
    });
    host_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = projection })).step);
    const check = b.addExecutable(.{ .name = "tf-exlr8-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exlr8_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "exlr8", .module = core }, .{ .name = "exlr8_projection", .module = projection } },
    }) });
    b.step("tf-exlr8-check", "EXLR8 Metal projections: codebook, FP64 bounds, row windows, frozen bytes (no models)").dependOn(&b.addInstallArtifact(check, .{}).step);
    b.step("test-exlr8-gpu", "Run the EXLR8 Metal projection checks on this Mac's GPU").dependOn(&b.addRunArtifact(check).step);
    const bench = b.addExecutable(.{ .name = "tf-exlr8-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exlr8_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "exlr8", .module = core }, .{ .name = "exlr8_projection", .module = projection } },
    }) });
    b.step("tf-exlr8-bench", "EXLR8 Metal projections on a real MoE layer: FP64, row windows, then GB/s by stage").dependOn(&b.addInstallArtifact(bench, .{}).step);
}
