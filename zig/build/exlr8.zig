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
