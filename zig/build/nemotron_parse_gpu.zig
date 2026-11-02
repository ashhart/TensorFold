//! The focused HIP operator checks; no model server runs in this build target.
const std = @import("std");

pub fn gpuTests(b: *std.Build, target: std.Build.ResolvedTarget, hip: *std.Build.Module, object: std.Build.LazyPath) void {
    const files = b.addWriteFiles();
    _ = files.addCopyFile(object, "vision.hsaco");
    const fixture = b.createModule(.{ .root_source_file = files.add("vision.zig", "const data align(8) = @embedFile(\"vision.hsaco\").*;\npub const bytes: []align(8) const u8 = &data;\n") });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/families/nemotron_parse/hip_ops_test.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = hip.import_table.get("core").? }, .{ .name = "parse_gpu", .module = fixture } },
    });
    const test_exe = b.addTest(.{ .root_module = mod });
    b.step("hip-parse-ops-build", "Compile the shared Parse image operators and focused HIP checks").dependOn(&test_exe.step);
    b.step("hip-parse-ops-test", "Run Parse image operators on the selected HIP device").dependOn(&b.addRunArtifact(test_exe).step);
}

pub fn modelTests(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, hip: *std.Build.Module) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/families/nemotron_parse/probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
    const exe = b.addExecutable(.{ .name = "tf-nemotron-parse-test", .root_module = mod });
    b.step("tf-nemotron-parse-test", "Build the native HIP Parse model-seam probe").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
