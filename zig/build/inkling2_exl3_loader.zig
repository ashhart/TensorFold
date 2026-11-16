//! EXL3 loader compilation and host validation, separate from Inkling execution.
const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const root = "zig/src/families/inkling2/";
    const checks = b.createModule(.{ .root_source_file = b.path(root ++ "exl3_loader_test.zig"), .target = b.graph.host, .link_libc = true });
    checks.addImport("cuda", cuda);
    checks.addImport("core", core);
    b.step("test-inkling2-exl3-loader", "EXL3 manifest/source agreement and shared family plan").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = checks })).step);
    const m = b.createModule(.{ .root_source_file = b.path(root ++ "exl3_load_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    const exe = b.addExecutable(.{ .name = "tf-inkling2-exl3-load", .root_module = m });
    b.step("check-inkling2-exl3-loader", "Compile the complete CUDA loader without running it").dependOn(&exe.step);
    b.step("tf-inkling2-exl3-load", "Build a bounded layer-prefix load canary").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
