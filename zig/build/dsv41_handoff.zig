const std = @import("std");

fn root(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/handoff_host.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

pub fn steps(b: *std.Build) void {
    const host = b.addTest(.{ .root_module = root(b, b.graph.host, .Debug) });
    b.step("test-dsv41-handoff", "DeepSeek-V4.1 CUDA-to-Metal hand-off: record conversions and part plan (host only)").dependOn(&b.addRunArtifact(host).step);
    inline for (.{ .{ "aarch64", .linux, .gnu, "compile-dsv41-handoff-aarch64-linux" }, .{ "aarch64", .macos, .none, "compile-dsv41-handoff-aarch64-macos" } }) |t| {
        const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = t[1], .abi = t[2] });
        const check = b.addTest(.{ .root_module = root(b, target, .Debug) });
        _ = check.getEmittedBin();
        b.step(t[3], "Emit the hand-off tests for the target without running them").dependOn(&check.step);
    }
}
