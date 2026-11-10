//! Proposed isolated developer target; the root CUDA build opts into registering it.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, qwen35: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tools/qwen_prefill_fidelity.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen35 } } });
    const exe = b.addExecutable(.{ .name = "tf-qwen-prefill-fidelity", .root_module = root });
    b.step("qwen-prefill-fidelity-bin", "Build the fixed-history Qwen prefill fidelity probe")
        .dependOn(&b.addInstallArtifact(exe, .{}).step);
}
