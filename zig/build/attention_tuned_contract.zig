//! Focused replacement-contract executable; the caller supplies the same embedded CUDA images as the server.
const std = @import("std");
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const executable = b.addExecutable(.{ .name = "tf-attention-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/attention_tuned_contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    b.step("attention-tuned-contract-bin", "Build the compact exact-attention replacement gates")
        .dependOn(&b.addInstallArtifact(executable, .{}).step);
}
