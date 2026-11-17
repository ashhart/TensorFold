//! The GPTQ int4 host tests: the packer, the stored codes, the binary64 reference and the epilogue's allowance.
const std = @import("std");

/// `test-int4-host` over the cuda module's `int4:` tests and the contract's derived allowance, and `test-cuda-host`.
pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const step = b.step("test-int4-host", "The GPTQ int4 packer, its stored codes and the binary64 reference (cuda/int4.zig)");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"int4:"} }));
    step.dependOn(&run.step);
    const allow = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/int4_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) }));
    step.dependOn(&allow.step);
    if (b.top_level_steps.get("test-cuda-host")) |all| {
        all.step.dependOn(&run.step);
        all.step.dependOn(&allow.step);
    }
}
