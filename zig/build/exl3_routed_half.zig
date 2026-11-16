//! Source-only operator contracts use the same registrations as tf-cuda-test and load their images at runtime.
const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/exl3_routed_half_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    const options = b.addOptions();
    options.addOption(bool, "controls", false);
    m.addOptions("contract_options", options);
    const exe = b.addExecutable(.{ .name = "tf-exl3-routed-half-contract", .root_module = m });
    b.step("tf-exl3-routed-half-contract", "Build the complete routed-half operator contract runner").dependOn(&b.addInstallArtifact(exe, .{}).step);
    b.step("check-exl3-routed-half-contract", "Compile every routed-half contract callback without a GPU").dependOn(&exe.step);
    const reference = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/exl3_routed_half_reference.zig"), .target = b.graph.host });
    b.step("test-exl3-routed-half-host", "Independent Walsh reference and the half-product rounding witness").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = reference })).step);
}
