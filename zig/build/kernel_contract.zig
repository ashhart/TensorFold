//! The kernel contract's programs: tf-kernel-contract on Metal, and `zig build kernel-contract` that runs the gates.
const std = @import("std");

/// The contract module alone (its host tests run in `zig build test` through core/root.zig).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/contract.zig"), .target = target, .optimize = optimize });
}

/// `test-kernel-contract` on every host: the gates' own tests and every registered launch against every chip's limits.
pub fn hostTests(b: *std.Build) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug) }));
    b.step("test-kernel-contract", "The kernel contract's host tests: gate logic, controls, every launch against every chip").dependOn(&run.step);
}

/// macOS: tf-kernel-contract over the engine's Metal kernels; `kernel-contract` runs it on this Mac's GPU.
pub fn metal(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal_mod: *std.Build.Module, engine: *std.Build.Module, sources: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "contract", .module = module(b, target, optimize) },
            .{ .name = "h3_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") }) },
        },
    }) });
    const install = b.addInstallArtifact(exe, .{});
    b.step("tf-kernel-contract", "Build the kernel contract's Metal gates and controls (no run)").dependOn(&install.step);
    const run = b.addRunArtifact(exe);
    b.step("kernel-contract", "Every registered kernel through the contract's gates on this GPU, each control red").dependOn(&run.step);
}

/// Linux: `kernel-contract` runs tf-cuda-test's kernel-contract command (kernels from -Dnvcc or -Dfatbins).
pub fn cuda(b: *std.Build, runner: *std.Build.Step.Compile, controls: ?std.Build.LazyPath) void {
    const opts = b.addOptions();
    opts.addOption(bool, "controls", controls != null);
    runner.root_module.addOptions("contract_options", opts);
    const empty = b.addWriteFiles().add("contract_controls.fatbin", "");
    runner.root_module.addAnonymousImport("fatbin_contract_controls", .{ .root_source_file = controls orelse empty });
    if (controls) |file| b.step("kernel-contract-fatbins", "Install the contract's control fatbin beside the others (fatbin/)").dependOn(&b.addInstallFile(file, "fatbin/contract_controls.fatbin").step);
    const run = b.addRunArtifact(runner);
    run.addArg("kernel-contract");
    b.step("kernel-contract", "Every registered CUDA kernel through the contract's gates on this GPU, each control red").dependOn(&run.step);
}
