//! DeepSeek-V4.1's build: the family's host side (zig/src/families/deepseek_v41) and its host tests on any OS.

const std = @import("std");

/// The backend-neutral core (the safetensors reader the pack headers use) for the family's host side.
fn core(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("tokenizer", tokenizer);
    return m;
}

/// The CUDA runtime without kernel images: the family's launch bindings and device loads compile on any host.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The family's kernel bindings and loader (`dsv41_kernels`, kernels.zig) over `cuda`, without fatbins (host-only).
pub fn kernels(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    options.addOption(usize, "dsv41_images", 0); // also keeps this options file apart from the runtime's
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kernels.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addOptions("dsv41_kernel_options", options);
    // the optional images (keyed sampler, two-tile linears) are embedded as empty files
    const empty = b.addWriteFiles().add("empty.fatbin", "");
    for ([_][]const u8{ "sampling", "lin2" }) |n| m.addAnonymousImport(b.fmt("dsv41_fatbin_dsv41_{s}", .{n}), .{ .root_source_file = empty });
    return m;
}

/// The family's host side (config, pack headers, a rank's weight plan, kernel bindings) as the `deepseek_v41` module.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/deepseek_v41.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const cuda = runtime(b, target, optimize);
    m.addImport("core", core(b, target, optimize));
    m.addImport("cuda", cuda);
    m.addImport("dsv41_kernels", kernels(b, target, optimize, cuda));
    return m;
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const step = b.step("test-dsv41", "DeepSeek-V4.1's host-side unit tests (no GPU work)");
    const family = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug) }));
    const bindings = b.addRunArtifact(b.addTest(.{ .root_module = kernels(b, host, .debug, runtime(b, host, .debug)) }));
    for ([_]*std.Build.Step.Run{ family, bindings }) |run| {
        step.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}
