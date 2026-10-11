//! DeepSeek-V4.1's KV memory build: the family's `kv` module over `sessions`, `cuda` and `tp`, host tests on any OS.

const std = @import("std");
const sessions = @import("sessions.zig");
const tp = @import("tp.zig");

/// The CUDA runtime without the engine's kernels: kv needs only the driver, memory and streams.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The `kv` module: imports sessions, cuda and tp (an engine passes its own runtime and tp modules).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, tp_module: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/kv.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("sessions", sessions.module(b, target, optimize));
    m.addImport("cuda", cuda);
    m.addImport("tp", tp_module);
    return m;
}

/// `tf-kv-gpu-test`, the device pool's runner (kv/gpu_main.zig) over `cuda` and `sessions`.
fn gpuRunner(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/gpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("sessions", sessions.module(b, target, optimize));
    m.addImport("cuda", runtime(b, target, optimize));
    return b.addExecutable(.{ .name = "tf-kv-gpu-test", .root_module = m });
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, tp.module(b, host, .debug, cuda)) }));
    const step = b.step("test-kv", "DeepSeek-V4.1 KV's host tests: families, sessions, scheduler, split KV on in-process ranks (no GPU)");
    step.dependOn(&run.step);
    test_step.dependOn(&run.step);
    // the device runner compiles on any host; running it needs a GPU
    const compiled = &gpuRunner(b, host, .debug).step;
    step.dependOn(compiled);
    test_step.dependOn(compiled);
    const bench = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/bench.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    bench.addImport("sessions", sessions.module(b, target, .ReleaseFast));
    const exe = b.addExecutable(.{ .name = "tf-kv-bench", .root_module = bench });
    b.step("tf-kv-bench", "Page ops and a 1M-token session parked to / restored from a real NVMe tier (host pool)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    if (target.result.os.tag != .linux) return;
    const gexe = gpuRunner(b, target, optimize);
    b.step("tf-kv-gpu-test", "The device pool on one GPU: tables, a 1M-token session through device memory and the NVMe tier, copy on write").dependOn(&b.addInstallArtifact(gexe, .{}).step);
}
