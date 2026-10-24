//! Tensor parallelism's build: Linux host tests, and on Linux the `tf-tp-test` GPU runner with the mailbox kernel.

const std = @import("std");

/// The CUDA runtime without the engine's kernels (tp only needs the driver, memory, streams and NCCL).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The `tp` module over `cuda` (an engine passes its own runtime module); verbs (zig/src/fabric/verbs.zig) for RoCE.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/tp.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("verbs", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/verbs.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return m;
}

/// mailbox.cu as a fatbin: sm_120 (RTX PRO 6000), sm_121 (GB10), and compute_90 PTX for anything else.
fn mailboxFatbin(b: *std.Build, nvcc: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin", "-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120", "-gencode=arch=compute_121,code=sm_121", "-gencode=arch=compute_90,code=compute_90", "-o" });
    const out = run.addOutputFileArg("tp_mailbox.fatbin");
    run.addFileArg(b.path("zig/src/fabric/tp/mailbox.cu"));
    _ = run.addFileInput(b.path("zig/src/fabric/tp/roce_device.cuh"));
    return out;
}

/// `pub const mailbox: []const u8`, the fatbin's bytes or empty.
fn kernelModule(b: *std.Build, image: ?std.Build.LazyPath) *std.Build.Module {
    const wf = b.addWriteFiles();
    if (image) |file| {
        _ = wf.addCopyFile(file, "tp_mailbox.fatbin");
        // the driver takes 8-byte aligned images (Module.load refuses others); @embedFile alone promises nothing
        return b.createModule(.{ .root_source_file = wf.add("tp_kernel.zig", "const bytes align(16) = @embedFile(\"tp_mailbox.fatbin\").*;\npub const mailbox: []const u8 = &bytes;\n") });
    }
    return b.createModule(.{ .root_source_file = wf.add("tp_kernel.zig", "pub const mailbox: []const u8 = \"\";\n") });
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step, opts: @import("options.zig").Tp) void {
    const host = b.graph.host;
    if (host.result.os.tag != .windows) {
        const fate_host = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/fate_lifetime_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = runtime(b, host, .debug) }} });
        b.step("test-tp-fate-host", "Failure hooks and their owners survive concurrent shutdown without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fate_host })).step);
    }
    const cli_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/gpu_cli.zig"), .target = host, .optimize = optimize }) }));
    const cli_step = b.step("test-tp-cli", "TP runner command validation (no GPU)");
    cli_step.dependOn(&cli_tests.step);
    test_step.dependOn(&cli_tests.step);
    const portable = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/portable_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = runtime(b, host, .debug) }} });
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = if (host.result.os.tag == .linux)
        module(b, host, .debug, runtime(b, host, .debug))
    else
        portable }));
    const tp_step = b.step("test-tp", "TP bootstrap, failure, settings and in-process transport tests");
    tp_step.dependOn(&tests.step);
    tp_step.dependOn(cli_step);
    const multi_root = module(b, host, .debug, runtime(b, host, .debug));
    multi_root.root_source_file = b.path("zig/src/fabric/tp/roce_multi_host.zig");
    const multi = b.addRunArtifact(b.addTest(.{ .root_module = multi_root }));
    multi.has_side_effects = true;
    b.step("test-roce-multi-host", "Frozen two-rank posts and multirank fake-verbs byte and completion contracts").dependOn(&multi.step);
    tp_step.dependOn(&multi.step);
    if (host.result.os.tag != .windows) test_step.dependOn(&multi.step);
    // the bootstrap and failure tests run over POSIX sockets: Windows checks only the Linux compile below
    if (host.result.os.tag != .windows) test_step.dependOn(&tests.step);
    const linux = b.resolveTargetQuery(.{ .cpu_arch = host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
    const check_cuda = runtime(b, linux, .debug);
    const object = b.addObject(.{ .name = "tp_host_compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/tp_compile.zig"),
        .target = linux,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = check_cuda }, .{ .name = "tp", .module = module(b, linux, .debug, check_cuda) } },
    }) });
    tests.step.dependOn(&object.step);
    if (host.result.os.tag == .windows) test_step.dependOn(&object.step);
    if (target.result.os.tag != .linux) return;
    const nvcc = opts.nvcc;
    const prebuilt = opts.fatbin;
    const image: ?std.Build.LazyPath = if (prebuilt) |p| b.graph.cwdRelativePath(p) else if (nvcc) |tool| mailboxFatbin(b, tool) else null;
    const cuda = runtime(b, target, optimize);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/gpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("tp", module(b, target, optimize, cuda));
    runner.addImport("tp_kernel", kernelModule(b, image));
    const exe = b.addExecutable(.{ .name = "tf-tp-test", .root_module = runner });
    if (host.result.os.tag == .linux and target.query.isNative()) {
        const cli = @import("../src/fabric/tp/gpu_cli.zig");
        for (cli.invalid) |args| {
            const check = b.addRunArtifact(exe);
            check.addArgs(args);
            check.setEnvironmentVariable("TF_TP_WORLD", "invalid");
            check.expectExitCode(2);
            check.expectStdErrEqual(cli.usage);
            cli_step.dependOn(&check.step);
            test_step.dependOn(&check.step);
        }
        if (image == null) {
            const kernel = b.addRunArtifact(exe);
            kernel.addArg("kernel");
            kernel.setEnvironmentVariable("TF_TP_WORLD", "invalid");
            kernel.expectExitCode(1);
            kernel.expectStdErrEqual("mailbox kernel 0 B\n");
            cli_step.dependOn(&kernel.step);
            test_step.dependOn(&kernel.step);
        }
    }
    b.installArtifact(exe);
    b.step("tf-tp-test", "TP on 2 to 16 GPUs: bootstrap, every collective bit-exact, graph capture, fail-fast, latency").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
