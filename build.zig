const std = @import("std");
const builtin = @import("builtin");

comptime {
    const minimum = std.SemanticVersion.parse(std.mem.trim(u8, @embedFile(".zig-version"), "\r\n")) catch unreachable;
    if (builtin.zig_version.order(minimum) == .lt)
        @compileError("TensorFold requires Zig 0.17.0-dev.2248+3f6a02acd or newer; run bash scripts/fetch-zig.sh, then .zig-toolchain/zig build");
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Optimization mode: debug, safe, fast, small") orelse .fast;
    const prefix = b.option([]const u8, "mlx-prefix", "MLX and mlx-c install prefix") orelse "build/mlx";
    const bindings = b.addTranslateC(.{
        .root_source_file = b.path(b.fmt("{s}/include/mlx/c/mlx.h", .{prefix})),
        .target = target,
        .optimize = optimize,
    });
    bindings.addIncludePath(b.path(b.fmt("{s}/include", .{prefix})));
    const mod = b.createModule(.{
        .root_source_file = b.path("native/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("mlx_c", bindings.createModule());
    mod.addIncludePath(b.path(b.fmt("{s}/include", .{prefix})));
    mod.addLibraryPath(b.path(b.fmt("{s}/lib", .{prefix})));
    mod.addRPath(b.path(b.fmt("{s}/lib", .{prefix})));
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    const exe = b.addExecutable(.{ .name = "tensorfold", .root_module = mod });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    b.step("run", "Run the native Mac engine").dependOn(&run.step);
    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run native host-side unit tests (GPU parity: run --check-exact)").dependOn(&run_tests.step);
    const metal_tests = b.step("test-metal", "Generate Python oracles and compare native Metal kernels (requires .venv)");
    const tensor_tests = b.option(bool, "metal-tensors", "Include M5 tensor attention fixtures in test-metal") orelse false;
    for ([_][]const u8{ "sampling", "sparse", "attention" }) |kind| {
        if (std.mem.eql(u8, kind, "attention") and !tensor_tests) continue;
        const dir = b.fmt("build/native-checks/{s}", .{kind});
        const fixture = b.addSystemCommand(&.{ ".venv/bin/python", b.fmt("tools/native_{s}_fixtures.py", .{kind}), dir });
        const check = b.addRunArtifact(exe);
        check.addArgs(&.{ b.fmt("check-{s}", .{kind}), dir });
        check.step.dependOn(&fixture.step);
        metal_tests.dependOn(&check.step);
    }
    const model_tests = b.step("test-models", "Real-model row/rollback/cache checks for all three Metal families (large RAM required)");
    const model_root = b.option([]const u8, "model-root", "Downloaded checkpoint directory for test-models") orelse "build/models";
    var previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, index| {
        for (0..if (index < 2) @as(usize, 2) else 1) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-exact" });
            if (backend == 1) check.addArg("--metal-simd");
            // Never load multiple full checkpoints concurrently on unified memory.
            if (previous) |step| check.step.dependOn(step);
            previous = &check.step;
        }
    }
    model_tests.dependOn(previous.?);
    const draft_tests = b.step("test-drafts", "Serial/draft regression matrix for all models, samplers and SIMD paths");
    const draft_family = b.option(usize, "draft-family", "Restrict draft regression to 0=Qwen, 1=Nemotron, 2=Flash");
    const directory = b.addSystemCommand(&.{ "mkdir", "-p", "build/native-checks/drafts" });
    var prior: *std.Build.Step = &directory.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (draft_family != null and draft_family.? != family) continue;
        for (0..if (family < 2) @as(usize, 2) else 1) |backend| {
            for (0..3) |scenario| {
                const count: []const u8 = if (scenario == 0) "17" else if (scenario == 1) "32" else "2";
                const common = &.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), if (family == 1) "--prompt" else "--tokens", if (family == 1) "Write a short Python function that computes the Fibonacci sequence." else "1,2,3,4,5,6,7,8,41,42,43,1,2,3,4,5,6,7,8", "--max-tokens", count, "--seed", "5678", "--temperature", if (scenario == 0) "0" else "0.7", "--top-k", "12", "--top-p", "0.8" };
                const serial_report = b.fmt("build/native-checks/drafts/{d}-{d}-{d}-serial.json", .{ family, backend, scenario });
                const serial = b.addRunArtifact(exe);
                serial.addArgs(common);
                if (family != 0) serial.addArg("--no-drafts");
                if (backend == 1) serial.addArg("--metal-simd");
                if (scenario == 1) serial.addArg("--metal-sampling");
                serial.addArgs(&.{ "--report", serial_report });
                serial.step.dependOn(prior);
                prior = &serial.step;
                for ([_][]const u8{ "1", "3", "15" }, 0..) |budget, index| {
                    if (family == 0 and index != 0) continue;
                    const report = b.fmt("build/native-checks/drafts/{d}-{d}-{d}-{s}.json", .{ family, backend, scenario, budget });
                    const draft_run = b.addRunArtifact(exe);
                    draft_run.addArgs(common);
                    if (family == 0) draft_run.addArgs(&.{ "--drafter", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}) }) else draft_run.addArgs(&.{ "--mtp-drafts", budget });
                    if (backend == 1) draft_run.addArg("--metal-simd");
                    if (scenario == 1) draft_run.addArg("--metal-sampling");
                    draft_run.addArgs(&.{ "--report", report });
                    draft_run.step.dependOn(prior);
                    const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--require-rounds", "--compare-reports", serial_report, report });
                    compare.step.dependOn(&draft_run.step);
                    prior = &compare.step;
                }
            }
        }
    }
    draft_tests.dependOn(prior);
    const cache_tests = b.step("test-cache-stress", "All accepted prefixes, randomized trees, EOS history, rejection and reset cycles");
    const cache_family = b.option(usize, "cache-family", "Restrict cache stress to 0=Qwen, 1=Nemotron, 2=Flash");
    var cache_previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (cache_family != null and cache_family.? != family) continue;
        for (0..if (family < 2) @as(usize, 2) else 1) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-cache-stress" });
            if (backend == 1) check.addArg("--metal-simd");
            if (cache_previous) |step| check.step.dependOn(step);
            cache_previous = &check.step;
        }
    }
    if (cache_previous) |step| cache_tests.dependOn(step);
}
