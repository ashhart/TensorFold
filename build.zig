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
    const file_tests = b.addRunArtifact(exe);
    file_tests.addArgs(&.{ "check-checkpoint-files", "build/native-checks/files" });
    b.step("test-checkpoint-files", "Exercise positional reads, corrupt checkpoints and allocation failures without a GPU").dependOn(&file_tests.step);
    const metal_tests = b.step("test-metal", "Generate Python oracles and compare native Metal kernels (requires .venv)");
    const tensor_tests = b.option(bool, "metal-tensors", "Include M5 tensor attention fixtures in test-metal") orelse false;
    const variants_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/variants" });
    const variants = b.addRunArtifact(exe);
    variants.addArgs(&.{ "check-variants", "build/native-checks/variants" });
    variants.step.dependOn(&variants_fixture.step);
    b.step("test-variants", "Run original optional-kernel tests and compare their launches through native embedded Metal").dependOn(&variants.step);
    for ([_][]const u8{ "sampling", "sparse", "attention", "ple_norm" }) |kind| {
        if (std.mem.eql(u8, kind, "attention") and !tensor_tests) continue;
        const dir = b.fmt("build/native-checks/{s}", .{kind});
        const fixture = b.addSystemCommand(&.{ ".venv/bin/python", b.fmt("tools/native_{s}_fixtures.py", .{kind}), dir });
        const check = b.addRunArtifact(exe);
        check.addArgs(&.{ b.fmt("check-{s}", .{kind}), dir });
        check.step.dependOn(&fixture.step);
        metal_tests.dependOn(&check.step);
        if (std.mem.eql(u8, kind, "ple_norm")) b.step("test-ple-norm", "Compare PLE normalization against original Python arithmetic").dependOn(&check.step);
    }
    const model_tests = b.step("test-models", "Real-model row/rollback/cache checks for all three Metal families (large RAM required)");
    const model_root = b.option([]const u8, "model-root", "Downloaded checkpoint directory for test-models") orelse "build/models";
    const schema_tests = b.step("test-model-schemas", "Validate all downloaded tensor names/shapes/dtypes and index references without loading payloads");
    const schema_failures = b.addSystemCommand(&.{ ".venv/bin/python", "tools/test_native_schemas.py" });
    schema_failures.addArtifactArg(exe);
    schema_failures.addArgs(&.{ "--model-root", model_root });
    b.step("test-schema-failures", "Reject malformed checkpoint metadata and missing MTP through the native CLI without a GPU").dependOn(&schema_failures.step);
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "Qwen3.8-27B-DFlash2", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, [_][]const u8{ "qwen", "dflash", "nemotron", "flash" }) |name, kind| {
        const check = b.addRunArtifact(exe);
        check.addArgs(&.{ "check-model-schema", kind, b.fmt("{s}/{s}", .{ model_root, name }) });
        schema_tests.dependOn(&check.step);
    }
    const ple_tests = b.addRunArtifact(exe);
    ple_tests.addArgs(&.{ "check-ple", b.fmt("{s}/Qwen3.8-Flash-Next-MLX-4bit-MTP", .{model_root}) });
    b.step("test-ple", "Compare native positional PLE reads against MLX at all 128 shard boundaries").dependOn(&ple_tests.step);
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
    const draft_scenario = b.option(usize, "draft-scenario", "Restrict draft regression to 0=greedy, 1=Metal, 2=CPU, 3=two-token CPU");
    if (draft_family != null and draft_family.? > 2) @panic("draft-family must be 0, 1 or 2");
    if (draft_scenario != null and draft_scenario.? > 3) @panic("draft-scenario must be 0, 1, 2 or 3");
    const directory = b.addSystemCommand(&.{ "mkdir", "-p", "build/native-checks/drafts" });
    var prior: *std.Build.Step = &directory.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (draft_family != null and draft_family.? != family) continue;
        for (0..if (family < 2) @as(usize, 2) else 1) |backend| {
            for (0..4) |scenario| {
                if (draft_scenario != null and draft_scenario.? != scenario) continue;
                const count: []const u8 = if (scenario == 0) "17" else if (scenario == 3) "2" else "32";
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
    const long_cache_tests = b.step("test-long-cache", "Full-model cache rollback with random prompts across sparse and 10K thresholds");
    const cache_family = b.option(usize, "cache-family", "Restrict cache stress to 0=Qwen, 1=Nemotron, 2=Flash");
    if (cache_family != null and cache_family.? > 2) @panic("cache-family must be 0, 1 or 2");
    var cache_previous: ?*std.Build.Step = null;
    var long_cache_previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (cache_family != null and cache_family.? != family) continue;
        for (0..if (family < 2) @as(usize, 2) else 1) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-cache-stress" });
            if (backend == 1) check.addArg("--metal-simd");
            if (cache_previous) |step| check.step.dependOn(step);
            cache_previous = &check.step;
            const long_check = b.addRunArtifact(exe);
            long_check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-long-cache" });
            if (backend == 1) long_check.addArg("--metal-simd");
            if (long_cache_previous) |step| long_check.step.dependOn(step);
            long_cache_previous = &long_check.step;
        }
    }
    if (cache_previous) |step| cache_tests.dependOn(step);
    if (long_cache_previous) |step| long_cache_tests.dependOn(step);

    const long_tests = b.step("test-long-context", "Full-model Python/native logits and drafted continuations across attention thresholds");
    const long_family = b.option(usize, "long-family", "Restrict long-context checks to 0=Qwen, 1=Nemotron, 2=Flash");
    const long_size = b.option(usize, "long-tokens", "Override context length for one threshold or diagnostic case");
    if (long_family != null and long_family.? > 2) @panic("long-family must be 0, 1 or 2");
    if (long_size != null and (long_size.? == 0 or long_size.? > 262128)) @panic("long-tokens must be 1..262128");
    const long_dir = b.addSystemCommand(&.{ "mkdir", "-p", "build/native-checks/long" });
    var long_prior: *std.Build.Step = &long_dir.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (long_family != null and long_family.? != family) continue;
        const lengths = if (family == 2) [_]usize{ 2051, 2063 } else [_]usize{ 9999, 10007 };
        for (lengths, 0..) |default_length, case| {
            if (long_size != null and case > 0) continue;
            const length = long_size orelse default_length;
            const ids = b.allocator.alloc([]const u8, length) catch @panic("OOM");
            // Four repeating tokens keep the Python PLE oracle's resident shard set
            // bounded while RoPE/cache lengths still exercise every threshold.
            for (ids, 0..) |*id, j| id.* = b.fmt("{d}", .{1000 + (j % 4) * 37});
            const prompt = std.mem.join(b.allocator, ",", ids) catch @panic("OOM");
            for (0..if (family == 1) @as(usize, 2) else 1) |backend| {
                const base = b.fmt("build/native-checks/long/{d}-{d}-{d}", .{ family, backend, length });
                const dir = b.fmt("{s}/{s}", .{ model_root, name });
                const common = &.{ "--tokens", prompt, "--seed", "5678", "--temperature", "0.7", "--top-k", "12", "--top-p", "0.8", "--metal-sampling" };
                const oracle = b.addSystemCommand(&.{ ".venv/bin/python", if (family == 0) "tools/native_reference.py" else "tools/native_families_reference.py" });
                if (family == 0) oracle.addArg("--model");
                oracle.addArg(dir);
                oracle.addArgs(common);
                oracle.addArgs(&.{ "--generate", "16", "--dump-logits", b.fmt("{s}-python.npy", .{base}), "--output", b.fmt("{s}-python.json", .{base}) });
                if (backend == 1) oracle.addArg("--simd");
                oracle.step.dependOn(long_prior);
                long_prior = &oracle.step;
                for (0..2) |drafts| {
                    const suffix = if (drafts == 0) "serial" else "draft";
                    const check = b.addRunArtifact(exe);
                    check.addArgs(&.{ "run", dir });
                    check.addArgs(common);
                    check.addArgs(&.{ "--max-tokens", "16", "--dump-logits", b.fmt("{s}-{s}.npy", .{ base, suffix }), "--report", b.fmt("{s}-{s}.json", .{ base, suffix }) });
                    if (backend == 1) check.addArg("--metal-simd");
                    if (drafts == 1) check.addArg("--no-copy");
                    if (family == 0 and drafts == 1) check.addArgs(&.{ "--drafter", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}) });
                    if (family > 0) check.addArgs(if (drafts == 0) &.{"--no-drafts"} else &.{ "--mtp-drafts", "15" });
                    check.step.dependOn(long_prior);
                    const logits = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare", b.fmt("{s}-python.npy", .{base}), b.fmt("{s}-{s}.npy", .{ base, suffix }) });
                    logits.step.dependOn(&check.step);
                    const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-reports", b.fmt("{s}-python.json", .{base}), b.fmt("{s}-{s}.json", .{ base, suffix }) });
                    compare.step.dependOn(&logits.step);
                    long_prior = &compare.step;
                }
            }
        }
    }
    long_tests.dependOn(long_prior);
}
