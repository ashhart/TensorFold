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
}
