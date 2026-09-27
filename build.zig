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
}
