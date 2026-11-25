//! Build script for TensorFold `mlx_tf`:
//! Builds the standalone `mlx_tf` CLI/server binary and `libmlxtf.dylib` C ABI shared library.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const shaders_mod = b.createModule(.{
        .root_source_file = b.path("kernels/embed.zig"),
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "shaders", .module = shaders_mod },
        },
    });
    exe_mod.linkSystemLibrary("objc", .{});
    exe_mod.linkFramework("Metal", .{});
    exe_mod.linkFramework("Foundation", .{});
    exe_mod.linkFramework("ImageIO", .{});
    exe_mod.linkFramework("CoreGraphics", .{});
    exe_mod.linkFramework("CoreFoundation", .{});

    const exe = b.addExecutable(.{
        .name = "mlx_tf",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "shaders", .module = shaders_mod },
        },
    });
    lib_mod.linkSystemLibrary("objc", .{});
    lib_mod.linkFramework("Metal", .{});
    lib_mod.linkFramework("Foundation", .{});
    lib_mod.linkFramework("ImageIO", .{});
    lib_mod.linkFramework("CoreGraphics", .{});
    lib_mod.linkFramework("CoreFoundation", .{});

    const lib = b.addLibrary(.{
        .name = "mlxtf",
        .linkage = .dynamic,
        .root_module = lib_mod,
    });
    b.installArtifact(lib);
}
