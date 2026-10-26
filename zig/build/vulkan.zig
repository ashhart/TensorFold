//! Vulkan, built only with -Dvulkan=true: host tests in `test`, named GPU steps; options on every target (#1165).

const std = @import("std");

/// Vulkan's options as given, declared whether or not -Dvulkan=true builds anything with them.
pub const Options = struct { include: ?[]const u8, glslang: ?[]const u8 };

pub fn options(b: *std.Build) Options {
    return .{
        .include = b.option([]const u8, "vulkan-include", "Directory holding vulkan/vulkan_core.h and vk_video/ (default /usr/include)"),
        .glslang = b.option([]const u8, "glslang", "glslang for the Vulkan shaders (default: glslang on PATH)"),
    };
}

/// vulkan-host-test (part of `test`), vulkan-gpu-build and vulkan-gpu-test, vulkan-probe-build and vulkan-probe.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, test_step: *std.Build.Step, o: Options) void {
    const host = b.step("vulkan-host-test", "Vulkan loader admission, device selection, buffer geometry and the qmm fixture, without GPU work");
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug, o) })).step);
    const fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/affine_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fixture })).step);
    test_step.dependOn(host);
    const vulkan = module(b, target, optimize, o);
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/vulkan/gpu_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "vulkan", .module = vulkan }},
    }) });
    b.step("vulkan-gpu-build", "Compile the Vulkan device tests without running GPU work").dependOn(&gpu.step);
    b.step("vulkan-gpu-test", "Real Vulkan device selection and staged device-local buffer round trips").dependOn(&b.addRunArtifact(gpu).step);
    const glslang = o.glslang orelse "glslang";
    // vulkan1.2 emits SPIR-V 1.5: 1.6's LocalSizeId needs maintenance4, which Device does not enable
    const spirv = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const qmm = spirv.addOutputFileArg("qmm.spv");
    spirv.addFileArg(b.path("zig/kernels/vulkan/qmm.comp"));
    const files = b.addWriteFiles();
    _ = files.addCopyFile(qmm, "qmm.spv");
    const shaders = b.createModule(.{ .root_source_file = files.add("shaders.zig", "pub const qmm align(4) = @embedFile(\"qmm.spv\").*;\n") });
    const probe = b.addExecutable(.{ .name = "tf-vulkan-probe", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "vulkan_shaders", .module = shaders },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
        },
    }) });
    const install = b.addInstallArtifact(probe, .{});
    b.step("vulkan-probe-build", "Build zig-out/bin/tf-vulkan-probe and its SPIR-V without running GPU work").dependOn(&install.step);
    const run = b.addRunArtifact(probe);
    run.addPassthruArgs();
    b.step("vulkan-probe", "Run the Vulkan qmm probe on the chosen GPU (args after --)").dependOn(&run.step);
}

/// The backend over `vk`, vulkan_core.h translated without prototypes: every entry point comes from the loader.
fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, o: Options) *std.Build.Module {
    const header = b.addWriteFiles().add("vulkan_c.h", "#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n");
    const vk = b.addTranslateC(.{ .root_source_file = header, .target = target, .optimize = optimize });
    // after the target's libc headers, so a cross build never takes the host's stdint.h
    vk.addAfterIncludePath(.{ .cwd_relative = o.include orelse "/usr/include" });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("vk", vk.createModule());
    return m;
}
