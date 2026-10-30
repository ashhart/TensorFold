//! The native Vulkan model build, opt-in -Dvulkan=true: the tf-qwen35-model probe, the native server, GPU-free tests.

const std = @import("std");
const vulkan_build = @import("vulkan.zig");
const cuda_build = @import("cuda.zig");
const server_build = @import("server.zig");

/// vulkan-model-probe-build, compile-native-vulkan-server, native-vulkan, qwen35-model-host-test.
pub fn steps(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Optimize,
    test_step: *std.Build.Step,
    o: vulkan_build.Options,
    build_options: *std.Build.Step.Options,
) void {
    const m = vulkan_build.serverModules(b, target, optimize, o, lanesModule(b, target, optimize));

    // (a) the real qwen3.5 model probe over the named family and shader modules; compile only.
    const probe = b.addExecutable(.{ .name = "tf-qwen35-model", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/qwen35_model_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "qwen35_vulkan", .module = m.qwen35_vulkan },
            .{ .name = "qwen35_shaders", .module = m.qwen35_shaders },
            .{ .name = "core", .module = m.core },
        },
    }) });
    b.step("vulkan-model-probe-build", "Build zig-out/bin/tf-qwen35-model without running GPU work").dependOn(&b.addInstallArtifact(probe, .{}).step);

    // (b) Server entry point, checkpoint CLI and factory engine_api over Vulkan engines.
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const server = b.addExecutable(.{ .name = "tensorfold-native-vulkan", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = m.core },
            .{ .name = "engine_api", .module = m.engine_api },
            .{ .name = "living", .module = m.core.import_table.get("living").? },
            .{ .name = "tokenizer", .module = m.core.import_table.get("tokenizer").? },
            .{ .name = "template", .module = template },
            .{ .name = "native_engines", .module = m.native_engines },
            .{ .name = "checkpoint_cli", .module = cuda_build.checkpointCli(b, target, .ReleaseSafe, null, m.native_engines, m.core) },
        },
    }) });
    server.root_module.addOptions("build_options", build_options);
    _ = server_build.addTo(b, server.root_module, target, .ReleaseSafe);
    b.step("compile-native-vulkan-server", "Compile the actual server entry point over the Vulkan engines without installing or running").dependOn(&server.step);
    b.step("native-vulkan", "tensorfold-native with the Vulkan engines into zig-out/native/bin").dependOn(&b.addInstallArtifact(server, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } }).step);

    // (c) GPU-free host admission: the adapter's tests, the family backend/engine/forward decls, the probe's parse.
    const host = b.graph.host;
    const h = vulkan_build.serverModules(b, host, .debug, o, lanesModule(b, host, .debug));
    const all = b.step("qwen35-model-host-test", "Qwen3.5 Vulkan engine/forward/native-backend and probe argument parser host tests, without GPU work");
    const native_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/native/vulkan.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = h.engine_api },
            .{ .name = "lanes", .module = h.engine_api.import_table.get("lanes").? },
            .{ .name = "qwen35_vulkan", .module = h.qwen35_vulkan },
            .{ .name = "qwen35_shaders", .module = h.qwen35_shaders },
        },
    }) });
    all.dependOn(&b.addRunArtifact(native_tests).step);
    // A file-local root discovers tests; a separate imported module does not.
    const family_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/vulkan_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = h.qwen35_vulkan.import_table.get("vulkan").? },
            .{ .name = "lanes", .module = h.engine_api.import_table.get("lanes").? },
            .{ .name = "core", .module = h.core },
        },
    }) });
    all.dependOn(&b.addRunArtifact(family_tests).step);
    const probe_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/qwen35_model_probe.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "qwen35_vulkan", .module = h.qwen35_vulkan },
            .{ .name = "qwen35_shaders", .module = h.qwen35_shaders },
            .{ .name = "core", .module = h.core },
        },
    }) });
    // TF_VULKAN_LOAD_DIR is not a cached input: the engine load reruns when the env changes, like TF_QWEN_DIR's.
    const probe_run = b.addRunArtifact(probe_tests);
    probe_run.has_side_effects = true;
    all.dependOn(&probe_run.step);
    test_step.dependOn(all);
}

/// The lane core the factory takes, built as build.zig builds it for the other seams.
fn lanesModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
}
