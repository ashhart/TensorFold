//! The kernel contract on HIP: tf-hip-kernel-contract, its controls' code object, and every served chip's objects.
const std = @import("std");
const hip = @import("hip.zig");
const limits = @import("../src/core/contract/limits.zig");

pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    hipcc: hip.Hipcc,
    arch: []const u8,
    /// The HIP backend over -Dhip-arch's kernels, and over stub kernels for the host build.
    gpu: *std.Build.Module,
    host: *std.Build.Module,
    test_step: *std.Build.Step,
    lighton_gpu: *std.Build.Module,
    lighton_host: *std.Build.Module,
};

/// hip-kernel-contract (gates 1-5 on this GPU), hip-kernel-contract-objects and hip-kernel-contract-limits (no GPU).
pub fn steps(b: *std.Build, o: Options) void {
    const pair_host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/contract_relu2_pair.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = o.host }, .{ .name = "core", .module = o.host.import_table.get("core").? } },
    }), .filters = &.{"HIP pair contract:"} }));
    pair_host.has_side_effects = true;
    o.test_step.dependOn(&pair_host.step);
    b.step("test-hip-pair-contract-host", "Paired projection simulation without uninitialized magnitude reads or GPU calls").dependOn(&pair_host.step);
    const select_host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/contract_select.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = o.host }, .{ .name = "core", .module = o.host.import_table.get("core").? } },
    }) }));
    o.test_step.dependOn(&select_host.step);
    const select_what = "Which entry and records `image <group>` runs, and its exit status, on fixtures without a GPU";
    b.step("test-hip-contract-select-host", select_what).dependOn(&select_host.step);
    const lighton_ref = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/contract_lighton_ref_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "core", .module = o.host.import_table.get("core").? }},
    }) }));
    b.step("test-lighton-hip-contract-host", "Independent LightOn HIP contract references without a device").dependOn(&lighton_ref.step);
    o.test_step.dependOn(&lighton_ref.step);
    const source = b.path("zig/kernels/hip/contract_controls.hip");
    const own = hip.compile(b, o.hipcc, o.arch, source, "contract_controls.hsaco");
    const exe = runner(b, o.target, o.optimize, o.gpu, o.lighton_gpu, controls(b, o.arch, own));
    b.step("hip-kernel-contract-build", "Compile and install the HIP kernel contract without running a GPU").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const install = b.addInstallArtifact(exe, .{});
    const build_only = "Build the kernel contract's HIP gates and controls (-Dhip-arch, no run)";
    b.step("tf-hip-kernel-contract", build_only).dependOn(&install.step);
    const run = b.addRunArtifact(exe);
    const gates = "Every registered HIP kernel through the contract's gates on this GPU, each control red";
    b.step("hip-kernel-contract", gates).dependOn(&run.step);
    const all = b.addWriteFiles();
    for (limits.rdna) |t| {
        _ = all.addCopyDirectory(hip.objectFiles(b, o.hipcc, t.name).getDirectory(), t.name, .{});
        const ctl = hip.compile(b, o.hipcc, t.name, source, "contract_controls.hsaco");
        _ = all.addCopyFile(ctl, b.fmt("{s}/contract_controls.hsaco", .{t.name}));
    }
    const to: std.Build.Step.InstallDir.Options = .{
        .source_dir = all.getDirectory(),
        .install_dir = .prefix,
        .install_subdir = "hsaco",
    };
    const dir = b.addInstallDirectory(to);
    const each = "Every kernel group and the controls for each RDNA chip we serve (hsaco/)";
    b.step("hip-kernel-contract-objects", each).dependOn(&dir.step);
    const audit = b.addRunArtifact(runner(b, b.graph.host, .Debug, o.host, o.lighton_host, controls(b, "", null)));
    audit.addArg("limits");
    audit.addDirectoryArg(all.getDirectory());
    const fits = "Every registered HIP launch against each served chip's code objects";
    b.step("hip-kernel-contract-limits", fits).dependOn(&audit.step);
    // the runner compiles on any host against stub kernels, so `test` catches what only it reaches
    const host_runner = runner(b, b.graph.host, .Debug, o.host, o.lighton_host, controls(b, "", null));
    o.test_step.dependOn(&host_runner.step);
    b.step("check-hip-kernel-contract-host", "Compile the complete HIP contract registry with inert kernels").dependOn(&host_runner.step);
}

/// The controls' code object as the `hip_controls` module (empty without hipcc).
fn controls(b: *std.Build, arch: []const u8, object: ?std.Build.LazyPath) *std.Build.Module {
    const files = b.addWriteFiles();
    if (object) |obj| _ = files.addCopyFile(obj, "controls.hsaco") else _ = files.add("controls.hsaco", "");
    const bytes = "pub const bytes align(8) = @embedFile(\"controls.hsaco\").*;\n";
    const src = b.fmt("pub const arch = \"{s}\";\n{s}", .{ arch, bytes });
    return b.createModule(.{ .root_source_file = files.add("controls.zig", src) });
}

fn runner(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    hip_mod: *std.Build.Module,
    lighton_mod: *std.Build.Module,
    ctl: *std.Build.Module,
) *std.Build.Step.Compile {
    const shaders = b.createModule(.{
        .root_source_file = b.addWriteFiles().add("glm_ocr_shaders.zig", "const hip = @import(\"hip\");\npub const arch = hip.kernels.arch;\npub const bytes = hip.kernels.images[@backingInt(hip.kernels.Group.glm_ocr)];\n"),
        .imports = &.{.{ .name = "hip", .module = hip_mod }},
    });
    const ocr_ops = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_ocr/hip_ops.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "hip", .module = hip_mod }, .{ .name = "glm_ocr_kernels", .module = shaders } },
    });
    return b.addExecutable(.{ .name = "tf-hip-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip_mod },
            .{ .name = "core", .module = hip_mod.import_table.get("core").? },
            .{ .name = "hip_controls", .module = ctl },
            .{ .name = "glm_ocr_ops", .module = ocr_ops },
            .{ .name = "lighton_ocr", .module = lighton_mod },
            .{ .name = "qwen_hip_vision", .module = @import("lighton_hip.zig").vision(b, target, optimize, hip_mod, hip_mod.import_table.get("core").?, lighton_mod.import_table.get("lanes").?, lighton_mod.import_table.get("hip_serve").?) },
        },
    }) });
}
