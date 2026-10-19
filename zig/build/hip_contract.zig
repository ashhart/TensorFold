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
};

/// hip-kernel-contract (gates 1-5 on this GPU), hip-kernel-contract-objects and hip-kernel-contract-limits (no GPU).
pub fn steps(b: *std.Build, o: Options) void {
    const source = b.path("zig/kernels/hip/contract_controls.hip");
    const own = hip.compile(b, o.hipcc, o.arch, source, "contract_controls.hsaco");
    const exe = runner(b, o.target, o.optimize, o.gpu, controls(b, o.arch, own));
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
    const audit = b.addRunArtifact(runner(b, b.graph.host, .Debug, o.host, controls(b, "", null)));
    audit.addArg("limits");
    audit.addDirectoryArg(all.getDirectory());
    const fits = "Every registered HIP launch against each served chip's code objects";
    b.step("hip-kernel-contract-limits", fits).dependOn(&audit.step);
    // the runner compiles on any host against stub kernels, so `test` catches what only it reaches
    o.test_step.dependOn(&runner(b, b.graph.host, .Debug, o.host, controls(b, "", null)).step);
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
    ctl: *std.Build.Module,
) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-hip-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip_mod },
            .{ .name = "core", .module = hip_mod.import_table.get("core").? },
            .{ .name = "hip_controls", .module = ctl },
        },
    }) });
}
