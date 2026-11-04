//! The XPU kernel contract's limits audit: the compiled images' .ze_info and gate 5, with no GPU.
const std = @import("std");
const kernel_contract = @import("kernel_contract.zig");

/// `-Dxpu-contract=true`: tf-xpu-kernel-contract, the objects it reads, and the no-GPU audit.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8) void {
    const core = kernel_contract.module(b, target, optimize);
    const exe = b.addExecutable(.{ .name = "tf-xpu-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) });
    b.step("tf-xpu-kernel-contract", "Build the XPU limits audit (no GPU, no ocloc)").dependOn(&b.addInstallArtifact(exe, .{}).step);

    // ocloc writes a native zebin beside the SPIR-V; the audit reads the zebin's .ze_info
    const files = b.addWriteFiles();
    const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread";
    const glue = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    glue.addFileArg(b.path("zig/kernels/xpu/qwen27/glue.cl"));
    glue.addArgs(&.{ "-device", device, "-options", b.fmt("{s} -I zig/kernels/xpu/qwen27", .{opts}), "-output_no_suffix", "-output", "glue", "-out_dir" });
    glue.setCwd(b.path("."));
    for ([_][]const u8{ "glue_attn.cl", "glue_quant.cl" }) |part| glue.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
    const glue_dir = glue.addOutputDirectoryArg("ocloc-glue");
    _ = files.addCopyFile(glue_dir.path(b, "glue.bin"), b.fmt("{s}/glue.bin", .{device}));
    const ctl = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    ctl.addFileArg(b.path("zig/kernels/xpu/contract_controls.cl"));
    ctl.addArgs(&.{ "-device", device, "-options", opts, "-output_no_suffix", "-output", "contract_controls", "-out_dir" });
    ctl.setCwd(b.path("."));
    const ctl_dir = ctl.addOutputDirectoryArg("ocloc-controls");
    _ = files.addCopyFile(ctl_dir.path(b, "contract_controls.bin"), b.fmt("{s}/contract_controls.bin", .{device}));
    const install = b.addInstallDirectory(.{
        .source_dir = files.getDirectory(),
        .install_dir = .prefix,
        .install_subdir = "xpu-contract",
    });
    b.step("xpu-kernel-contract-objects", "Compile every registered XPU image and the controls to zebins (needs ocloc, no GPU)").dependOn(&install.step);

    const audit = b.addRunArtifact(exe);
    audit.addArg("limits");
    audit.addDirectoryArg(files.getDirectory());
    b.step("xpu-kernel-contract-limits", "Every registered XPU launch against the BMG card and its compiled images (no GPU)").dependOn(&audit.step);
}
