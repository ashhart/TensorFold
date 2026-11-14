//! The XPU kernel contract's limits audit: the compiled images' .ze_info and gate 5, with no GPU.
const std = @import("std");
const kernel_contract = @import("kernel_contract.zig");

/// `-Dxpu-contract=true`: tf-xpu-kernel-contract, the objects it reads, and the no-GPU audit.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8, prebuilt: ?[]const u8) void {
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
    for (sources) |src| {
        _ = files.addCopyFile(object(b, ocloc, device, src), b.fmt("{s}/{s}.bin", .{ device, src.name }));
    }
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
    runner(b, target, optimize, ocloc, device, prebuilt);
}

/// The device runner: gates 1 to 4 and gate 5 live on a card, with the images linked in; `prebuilt` is -Dxpu-images.
pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8, prebuilt: ?[]const u8) void {
    const core = kernel_contract.module(b, target, optimize);
    const ze = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false); // the runner links its own images
    options.addOption(bool, "dpas", true);
    const xpu = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    xpu.addOptions("kernel_options", options);
    xpu.addImport("ze", ze);
    xpu.addImport("core", core);

    const files = b.addWriteFiles();
    for (sources) |src| {
        const bin = if (prebuilt) |dir|
            std.Build.LazyPath{ .cwd_relative = b.fmt("{s}/{s}.bin", .{ dir, src.name }) }
        else
            object(b, ocloc, device, src);
        _ = files.addCopyFile(bin, b.fmt("{s}.bin", .{src.name}));
    }
    const images = b.createModule(.{ .root_source_file = files.add("xpu_contract_images.zig",
        \\pub const glue align(8) = @embedFile("glue.bin").*;
        \\pub const decode align(8) = @embedFile("decode.bin").*;
        \\pub const attn_upstream align(8) = @embedFile("contract_attn_upstream.bin").*;
        \\pub const controls align(8) = @embedFile("contract_controls.bin").*;
        \\pub const gate_controls align(8) = @embedFile("contract_gate_controls.bin").*;
    ), .target = target, .optimize = optimize });
    const exe = b.addExecutable(.{ .name = "tf-xpu-contract-run", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/contract_run.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "ze", .module = ze },
            .{ .name = "xpu_contract_images", .module = images },
        },
    }) });
    b.step("tf-xpu-contract-run", "Build the XPU kernel-contract runner: gates 1 to 4 and gate 5 live on a card").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const run = b.addRunArtifact(exe);
    b.step("xpu-contract-run", "Run the XPU kernel contract on this card (tf-xpu-contract-run [--device N])").dependOn(&run.step);
}

const Source = struct { name: []const u8, path: []const u8 };
const sources = [_]Source{
    .{ .name = "glue", .path = "zig/kernels/xpu/qwen27/glue.cl" },
    .{ .name = "decode", .path = "zig/kernels/xpu/qwen27/decode.cl" },
    .{ .name = "contract_attn_upstream", .path = "zig/kernels/xpu/contract_attn_upstream.cl" },
    .{ .name = "contract_controls", .path = "zig/kernels/xpu/contract_controls.cl" },
    .{ .name = "contract_gate_controls", .path = "zig/kernels/xpu/contract_gate_controls.cl" },
};

fn object(b: *std.Build, ocloc: []const u8, device: []const u8, src: Source) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    run.addFileArg(b.path(src.path));
    const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread -I zig/kernels/xpu/qwen27";
    run.addArgs(&.{ "-device", device, "-options", opts, "-output_no_suffix", "-output", src.name, "-out_dir" });
    run.setCwd(b.path("."));
    for ([_][]const u8{ "shape.h", "glue_attn.cl", "glue_quant.cl" }) |part| {
        run.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
    }
    const dir = run.addOutputDirectoryArg(b.fmt("ocloc-contract-{s}", .{src.name}));
    return dir.path(b, b.fmt("{s}.bin", .{src.name}));
}
