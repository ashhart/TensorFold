//! XPU (Intel Xe2 over Level Zero): opt-in steps; `xpu-qwen27` builds the Qwen3.8-27B forward runner and its kernels.

const std = @import("std");
const exlr8_build = @import("exlr8.zig");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    const ocloc = b.option([]const u8, "ocloc", "Intel offline compiler") orelse "ocloc";
    const device = b.option([]const u8, "xpu-device", "ocloc device for the XPU kernels") orelse "bmg";
    const ngen_include = b.option([]const u8, "ngen-include", "nGEN header directory (oneDNN)") orelse "ngen";
    const files = b.addWriteFiles();
    const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread";
    for ([_]struct { []const u8, []const u8, []const u8 }{
        .{ "glue", "zig/kernels/xpu/qwen27/glue.cl", "" },
        .{ "decode", "zig/kernels/xpu/qwen27/decode.cl", "" },
        .{ "gemm", "zig/kernels/xpu/qwen27/gemm.cl", " -DGSZ=64" },
        .{ "gemm128", "zig/kernels/xpu/qwen27/gemm.cl", " -DGSZ=128" },
        .{ "gdn", "zig/kernels/xpu/qwen27/gdn_chunk.cl", "" },
        .{ "drafter", "zig/kernels/xpu/qwen27/drafter.cl", "" },
    }) |k| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
        c.addFileArg(b.path(k[1]));
        c.addArgs(&.{ "-device", device, "-options", b.fmt("{s}{s} -I zig/kernels/xpu/qwen27", .{ opts, k[2] }), "-output_no_suffix", "-output", k[0], "-out_dir" });
        c.setCwd(b.path("."));
        for ([_][]const u8{ "glue_attn.cl", "glue_quant.cl" }) |part| c.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
        const dir = c.addOutputDirectoryArg(b.fmt("ocloc-{s}", .{k[0]}));
        _ = files.addCopyFile(dir.path(b, b.fmt("{s}.bin", .{k[0]})), b.fmt("{s}.bin", .{k[0]}));
    }
    // nGEN generator, built for the host and run at build time for the two prompt GEMM families (groups of 64 and 128).
    const gen = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O1", "-fno-operator-names", b.fmt("-I{s}", .{ngen_include}), "-o" });
    const gen_bin = gen.addOutputFileArg("w4_ngen");
    gen.addFileArg(b.path("zig/tools/xpu/w4_ngen.cpp"));
    for ([_][]const u8{ "w4_prompt.hpp", "w4a8_prompt.hpp" }) |h| gen.addFileInput(b.path(b.fmt("zig/tools/xpu/{s}", .{h})));
    for ([_]struct { []const u8, []const u8, []const []const u8 }{
        .{ "ngen_wide", "64", &.{"b3"} },
        .{ "ngen_narrow", "64", &.{ "s4", "asplit" } },
        .{ "ngen_wide128", "128", &.{"b3"} },
        .{ "ngen_narrow128", "128", &.{ "s4", "asplit" } },
        .{ "ngen_swiglu", "64", &.{ "b3", "swiglu" } },
        .{ "ngen_swiglu128", "128", &.{ "b3", "swiglu" } },
        .{ "ngen_a8", "128", &.{"a8"} },
        .{ "ngen_a8_swiglu", "128", &.{ "a8", "swiglu" } },
        .{ "ngen_a4", "128", &.{"a4"} },
    }) |z| {
        const run = std.Build.Step.Run.create(b, b.fmt("nGEN {s}", .{z[0]}));
        run.addFileArg(gen_bin);
        run.addArgs(&.{ "xe2", z[1] });
        const out = run.addOutputFileArg(b.fmt("{s}.zebin", .{z[0]}));
        run.addArgs(z[2]);
        _ = files.addCopyFile(out, b.fmt("{s}.zebin", .{z[0]}));
    }
    const embedded = b.createModule(.{ .root_source_file = files.add("xpu_qwen27_kernels.zig",
        \\pub const glue align(8) = @embedFile("glue.bin").*;
        \\pub const decode align(8) = @embedFile("decode.bin").*;
        \\pub const gemm align(8) = @embedFile("gemm.bin").*;
        \\pub const gemm128 align(8) = @embedFile("gemm128.bin").*;
        \\pub const gdn align(8) = @embedFile("gdn.bin").*;
        \\pub const drafter align(8) = @embedFile("drafter.bin").*;
        \\pub const ngen_wide align(8) = @embedFile("ngen_wide.zebin").*;
        \\pub const ngen_narrow align(8) = @embedFile("ngen_narrow.zebin").*;
        \\pub const ngen_wide128 align(8) = @embedFile("ngen_wide128.zebin").*;
        \\pub const ngen_narrow128 align(8) = @embedFile("ngen_narrow128.zebin").*;
        \\pub const ngen_swiglu align(8) = @embedFile("ngen_swiglu.zebin").*;
        \\pub const ngen_swiglu128 align(8) = @embedFile("ngen_swiglu128.zebin").*;
        \\pub const ngen_a8 align(8) = @embedFile("ngen_a8.zebin").*;
        \\pub const ngen_a8_swiglu align(8) = @embedFile("ngen_a8_swiglu.zebin").*;
        \\pub const ngen_a4 align(8) = @embedFile("ngen_a4.zebin").*;
        \\
    ) });
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    module.addImport("xpu_qwen27_kernels", embedded);
    // the FP64 references of the weight formats (core/quant), for XPU_FP64_REF and XPU_FIDELITY
    module.addImport("core", b.createModule(.{
        .root_source_file = b.path("zig/src/core/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }));
    const exe = b.addExecutable(.{ .name = "xpu-qwen27", .root_module = module });
    const install = b.addInstallArtifact(exe, .{});
    b.step("xpu-qwen27", "Build the Qwen3.8-27B XPU forward runner (needs ocloc, a C++17 compiler and nGEN headers)").dependOn(&install.step);

    // tensorfold-native with the XPU engines (native/xpu.zig), as the CUDA build's `native` step makes it.
    const qwen27 = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen27.addImport("xpu_qwen27_kernels", embedded);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "xpu_qwen27", .module = qwen27 }, .{ .name = "lanes", .module = lanes } } });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const checkpoint_cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, .ReleaseSafe, core) } });
    const server = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = checkpoint_cli } },
    }) });
    server.root_module.addOptions("build_options", build_options);
    const install_server = b.addInstallArtifact(server, .{ .dest_dir = .{ .override = .{ .custom = "xpu/bin" } } });
    b.step("xpu-native", "tensorfold-native with the XPU engines into zig-out/xpu/bin").dependOn(&install_server.step);
}
