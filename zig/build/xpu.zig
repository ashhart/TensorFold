//! XPU (Intel Xe2 over Level Zero): opt-in steps for the Qwen3.8-27B runner (`xpu-qwen27`) and Nemotron (`-Dxpu`).

const std = @import("std");
const exlr8_build = @import("exlr8.zig");

/// ze.zig as its own module: the Qwen27 runner and the Nemotron runtime share one Level Zero table.
fn zeModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

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
    const ze_mod = zeModule(b, target, optimize);
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    module.addImport("xpu_qwen27_kernels", embedded);
    module.addImport("ze", ze_mod);
    const exe = b.addExecutable(.{ .name = "xpu-qwen27", .root_module = module });
    const install = b.addInstallArtifact(exe, .{});
    b.step("xpu-qwen27", "Build the Qwen3.8-27B XPU forward runner (needs ocloc, a C++17 compiler and nGEN headers)").dependOn(&install.step);

    // tensorfold-native with the XPU engines (native/xpu.zig), as the CUDA build's `native` step makes it.
    const qwen27 = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen27.addImport("xpu_qwen27_kernels", embedded);
    qwen27.addImport("ze", ze_mod);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nem = nemotronModules(b, target, optimize, ocloc, device, ze_mod, lanes);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "xpu_qwen27", .module = qwen27 }, .{ .name = "lanes", .module = lanes }, .{ .name = "xpu", .module = nem.xpu }, .{ .name = "nemotron_xpu", .module = nem.nemotron } } });
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
    nemotronSteps(b, target, optimize, nem);
}

/// One SPIR-V image: the .cl in zig/kernels/xpu (`src`, else `name`) and its ocloc options (dot/matrix kernels: CL3.0).
const Kernel = struct { name: []const u8, src: ?[]const u8 = null, options: []const u8 = "", include: bool = false };

const cl3 = "-cl-std=CL3.0";

/// An explicit list: other files in the folder never join the build. The SPIR-V is device independent.
const kernels = [_]Kernel{
    .{ .name = "attn", .options = cl3 },
    .{ .name = "basic", .options = cl3 },
    .{ .name = "moe", .options = cl3 },
    .{ .name = "mamba" },
    .{ .name = "glue" },
    .{ .name = "qmv4" },
    .{ .name = "vadd" },
    .{ .name = "nem_rows", .options = cl3 },
    .{ .name = "nem_rows_r1", .src = "nem_rows", .options = cl3 ++ " -DMAXR=1" },
    .{ .name = "nem_rows_r2", .src = "nem_rows", .options = cl3 ++ " -DMAXR=2" },
    .{ .name = "nem_rows_r4", .src = "nem_rows", .options = cl3 ++ " -DMAXR=4" },
    .{ .name = "nem_rows_r8", .src = "nem_rows", .options = cl3 ++ " -DMAXR=8" },
    .{ .name = "nem_prow", .options = cl3 },
    .{ .name = "nem_attn_dec", .options = cl3 },
};

/// The runtime module; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const std.Build.LazyPath, ze: *std.Build.Module) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", images.len == kernels.len);
    const xpu = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    xpu.addOptions("kernel_options", options);
    xpu.addImport("ze", ze);
    if (images.len == kernels.len) for (kernels, images) |k, image| xpu.addAnonymousImport(b.fmt("spv_{s}", .{k.name}), .{ .root_source_file = image });
    return xpu;
}

const Modules = struct { xpu: *std.Build.Module, core: *std.Build.Module, nemotron: *std.Build.Module, images: [kernels.len]std.Build.LazyPath = undefined };

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module, lanes: *std.Build.Module) Modules {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("xpu", xpu);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    return .{ .xpu = xpu, .core = core, .nemotron = nemotron };
}

/// Nemotron's SPIR-V images (built only when a step needs them), its runtime and its family module.
fn nemotronModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8, ze: *std.Build.Module, lanes: *std.Build.Module) Modules {
    var images: [kernels.len]std.Build.LazyPath = undefined;
    for (kernels, &images) |k, *image| image.* = spirv(b, ocloc, device, k);
    var m = family(b, target, optimize, runtime(b, target, optimize, &images, ze), lanes);
    m.images = images;
    return m;
}

fn cliModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module, mods: Modules) *std.Build.Module {
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/xpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("xpu", xpu);
    cli.addImport("core", mods.core);
    cli.addImport("nemotron_xpu", mods.nemotron);
    return cli;
}

/// ocloc -spv_only on one kernel source: `name`.spv in its own output directory.
fn spirv(b: *std.Build, ocloc: []const u8, device: []const u8, k: Kernel) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ ocloc, "compile", "-device", device, "-spv_only", "-output", k.name, "-output_no_suffix", "-q" });
    // ocloc builds from a copy of the source, so the header directory is named (the Run step's cwd is the build root)
    if (k.options.len > 0) run.addArgs(&.{ "-options", if (k.include) b.fmt("{s} -I zig/kernels/xpu", .{k.options}) else k.options });
    _ = k.include;
    run.addArg("-file");
    run.addFileArg(b.path(b.fmt("zig/kernels/xpu/{s}.cl", .{k.src orelse k.name})));
    run.addArg("-out_dir");
    const out = run.addOutputDirectoryArg(b.fmt("{s}_spirv", .{k.name}));
    return out.path(b, b.fmt("{s}.spv", .{k.name}));
}

/// The standalone op and model tests, each its own program `tf-xpu-<name>-test` (step `xpu-tests`).
const test_programs = [_][]const u8{
    "panic", "nem_rows", "nem_prefill", "nem_gen", "nem_mtp", "nem_kl", "nem_norm_bench", "nem_mv_bench", "nem_slot", "nem_format", "nem_seam", "nem_attn_dec", "nem_fp64",
};

/// `-Dxpu` (Linux): Nemotron's SPIR-V images, `tensorfold-xpu` and `tf-xpu-test`; needs ocloc, not nvcc or CUDA.
fn nemotronSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Modules) void {
    const enabled = b.option(bool, "xpu", "Build Nemotron on the Intel GPU into the install step (needs ocloc)") orelse false;
    if (!enabled or target.result.os.tag != .linux) return;
    const spirv_step = b.step("xpu-spirv", "Compile the Intel GPU kernels to SPIR-V and install them");
    for (kernels, mods.images) |k, image| spirv_step.dependOn(&b.addInstallFile(image, b.fmt("spirv/{s}.spv", .{k.name})).step);
    const xpu = mods.xpu;
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold-xpu", .root_module = cliModule(b, target, optimize, xpu, mods) }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/xpu/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("xpu", xpu);
    runner.addImport("core", mods.core);
    runner.addImport("nemotron_xpu", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tf-xpu-test", .root_module = runner }));
    const tests_step = b.step("xpu-tests", "Build the standalone Intel GPU op and model tests (tf-xpu-<name>-test)");
    for (test_programs) |name| {
        const mod = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/tests/xpu/{s}.zig", .{name})), .target = target, .optimize = optimize, .link_libc = true });
        mod.addImport("xpu", xpu);
        mod.addImport("core", mods.core);
        mod.addImport("nemotron_xpu", mods.nemotron);
        tests_step.dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = b.fmt("tf-xpu-{s}-test", .{name}), .root_module = mod }), .{}).step);
    }
}

/// Host unit tests of the Intel GPU runtime, the Nemotron family on it and its CLI (no GPU, no ocloc), on any host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const xpu = runtime(b, host, .debug, &.{}, zeModule(b, host, .debug));
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const mods = family(b, host, .debug, xpu, lanes);
    for ([_]*std.Build.Module{ xpu, mods.nemotron, cliModule(b, host, .debug, xpu, mods) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}
