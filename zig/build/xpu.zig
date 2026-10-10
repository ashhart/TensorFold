//! The Intel GPU half of the root build: OpenCL C kernels to SPIR-V with ocloc, the runtime, Nemotron, the CLI.

const std = @import("std");

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
    .{ .name = "nem_pf", .options = cl3 },
    .{ .name = "nem_attn_pfs", .options = cl3 },
    .{ .name = "nem_attn_dec", .options = cl3 },
};

/// The runtime module; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", images.len == kernels.len);
    const xpu = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    xpu.addOptions("kernel_options", options);
    if (images.len == kernels.len) for (kernels, images) |k, image| xpu.addAnonymousImport(b.fmt("spv_{s}", .{k.name}), .{ .root_source_file = image });
    return xpu;
}

const Modules = struct { core: *std.Build.Module, nemotron: *std.Build.Module, lanes: *std.Build.Module };

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module) Modules {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("xpu", xpu);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    return .{ .core = core, .nemotron = nemotron, .lanes = lanes };
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
    "panic", "nem_rows", "nem_prefill", "nem_gen", "nem_mtp", "nem_kl", "nem_norm_bench", "nem_mv_bench", "nem_slot", "nem_format", "nem_seam", "nem_attn_pf", "nem_attn_dec", "nem_fp64",
};

/// `-Dxpu` (Linux): the SPIR-V images, `tensorfold-xpu` and `tf-xpu-test`; needs ocloc, not nvcc or CUDA.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    _ = build_options;
    const enabled = b.option(bool, "xpu", "Build the Intel GPU engine (needs ocloc, no CUDA)") orelse false;
    const ocloc = b.option([]const u8, "ocloc", "ocloc for the Intel GPU kernels (default: ocloc)") orelse "ocloc";
    const device = b.option([]const u8, "xpu-device", "ocloc -device for the SPIR-V (default bmg)") orelse "bmg";
    if (!enabled) return;
    var images: [kernels.len]std.Build.LazyPath = undefined;
    const spirv_step = b.step("xpu-spirv", "Compile the Intel GPU kernels to SPIR-V and install them");
    for (kernels, &images) |k, *image| {
        image.* = spirv(b, ocloc, device, k);
        spirv_step.dependOn(&b.addInstallFile(image.*, b.fmt("spirv/{s}.spv", .{k.name})).step);
    }
    const xpu = runtime(b, target, optimize, &images);
    const mods = family(b, target, optimize, xpu);
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
    const xpu = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, xpu);
    for ([_]*std.Build.Module{ xpu, mods.nemotron, cliModule(b, host, .debug, xpu, mods) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}
