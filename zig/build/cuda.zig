//! The CUDA half of the root build: kernel fatbins with their Python extensions' nvcc flags, the runtime, the CLI.
const std = @import("std");
const exlr8_build = @import("exlr8.zig");
const golden_build = @import("golden.zig");
const catalog = @import("cuda_kernels.zig");
pub const Kernel = catalog.Kernel;
pub const kernels = catalog.kernels;

pub const runtime = @import("cuda_runtime.zig").runtime;
const stagger = @import("cuda_runtime.zig").stagger;

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module, tokenizer: *std.Build.Module, heat: *std.Build.Module, radio: *std.Build.Module, living: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = optimize, .link_libc = true });
    // Living Weights' family-neutral learner: a family's lessons, and the server's view of them, import it
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("living", living);
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("living", living);
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("heat", heat);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const radio = radioModule(b, target, optimize, cuda, core, lanes);
    nemotron.addImport("radio", radio);
    nemotron.addImport("living", living);
    const qwen35 = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen35.addImport("living", living);
    qwen35.addImport("nemotron", nemotron);
    qwen35.addImport("cuda", cuda);
    qwen35.addImport("core", core);
    qwen35.addImport("lanes", lanes);
    const qwen_vision = qwenVisionModule(b, target, optimize, cuda, core, lanes);
    qwen35.addImport("qwen_vision", qwen_vision);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .qwen35 = qwen35, .qwen_vision = qwen_vision, .tokenizer = tokenizer, .heat = heat, .radio = radio, .living = living };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, opts: @import("options.zig").Cuda) void {
    const nvcc = opts.nvcc;
    const prebuilt = opts.fatbins;
    const sms = opts.sms;
    const strip = opts.strip;
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    // MiniMax H3's kernel image: its library and the contract's runner take it
    const h3_image = if (nvcc) |tool| fatbin(b, tool, version orelse blk: {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    }, .{ .name = "h3", .flags = &.{"-O3"} }, sms) else null;
    @import("h3_host.zig").library(b, target, optimize, h3_image);
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    const native = engines(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "cuda", cuda }, .{ "core", mods.core }, .{ "lanes", mods.lanes }, .{ "nemotron", mods.nemotron }, .{ "dory", native.engines.import_table.get("dory").? } }) |m| cli.addImport(m[0], m[1]);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, target, optimize, strip, native.engines, mods.core));
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    const dump_options = b.addOptions();
    dump_options.addOption(bool, "enabled", opts.test_dump);
    runner.addOptions("cuda_dump_options", dump_options);
    runner.addImport("cuda", cuda);
    runner.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    runner.addImport("core", mods.core);
    runner.addImport("lanes", mods.lanes);
    runner.addImport("nemotron", mods.nemotron);
    runner.addImport("qwen35", mods.qwen35);
    runner.addImport("flashnext", flashnextModule(b, target, optimize, cuda, mods.core, mods.lanes, mods.qwen_vision));
    runner.addImport("lfm2", lfm2Module(b, target, optimize, cuda, mods.core));
    runner.addImport("radio", mods.radio);
    const exlr8 = exlr8_build.coreAlias(b, target, optimize, mods.core);
    runner.addImport("exlr8", exlr8);
    const exlr8_tests = exlr8_build.testModules(b, target, optimize, exlr8);
    for (exlr8_tests) |m| runner.addImport(m.name, m.module);
    runner.addImport("exlr8_frozen", b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/exlr8/frozen.zig"), .target = target, .optimize = optimize }));
    runner.addImport("exlr8_frozen_dense", b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/exlr8/frozen_dense.zig"), .target = target, .optimize = optimize }));
    for (exlr8_build.denseModules(b, target, optimize, exlr8, exlr8_tests[0])) |m| runner.addImport(m.name, m.module);
    const glm_cuda = glmCuda(b, target, optimize, cuda, mods.core, mods.lanes, mods.heat);
    runner.addImport("glm_cuda", glm_cuda);
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "kolibri1", kolibri1Module(b, target, optimize, cuda, mods.core, mods.lanes) }, .{ "dory", @import("dory.zig").module(b, target, optimize, cuda, mods.core, mods.lanes) } }) |m| runner.addImport(m[0], m[1]);
    runner.addImport("gemma4_cuda", @import("gemma4_cuda.zig").module(b, target, optimize, cuda, mods.core, mods.lanes));
    const runner_exe = b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner });
    b.installArtifact(runner_exe);
    // tf-tp-watch-check: the ranks' lines with no GPU or NCCL; a failing or dying rank stops every rank
    b.installArtifact(b.addExecutable(.{ .name = "tf-tp-watch-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/tp_watch_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "cuda", .module = cuda }} }) }));
    @import("living_dense.zig").target(b, target, optimize, cuda, mods.qwen35);
    flashnextChecks(b, target, optimize, cuda, mods.core, mods.lanes, mods.qwen_vision, opts);
    b.step("compile-cuda-tests", "Compile CUDA runner without executing a GPU").dependOn(&runner_exe.step);
    const startup = b.addObject(.{ .name = "modelopt-startup", .root_module = b.createModule(.{ .root_source_file = b.path("zig/modelopt_startup_compile.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "native", .module = native.engines }} }) });
    _ = startup.getEmittedBin();
    b.step("compile-modelopt-startup", "Compile actual native startup with test-only absent GPU images").dependOn(&startup.step);
    const controls_file = opts.contract_controls;
    const controls = if (controls_file) |file| b.graph.cwdRelativePath(file) else if (nvcc) |tool| fatbin(b, tool, version.?, .{ .name = "contract_controls", .flags = &.{"-O3"} }, sms) else null;
    const contract_images = @import("qwen_contract.zig").images(b, runner_exe, controls, h3_image, kernels, &images);
    contract_images.dependOn(fatbin_step);
    @import("cuda_contract_deepseek.zig").images(b, runner_exe, contract_images, kernels, &images, prebuilt, nvcc, version, sms, opts.cuobjdump);
    const server_exe = nativeServer(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, true, strip);
    b.step("compile-config-server", "Compile the actual CUDA server config entry point without installing or running").dependOn(&server_exe.step);
    // tf-qwen35-cuda: the Qwen CUDA forward's rows for a reference to compare, and its tower on picture files
    const q35 = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen35_cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    q35.addImport("cuda", cuda);
    q35.addImport("core", mods.core);
    q35.addImport("qwen35", mods.qwen35);
    b.installArtifact(b.addExecutable(.{ .name = "tf-qwen35-cuda", .root_module = q35 }));
    const d3_tool = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = mods.core }, .{ .name = "drama3", .module = @import("drama3_host.zig").module(b, target, optimize, cuda, mods.core) } } });
    const d3_exe = b.addExecutable(.{ .name = "drama3-generate", .root_module = d3_tool });
    b.step("drama3-generate", "Drama generation and replay controls on CUDA").dependOn(&b.addInstallArtifact(d3_exe, .{}).step);
}

/// The GLM-5.3-Flash CUDA family over the runtime, the core, the lane core, canon's shares and core/heat.zig.
fn glmCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, heat: *std.Build.Module) *std.Build.Module {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("canon", canon);
    m.addImport("lanes", lanes);
    m.addImport("heat", heat);
    return m;
}

/// The Kolibri-1 CUDA family over the runtime, the core and the lane core.
const kolibri1Module = @import("kolibri.zig").module;

/// LFM2 (d1's decision model) on CUDA: config, weights, kernels and the forward that scores labels.
fn lfm2Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm2/lfm2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// RADIO image input (Nemotron Omni checkpoints): the processor, the tower and projector, the prompt rows.
fn radioModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/radio/radio.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

/// The Qwen-VL tower on CUDA that Qwen and Flash Next share, one module for a build so both import the same one.
fn qwenVisionModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

/// Flash Next's CUDA family, sharing the maintained runtime and lane core.
fn flashnextModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_launch.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("qwen_vision", qwen_vision);
    m.addImport("living", core.import_table.get("living").?);
    return m;
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "engine_api", .module = api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "nemotron", .module = nemotron },
            .{ .name = "drama3", .module = @import("drama3_host.zig").module(b, target, optimize, cuda, core) },
            .{ .name = "lfm2", .module = lfm2Module(b, target, optimize, cuda, core) },
            .{ .name = "core", .module = core },
            .{ .name = "flashnext", .module = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision) },
            .{ .name = "glm_cuda", .module = glmCuda(b, target, optimize, cuda, core, lanes, heat) },
            .{ .name = "qwen35", .module = qwen35 },
            .{ .name = "kolibri1", .module = kolibri1Module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "gemma4_cuda", .module = @import("gemma4_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "dory", .module = @import("dory.zig").module(b, target, optimize, cuda, core, lanes) },
        },
    });
    return .{ .api = api, .engines = mod };
}

/// The checkpoint subcommands' module: `models`, `info` and `pull` over the CUDA families.
fn checkpointCli(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, engines_mod: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "native_engines", .module = engines_mod }, exlr8_build.cliImport(b, target, optimize, core), .{ .name = "core", .module = core } } });
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module, tokenizer: *std.Build.Module, living: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool, strip: ?bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, core, lanes, nemotron, heat, qwen35, qwen_vision);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "living", .module = living }, .{ .name = "checkpoint_cli", .module = checkpointCli(b, target, .ReleaseSafe, null, m.engines, core) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    _ = @import("server.zig").addTo(b, exe.root_module, target, .ReleaseSafe);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    // before the install step, which decides then whether a .pdb comes with the executable
    exe.root_module.strip = strip;
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}
/// CPU tests for the CUDA runtime, shared core, lanes and CUDA family, on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, all: *std.Build.Step, server_steps: bool) void {
    const step = b.step("test-cuda-host", "The CUDA side's host unit tests alone (no GPU work)");
    all.dependOn(step);
    @import("release_version.zig").tests(b, step);
    const flags = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/build/cuda_release_flags.zig"),
        .target = b.graph.host,
    }) }));
    b.step("test-cuda-release-flags-host", "All CUDA image flags omit private source line data").dependOn(&flags.step);
    step.dependOn(&flags.step);
    @import("cuda_release_paths.zig").tests(b, step);
    @import("runtime_libraries.zig").tests(b, step);
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = host }) })).step);
    const cuda = runtime(b, host, .debug, &.{});
    b.step("test-nccl-version-host", "Shared NCCL minimum without a communicator or GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"NCCL minimum:"} })).step);
    // H3's library opens its checkpoint with mmap: not Windows
    if (host.result.os.tag != .windows) @import("h3_host.zig").targets(b, cuda, step);
    b.step("test-gradient-host", "Gradient bounds through real wrappers with recorded CUDA launches").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"train launch bounds:"} })).step);
    const runtime_host = b.step("test-runtime-host", "Shared CUDA runtime and graph host regressions");
    runtime_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda })).step);
    const mods = family(b, host, .debug, cuda, draft_ids);
    @import("qwen_contract.zig").host(b, mods.core, step);
    @import("qwen_vision_weights_host.zig").host(b, mods.core, step);
    @import("inc4_host.zig").round(b, cuda, mods.core);
    @import("inc4_host.zig").rows(b, cuda, mods.core);
    @import("living_dense.zig").host(b, cuda, mods.qwen35, draft_ids, runtime, family, kernels.len);
    @import("cuda_family_host.zig").steps(b, cuda, mods, step);
    const exl3_host = b.step("test-exl3-cuda-host", "EXL3 loader and CUDA host regressions without a GPU");
    const exl3_compile = b.addObject(.{ .name = "exl3_host_compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exl3_compile.zig"),
        .target = host,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) });
    step.dependOn(&exl3_compile.step);
    exl3_host.dependOn(&exl3_compile.step);
    for ([_]*std.Build.Module{ cuda, mods.core }) |m| {
        exl3_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_arch_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } }) })).step);
    @import("flashnext_glue_host.zig").steps(b, cuda, mods.core, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), step);
    @import("flashnext_scoring_host.zig").steps(b, cuda, mods.core, mods.lanes, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), step);
    b.step("test-pocket-device-host", "Actual pocket device hooks under CUDA ABI failures").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), .filters = &.{"Learning persistence:"} })).step);
    b.step("test-living-host", "Family-neutral learner, saved form and commit gate on host").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.living })).step);
    const shard_tests = b.createModule(.{ .root_source_file = b.path("zig/src/living_shards_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    shard_tests.addImport("living", mods.living);
    b.step("test-living-shards-host", "Quantized save and reopen through actual shard writer").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = shard_tests })).step);
    const writeback_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_flashnext_writeback.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) }));
    step.dependOn(&writeback_contract.step);
    @import("cuda_row_contracts.zig").host(b, host, cuda, mods.core, step);
    @import("cuda_window_host.zig").steps(b, host, step);
    const utility_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_utility_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&utility_contract.step);
    const flash_contract = b.step("test-flashnext-contract-host", "Flash Next contract stress and rounding references without a GPU");
    flash_contract.dependOn(&writeback_contract.step);
    flash_contract.dependOn(&utility_contract.step);
    const row_best_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_row_best_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&row_best_contract.step);
    flash_contract.dependOn(&row_best_contract.step);
    const qsa_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_qsa_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&qsa_contract.step);
    flash_contract.dependOn(&qsa_contract.step);
    @import("cuda_window_host.zig").contracts(b, host, cuda, mods.core, step);
    const reference_step = b.step("test-cuda-reference", "CUDA's FP64 comparison with CPU-only readback (no GPU)");
    const projection_step = b.step("test-cuda-projection-fixture", "Projection fixture dispatch controls with CPU-only CUDA calls");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const reference = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/quant_ref_host_tests.zig"),
            .target = host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
        }) }));
        step.dependOn(&reference.step);
        reference_step.dependOn(&reference.step);
        const projection = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/projection_f32_host_tests.zig"),
            .target = host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
        }) }));
        step.dependOn(&projection.step);
        projection_step.dependOn(&projection.step);
    }
    const bolt = b.step("test-bolt-host", "Nemotron Bolt and shared lane host contracts");
    const cuda_target = b.resolveTargetQuery(.{ .cpu_arch = host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
    const check_cuda = runtime(b, cuda_target, .debug, &.{});
    const check_mods = family(b, cuda_target, .debug, check_cuda, draft_ids);
    @import("cuda_census.zig").steps(b, cuda_target, check_cuda, check_mods.qwen35, check_mods.core, flashnextModule(b, cuda_target, .debug, check_cuda, check_mods.core, check_mods.lanes, check_mods.qwen_vision));
    const compile_bolt = b.addObject(.{ .name = "bolt_host_compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/bolt_compile.zig"),
        .target = cuda_target,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = check_cuda }, .{ .name = "nemotron", .module = check_mods.nemotron } },
    }) });
    bolt.dependOn(&compile_bolt.step);
    step.dependOn(&compile_bolt.step);
    for ([_]*std.Build.Module{ mods.nemotron, mods.lanes, mods.radio }) |m|
        bolt.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const video_step = b.step("test-omni-video-host", "Actual RADIO and Bolt video boundaries without a GPU or decoder library");
    const video_filter = b.option([]const u8, "omni-video-filter", "One Omni video regression");
    for ([_]*std.Build.Module{ mods.nemotron, mods.radio }) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = if (video_filter) |f| &.{f} else &.{} }));
        run.has_side_effects = true;
        video_step.dependOn(&run.step);
    }
    const drama3 = @import("drama3_host.zig").hostTests(b, step, cuda, mods.core);
    const native_modules = engines(b, host, .debug, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    const native = native_modules.engines;
    @import("cuda_fault_host.zig").steps(b, cuda, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), native);
    @import("release_cuda_refusal.zig").tests(b, step, native_modules.api, mods.nemotron);
    @import("qwen_cost_open.zig").tests(b, mods, native_modules, step);
    const windows_port = @import("windows_host.zig").steps(b, cuda, native_modules.api);
    const adapter_tests = b.createModule(.{ .root_source_file = b.path("zig/src/native/slide_adapter_test.zig"), .target = host, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = native_modules.api }} });
    b.step("test-slide-adapter-host", "Slide adapter: initial learner progress, and --slide's startup refusals").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = adapter_tests })).step);
    @import("drama3_host.zig").native(b, step, drama3, mods.core, native_modules.api, cuda, "drama_cuda_native_host");
    @import("drama3_host.zig").plan(b, drama3, mods.core, native_modules.api, cuda);
    @import("drama3_host.zig").metal(b, mods.core, native_modules.api);
    const safety = b.step("test-kv-safety-host", "Actual native KV startup budget and paged constructors under allocation/completion faults");
    for ([_]*std.Build.Module{ native, mods.nemotron }) |m| safety.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"kv safety:"} })).step);
    b.step("test-cache-capacity-host", "Startup cache capacity facts and warnings without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"cache capacity:"} })).step);
    const merge_checks = b.step("test-kv-merge-host", "KV page integration with landed snapshot modes and budgets");
    for ([_]*std.Build.Module{ native, mods.nemotron }) |m| merge_checks.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"kv merge:"} })).step);
    const nemo_safety = b.step("test-nemotron-learning-host", "Actual Nemotron CUDA close, checked pocket reads and retry under fake driver failures");
    for ([_]*std.Build.Module{ native, mods.nemotron }) |m| nemo_safety.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"Learner ownership:"} })).step);
    const load_check = b.step("test-bolt-load-host", "Bolt load destination and native-parent retention without a GPU");
    const load_filter = b.option([]const u8, "bolt-load-filter", "Select one load regression");
    const load_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_load_gpu.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "nemotron", .module = mods.nemotron } } });
    for ([_]*std.Build.Module{ load_module, mods.radio }) |m| {
        const check = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = if (load_filter) |f| &.{f} else &.{} }));
        check.has_side_effects = true;
        load_check.dependOn(&check.step);
    }

    b.step("test-runtime-api", "Optional planner, prompt piece, proposer and halt API tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api })).step);
    b.step("test-api-boundaries", "Runtime thread initialization failure regression").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api, .filters = &.{"Failure boundaries:"} })).step);
    b.step("test-review-api", "Ownership rollback: ownership and final batching regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api, .filters = &.{"Ownership rollback:"} })).step);
    const check_native = nativeServer(b, cuda_target, .debug, check_cuda, check_mods.core, check_mods.lanes, check_mods.nemotron, check_mods.heat, check_mods.qwen35, check_mods.qwen_vision, check_mods.tokenizer, check_mods.living, build_options, false, null);
    b.step("check-native-cuda-host", "Compile the real native CUDA server without kernels or GPU execution").dependOn(&check_native.step);

    const offer_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_vision_offer.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "nemotron", .module = mods.nemotron }, .{ .name = "qwen35", .module = mods.qwen35 } } }), .filters = if (video_filter) |f| &.{f} else &.{} }));
    offer_tests.has_side_effects = true;
    video_step.dependOn(&offer_tests.step);
    const compile_modules = engines(b, cuda_target, .debug, check_cuda, check_mods.core, check_mods.lanes, check_mods.nemotron, check_mods.heat, check_mods.qwen35, check_mods.qwen_vision);
    @import("glm_family_host.zig").compile(b, cuda_target, check_cuda, check_mods, glmCuda);
    const open_object = b.addObject(.{ .name = "omni-video-native-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/omni_video_compile.zig"), .target = cuda_target, .link_libc = true, .imports = &.{.{ .name = "native", .module = compile_modules.engines }} }) });
    _ = open_object.getEmittedBin();
    b.step("compile-omni-video-host", "Compile actual native CUDA startup and offer for Linux without images or GPU execution").dependOn(&open_object.step);
    const h3_checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/checkpoint.zig"), .target = host, .optimize = .debug, .link_libc = true });
    // the checkpoint maps its file with mmap: not Windows
    if (host.result.os.tag != .windows) step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = h3_checkpoint })).step);
    @import("pull_tests.zig").add(b, host, .debug, native_modules.engines, mods.core, step);
    b.step("test-save-reopen-host", "Unresolved save through actual checkpoint admission").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.core, .filters = &.{"Save recovery:"} })).step);
    b.step("test-trainer-close-host", "Actual native close and real Flash Next trainer ownership sentinels").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"Save recovery:"} })).step);
    // the HTTP server's unit tests (zig/src/server/root.zig), which the macOS build runs, on Linux hosts too
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template }, .{ .name = "living", .module = mods.living } },
    });
    const server_json = @import("server.zig").addTo(b, server_tests, host, .debug);
    @import("proposal_vocab_host.zig").steps(b, mods.core);
    const eos_policy = b.step("test-eos-policy-host", "Shared EOS, real HfText and HTTP policy without a GPU");
    eos_policy.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"EOS policy:"} })).step);
    eos_policy.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.tokenizer, .filters = &.{"EOS policy:"} })).step);
    const eos_family = kolibri1Module(b, host, .debug, cuda, mods.core, mods.lanes);
    eos_policy.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eos_family, .filters = &.{"EOS policy:"} })).step);
    b.step("test-dory-cli-host", "Dory CUDA loop option forwarding and invalid-value refusals").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"--device and --segments:"} })).step);
    b.step("test-dory-chat-host", "Dory chat template, EOS, reasoning split and qwen3_coder call on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Dory chat:"} })).step);
    b.step("test-embeddings-protocol-host", "Embedding request and response contracts without an admitted model or GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"embeddings protocol:"} })).step);
    b.step("test-embeddings-http-host", "Embedding route capability and borrowed request lifetime without a model or GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"embeddings HTTP:"} })).step);
    @import("safetensors_header_host.zig").steps(b, mods.core, step);
    b.step("test-runtime-capabilities-host", "Optional library discovery in actual capabilities JSON").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"runtime library:"} })).step);
    b.step("test-vision-cli-host", "Image token budgets through parsing and engine admission options without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"vision image budget:"} })).step);
    windows_port.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"hardware names"} })).step);
    b.step("test-benchmark-manifest-host", "Manifest identity through the HTTP server without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"manifest"} })).step);
    b.step("test-lesson-questions-host", "Supplied questions through the production lesson without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Lesson questions:"} })).step);
    b.step("test-slide-recovery-host", "Living Weights conversation and stream regressions without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "lesson conversations:", "lesson stream:", "lesson writer:" } })).step);
    b.step("test-issue-934", "Benchmark config admission and receipt regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"issue 934:"} })).step);
    b.step("test-issue-969", "First-EOS classification and incomplete benchmark messages without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"issue 969"} })).step);
    b.step("test-tool-required-host", "Required tool parsing, grammar form and constrained lane replies on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"required tools:"} })).step);
    const q27 = @import("qwen27_required.zig").modules(b, mods.lanes);
    server_tests.addImport("qwen27_mask_fake", q27.fake);
    server_tests.addImport("qwen27_masks", q27.masks);
    b.step("test-qwen27-required-host", "Qwen27 required tool routes through the masked lane fake").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"qwen27 tool calls:"} })).step);
    b.step("test-save-graph-host", "Actual graph receipts paired with generic learner persistence failures").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Save recovery:"} })).step);
    b.step("test-lesson-gate-host", "Production server rounds and earlier-fact retention").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Learning persistence:"} })).step);
    b.step("test-frame-cache-identity-host", "Image frame identities through the actual shared cache").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api, .filters = &.{"Frame cache:"} })).step);
    const media_gate = b.step("test-media-request-ownership-host", "Shared request ownership, image frame caches and media expansion on the CPU");
    media_gate.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api })).step);
    media_gate.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "image", "video" } })).step);
    media_gate.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.core, .filters = &.{ "qwen_vl", "video" } })).step);
    b.step("test-http-limits-host", "Total HTTP read deadlines and connection admission without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"HTTP limits:"} })).step);
    b.step("test-server-recovery-host", "Incomplete replies, token allocation and bounded SSE storage without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Server recovery:"} })).step);
    const server_test = b.addRunArtifact(b.addTest(.{ .root_module = server_tests }));
    b.step("test-done-times-host", "Request-local done line times and lane admission on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"done times:"} })).step);
    b.step("test-request-seeds-host", "Fresh and explicit request seeds and completion logs without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"request seed:"} })).step);
    const json_test = b.addRunArtifact(b.addTest(.{ .root_module = server_json }));
    const publication = b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"benchmark publication:"} }));
    b.step("test-benchmark-publication-host", "Benchmark command publication boundaries with inert effects").dependOn(&publication.step);
    b.step("test-benchmark-routing-host", "Benchmark command routing from explicit model capabilities").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"benchmark routing:"} })).step);
    const server_step = b.step(
        "test-server-cpu",
        "The HTTP server's unit tests (routes, templates, tool parsing) without a GPU",
    );
    for ([_]*std.Build.Step.Run{ server_test, json_test }) |run| {
        step.dependOn(&run.step);
        server_step.dependOn(&run.step);
    }
    // the server goldens and Cortheon's fuzz step here when the target is not macOS, whose Metal build defines them
    if (server_steps) {
        golden_build.step(b, host, server_tests, native_modules.api, mods.tokenizer, template, native_modules.engines);
        @import("cortheon_fuzz.zig").add(b, host, native_modules.api, mods.tokenizer, template);
    }
    // the tokenizer's own tests (memo, steps, regex, unicode) run here: no other test binary reaches them
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.qwen35, mods.qwen_vision, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), lfm2Module(b, host, .debug, cuda, mods.core), mods.radio, mods.heat, mods.tokenizer, native_modules.api, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const selected = b.option([]const u8, "modelopt-host-filter", "One portable ModelOpt regression") orelse "modelopt host:";
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mods.nemotron, .filters = &.{selected} }));
    run.has_side_effects = true;
    b.step("test-modelopt-host", "Actual Nemotron scratch and CUDA launch records without GPU execution").dependOn(&run.step);
    // the contract's attention reference (tests/cuda/contract_attn.zig): its allocations under failure
    const attn_ref = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_attn.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "core", .module = mods.core }} });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = attn_ref })).step);
    const glm = glmCuda(b, host, .debug, cuda, mods.core, mods.lanes, mods.heat);
    @import("glm_family_host.zig").steps(b, cuda, glm, mods, step);
    @import("kolibri.zig").host(b, kolibri1Module(b, host, .debug, cuda, mods.core, mods.lanes), native_modules.api, step);
    @import("gemma4_cuda.zig").host(b, @import("gemma4_cuda.zig").module(b, host, .debug, cuda, mods.core, mods.lanes), step);
    @import("glm_contracts.zig").tests(b, glm, mods.lanes, cuda, mods.core);
    // the shares a GLM-5.3-Flash rank takes (cluster/canon.zig against the Python family's)
    const shares = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon_glm_test.zig"), .target = host, .optimize = .debug });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = shares })).step);
    // the tool-call parser's tests, run beside the server's on Macs: the GLM calls the CUDA port serves
    const parse_tests = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } } });
    _ = @import("server.zig").addTo(b, parse_tests, host, .debug);
    @import("server.zig").toolHostTests(b, parse_tests, server_tests, step);
    // Living Weights: the family-neutral learner, then the pocket in each family's shards over the core's shard files
    const living_run = b.addRunArtifact(b.addTest(.{ .root_module = mods.living }));
    const shards = b.createModule(.{ .root_source_file = b.path("zig/src/living_shards_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "living", .module = mods.living }} });
    const shards_run = b.addRunArtifact(b.addTest(.{ .root_module = shards }));
    step.dependOn(&living_run.step);
    step.dependOn(&shards_run.step);
    const living_step = b.step("test-living", "Living Weights' learner and the pocket in each family's shards, on the host");
    living_step.dependOn(&living_run.step);
    living_step.dependOn(&shards_run.step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, host, .debug, false, native_modules.engines, mods.core));
    const memory_cli = checkpointCli(b, host, .debug, false, native_modules.engines, mods.core);
    memory_cli.root_source_file = b.path("zig/src/cli/tests.zig");
    b.step("test-cli-memory-host", "Actual checkpoint info memory tests without device or shader producers").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = memory_cli, .filters = &.{ "info prints family", "issue 934:" } })).step);
    b.step("test-info-config", "Info shares the checkpoint config reader and diagnostics").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli, .filters = &.{"issue 934:"} })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
pub const fatbin = @import("cuda_fatbin.zig").fatbin;

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, false, null);
}

/// SASS targets every release fatbin holds: Ampere (sm_80, sm_86), RTX 50/PRO Blackwell (sm_120), GB10 (sm_121).
pub const release_sms = "80,86,120,121";

/// Validate the complete named input set, including images unused by today's server, and each file's release targets.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8, privacy_tool: *std.Build.Step.Compile) *std.Build.Step {
    const tool = b.addExecutable(.{ .name = "check-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = b.graph.host }) });
    const check = b.addRunArtifact(tool);
    check.addArg(release_sms);
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    const privacy = @import("cuda_release_paths.zig").directory(b, privacy_tool, b.graph.cwdRelativePath(dir));
    privacy.dependOn(&check.step);
    return privacy;
}

/// Explicit model diagnostics; these tools do not qualify a checkpoint merely by compiling.
fn flashnextChecks(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module, opts: @import("options.zig").Cuda) void {
    const flashnext = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision);
    // not installed: one embedding row through the captured _embed cubin
    const embed_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_embed_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    embed_check.addImport("cuda", cuda);
    embed_check.addImport("core", core);
    embed_check.addImport("flashnext", flashnext);
    const embed_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-embed", .root_module = embed_check }));
    const embed_model = opts.flashnext_model;
    const embed_kernels = opts.flashnext_kernels;
    const embed_token = opts.flashnext_token;
    if (embed_model != null and embed_kernels != null) embed_run.addArgs(&.{ embed_model.?, embed_kernels.?, embed_token });
    b.step("flashnext-embed", "Upload the Flash Next embedding table and run _embed for one token").dependOn(&embed_run.step);
    const prompt_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_prompt_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    prompt_check.addImport("cuda", cuda);
    prompt_check.addImport("core", core);
    prompt_check.addImport("flashnext", flashnext);
    const prompt_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-prompt", .root_module = prompt_check }));
    const prompt_ids = opts.flashnext_ids;
    const prompt_count = opts.flashnext_count;
    if (embed_model != null and embed_kernels != null and prompt_ids != null) prompt_run.addArgs(&.{ embed_model.?, embed_kernels.?, prompt_ids.?, prompt_count });
    b.step("flashnext-prompt", "Run one Flash Next prompt on the captured CUDA kernels").dependOn(&prompt_run.step);
    // not installed: our own glue against a captured set on one loaded engine (scores of every prompt row, replies)
    const compare_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_compare_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    compare_check.addImport("cuda", cuda);
    compare_check.addImport("flashnext", flashnext);
    const compare_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-compare", .root_module = compare_check }));
    if (embed_model != null and embed_kernels != null and prompt_ids != null) compare_run.addArgs(&.{ embed_model.?, embed_kernels.?, prompt_ids.? });
    b.step("flashnext-compare", "Our Flash Next glue against a captured set: every prompt row's scores, perplexity, greedy replies").dependOn(&compare_run.step);
    // not installed: the native block selection against Triton's lists (capture_sparse.sh's fixtures)
    const select_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_select_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    select_check.addImport("cuda", cuda);
    const select_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-select", .root_module = select_check }));
    if (opts.flashnext_fixtures) |dir| select_run.addArgs(&.{dir});
    b.step("flashnext-select", "fn_qsa_select against Triton's _select / _select_tiles lists on captured score rows").dependOn(&select_run.step);
    const learn_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_learn_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    learn_check.addImport("cuda", cuda);
    learn_check.addImport("flashnext", flashnext);
    const learn_exe = b.addExecutable(.{ .name = "flashnext-learn", .root_module = learn_check });
    const learn_run = b.addRunArtifact(learn_exe);
    if (embed_model != null and prompt_ids != null) learn_run.addArgs(&.{ embed_model.?, prompt_ids.? });
    b.step("flashnext-learn", "Trainer forward identity and all-path finite-difference gates").dependOn(&learn_run.step);
    b.step("compile-flashnext-learn", "Compile the real trainer runner without executing it").dependOn(&learn_exe.step);
}
