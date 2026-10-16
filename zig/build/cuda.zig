//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");
const exlr8_build = @import("exlr8.zig");
const golden_build = @import("golden.zig");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

/// IEEE division and square root, no contraction or flush: host glue references match bit for bit.
const glue = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true" };

const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's serial feed, routed plan and rests
    .{ .name = "affine4_pack", .flags = &.{"-O3"} }, // ours: the MLX affine-4 repack into qlinear's tiles
    .{ .name = "fp8_experts", .flags = &.{"-O3"} }, // fp8/experts.cu's device code, tensorfold_fp8_experts_v6
    .{ .name = "exlr8", .flags = &.{ "-O3", "--fmad=false" } }, // ours: EXLR8 lane projections, the Metal kernels' order
    .{ .name = "exlr8_dense", .flags = &.{ "-O3", "--fmad=false" } }, // ours: EXLR8 dense matrices by row, the Metal order
    .{ .name = "nvfp4_experts", .flags = &.{"-O3"} }, // nvfp4/experts.cu's device code, the NVFP4 expert kernels
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "train", .flags = &.{"-O3"} }, // ours: Sliding Weights' learning step (train.metal's kernels)
    .{ .name = "nemotron_norms", .flags = glue }, // ours, each with a host reference (glue_ref.zig): Nemotron's glue
    .{ .name = "nemotron_route", .flags = glue },
    .{ .name = "nemotron_mamba", .flags = glue },
    .{ .name = "nemotron_attention", .flags = glue },
    .{ .name = "nemotron_keyed", .flags = glue },
    .{ .name = "norms", .flags = &.{"-O3"} }, // ours: RMS norm rows for every family (cuda/norms.zig)
    .{ .name = "embed_rows", .flags = &.{"-O3"} }, // ours: 4-bit affine and bf16 token rows (cuda/embed_rows.zig)
    .{ .name = "qmmf", .flags = &.{"-O3"} }, // nvfp4/qmmf.cu's device code (FP8G, FP4), tensorfold_nvfp4_v3
    .{ .name = "prompt16", .flags = &.{"-O3"} }, // nvfp4/prompt.cu's FP4 and FP8 tile 4, tensorfold_nvfp4_prompt_v1
    .{ .name = "modelopt_pack", .flags = &.{"-O3"} }, // ours: ModelOpt tensors into the lane, prompt and expert layouts
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    .{ .name = "drama_backbone", .flags = glue },
    .{ .name = "drama_backbone_gemm", .flags = &.{"-O3"} },
    .{ .name = "drama_tower", .flags = glue },
    .{ .name = "drama_codec", .flags = glue },
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, drama: *std.Build.Module, tokenizer: *std.Build.Module, heat: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("heat", heat);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const drama = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    drama.addImport("cuda", cuda);
    drama.addImport("core", core);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .drama = drama, .tokenizer = tokenizer, .heat = heat };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (default 121; 80, 86, 89, 120 and 121 build)") orelse "121";
    const strip = b.option(bool, "strip", "No debug info in the CUDA executables: no build machine paths leave with them") orelse false;
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
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    const native = engines(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.drama);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, target, optimize, strip, native.engines, mods.core));
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    runner.addImport("cuda", cuda);
    runner.addImport("core", mods.core);
    runner.addImport("lanes", mods.lanes);
    runner.addImport("nemotron", mods.nemotron);
    const exlr8 = exlr8_build.coreAlias(b, target, optimize, mods.core);
    runner.addImport("exlr8", exlr8);
    const exlr8_tests = exlr8_build.testModules(b, target, optimize, exlr8);
    for (exlr8_tests) |m| runner.addImport(m.name, m.module);
    runner.addImport("exlr8_frozen", b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/exlr8/frozen.zig"), .target = target, .optimize = optimize }));
    runner.addImport("exlr8_frozen_dense", b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/exlr8/frozen_dense.zig"), .target = target, .optimize = optimize }));
    for (exlr8_build.denseModules(b, target, optimize, exlr8, exlr8_tests[0])) |m| runner.addImport(m.name, m.module);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    nativeServer(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.drama, mods.tokenizer, build_options, true).root_module.strip = strip;
    const codec_oracle = b.addExecutable(.{ .name = "tf-drama-codec-oracle", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_codec_oracle.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_oracle = b.addInstallArtifact(codec_oracle, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-codec-oracle", "Build the production-codec numerical oracle driver; does not run a GPU").dependOn(&install_oracle.step);
    const codec_rvq_replay = b.addExecutable(.{ .name = "tf-drama-codec-rvq-replay", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_codec_rvq_replay.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_codec_rvq_replay = b.addInstallArtifact(codec_rvq_replay, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-codec-rvq-replay", "Build the isolated codec RVQ counterfactual; does not run a GPU").dependOn(&install_codec_rvq_replay.step);
    const codec_weight_replay = b.addExecutable(.{ .name = "tf-drama-codec-weight-replay", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_codec_weight_replay.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_codec_weight_replay = b.addInstallArtifact(codec_weight_replay, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-codec-weight-replay", "Build the isolated codec weight counterfactual; does not run a GPU").dependOn(&install_codec_weight_replay.step);
    const tower_oracle = b.addExecutable(.{ .name = "tf-drama-tower-oracle", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_oracle.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_oracle = b.addInstallArtifact(tower_oracle, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-oracle", "Build the production-tower numerical oracle driver; does not run a GPU").dependOn(&install_tower_oracle.step);
    const semantic_fixture = b.addExecutable(.{ .name = "tf-drama-semantic-fixture", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_semantic_fixture.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_semantic_fixture = b.addInstallArtifact(semantic_fixture, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-semantic-fixture", "Build the production generator fixture capture; does not run a GPU").dependOn(&install_semantic_fixture.step);
    const backbone_oracle = b.addExecutable(.{ .name = "tf-drama-backbone-oracle", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_backbone_oracle.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_backbone_oracle = b.addInstallArtifact(backbone_oracle, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-backbone-oracle", "Build the production-backbone numerical oracle driver; does not run a GPU").dependOn(&install_backbone_oracle.step);
    const gdn_oracle = b.addExecutable(.{ .name = "tf-drama-gdn-canonical-oracle", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_gdn_canonical_oracle.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_gdn_oracle = b.addInstallArtifact(gdn_oracle, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-gdn-canonical-oracle", "Build the GDN canonical controller verification driver; does not run a GPU").dependOn(&install_gdn_oracle.step);
    const codec_trace = b.addExecutable(.{ .name = "tf-drama-codec-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_codec_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_trace = b.addInstallArtifact(codec_trace, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-codec-trace", "Build the production-codec trace/replay diagnostic; does not run a GPU").dependOn(&install_trace.step);
    const backbone_trace = b.addExecutable(.{ .name = "tf-drama-backbone-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_backbone_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_backbone_trace = b.addInstallArtifact(backbone_trace, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-backbone-trace", "Build the production-backbone stage trace diagnostic; does not run a GPU").dependOn(&install_backbone_trace.step);
    const backbone_replay = b.addExecutable(.{ .name = "tf-drama-backbone-replay", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_backbone_replay.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_backbone_replay = b.addInstallArtifact(backbone_replay, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-backbone-replay", "Build the production-backbone operator replay diagnostic; does not run a GPU").dependOn(&install_backbone_replay.step);
    const gdn_layer0 = b.addExecutable(.{ .name = "tf-drama-gdn-layer0-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_gdn_layer0_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_gdn_layer0 = b.addInstallArtifact(gdn_layer0, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-gdn-layer0-trace", "Build the layer-zero production attention scratch diagnostic; does not run a GPU").dependOn(&install_gdn_layer0.step);
    const tower_mixer = b.addExecutable(.{ .name = "tf-drama-tower-mixer-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_mixer_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_mixer = b.addInstallArtifact(tower_mixer, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-mixer-trace", "Build the full-context tower mixer trace diagnostic; does not run a GPU").dependOn(&install_tower_mixer.step);
    const tower_qk = b.addExecutable(.{ .name = "tf-drama-tower-qk-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_qk_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_qk = b.addInstallArtifact(tower_qk, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-qk-trace", "Build the full-context tower Q/K and rotary diagnostic; does not run a GPU").dependOn(&install_tower_qk.step);
    const tower_head_replay = b.addExecutable(.{ .name = "tf-drama-tower-head-replay", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_head_replay.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_head_replay = b.addInstallArtifact(tower_head_replay, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-head-replay", "Build the production-tower head9 identical-input replay diagnostic; does not run a GPU").dependOn(&install_tower_head_replay.step);
    const tower_trace = b.addExecutable(.{ .name = "tf-drama-tower-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_trace = b.addInstallArtifact(tower_trace, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-trace", "Build the production-tower real-input stage trace diagnostic; does not run a GPU").dependOn(&install_tower_trace.step);
    const tower_context = b.addExecutable(.{ .name = "tf-drama-tower-context-oracle", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_context_oracle.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_context = b.addInstallArtifact(tower_context, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-context-oracle", "Build the full-context teacher-forced tower diagnostic; does not run a GPU").dependOn(&install_tower_context.step);
    const tower_context_trace = b.addExecutable(.{ .name = "tf-drama-tower-context-trace", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_context_trace.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "drama", .module = mods.drama } },
    }) });
    const install_tower_context_trace = b.addInstallArtifact(tower_context_trace, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-context-trace", "Build the full-context tower stage trace diagnostic; does not run a GPU").dependOn(&install_tower_context_trace.step);
    const tower_linear_replay = b.addExecutable(.{ .name = "tf-drama-tower-linear-replay", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/drama_tower_linear_replay.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    }) });
    const install_tower_linear_replay = b.addInstallArtifact(tower_linear_replay, .{ .dest_dir = .{ .override = .{ .custom = "oracle/bin" } } });
    b.step("drama-tower-linear-replay", "Build the tower eh_proj cuBLASLt identical-input replay diagnostic; does not run a GPU").dependOn(&install_tower_linear_replay.step);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, drama: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "drama", .module = drama } },
    });
    return .{ .api = api, .engines = mod };
}

/// The checkpoint subcommands' module: `models`, `info` and `pull` over the CUDA families.
fn checkpointCli(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, engines_mod: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "native_engines", .module = engines_mod }, exlr8_build.cliImport(b, target, optimize, core) } });
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, drama: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, drama);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = checkpointCli(b, target, .ReleaseSafe, null, m.engines, core) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, all: *std.Build.Step, server_steps: bool) void {
    const step = b.step("test-cuda-host", "The CUDA side's host unit tests alone (no GPU work)");
    all.dependOn(step);
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = host }) })).step);
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const native_modules = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, mods.drama);
    const native = native_modules.engines;
    // the HTTP server's unit tests (zig/src/server/root.zig), which the macOS build runs, on Linux hosts too
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } },
    });
    const server_test = b.addRunArtifact(b.addTest(.{ .root_module = server_tests }));
    step.dependOn(&server_test.step);
    b.step("test-server-cpu", "The HTTP server's unit tests (routes, templates, tool parsing) without a GPU").dependOn(&server_test.step);
    // the server goldens and Cortheon's fuzz step here when the target is not macOS, whose Metal build defines them
    if (server_steps) {
        golden_build.step(b, host, server_tests, native_modules.api, mods.tokenizer, template);
        @import("cortheon_fuzz.zig").add(b, host, native_modules.api, mods.tokenizer, template);
    }
    // the tokenizer's own tests (memo, steps, regex, unicode) run here: no other test binary reaches them
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.drama, mods.heat, mods.tokenizer, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, host, .debug, false, native_modules.engines, mods.core));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.core, mods.lanes, mods.nemotron, mods.drama, mods.tokenizer, build_options, false);
}

/// SASS targets every release fatbin holds: Ampere (sm_86), RTX 50 and RTX PRO Blackwell (sm_120), GB10 (sm_121).
pub const release_sms = "86,120,121";

/// Validate the complete named input set, including images unused by today's server, and each file's release targets.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const tool = b.addExecutable(.{ .name = "check-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = b.graph.host }) });
    const check = b.addRunArtifact(tool);
    check.addArg(release_sms);
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
