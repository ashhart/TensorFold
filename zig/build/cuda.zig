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
    .{ .name = "flashnext_gdn", .flags = &.{ "-O3", "--fmad=false" } }, // qwen4_exp/cuda/gdn.cu chain, one row
    .{ .name = "flashnext_batch", .flags = &.{ "-O3", "--fmad=false" } }, // ours: Flash Next's batched prompt pass glue
    .{ .name = "flashnext_mtp", .flags = &.{"-O3"} }, // ours: Flash Next's MTP drafts, a row's best score and log-sum-exp
    .{ .name = "qmm", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py lane matmul, groups of 32
    .{ .name = "experts_rows", .flags = &.{"-O3"} }, // experts.cu's decode sums over a window or prompt chunk's rows
    .{ .name = "route_rows", .flags = glue }, // ours: router logits and softmax top-k rows (cuda/route_rows.zig)
    .{ .name = "flashnext_hc", .flags = glue }, // ours: Flash Next's hyper-connection and n-gram glue
    .{ .name = "flashnext_attn", .flags = glue }, // ours: Flash Next's attention glue and its indexer's pooling and scores
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
    .{ .name = "affine8", .flags = &.{"-O3"} }, // ours: MLX affine 2 to 8 bits on lane_gemv tiles of their own width
    .{ .name = "affine_prefill", .flags = &.{"-O3"} }, // ours: qmm_prefill's prompt GEMM over affine8's tiles
    .{ .name = "fp8_experts", .flags = &.{"-O3"} }, // fp8/experts.cu's device code, tensorfold_fp8_experts_v6
    .{ .name = "exlr8", .flags = &.{ "-O3", "--fmad=false" } }, // ours: EXLR8 lane projections, the Metal kernels' order
    .{ .name = "exlr8_dense", .flags = &.{ "-O3", "--fmad=false" } }, // ours: EXLR8 dense matrices by row, the Metal order
    .{ .name = "nvfp4_experts", .flags = &.{"-O3"} }, // nvfp4/experts.cu's device code, the NVFP4 expert kernels
    .{ .name = "rank_sum", .flags = &.{"-O3"} }, // ours: the rank-order sum of a reduce-scatter (rank_scatter.zig)
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "train", .flags = &.{"-O3"} }, // ours: Sliding Weights' learning step (train.metal's kernels)
    .{ .name = "qwen35", .flags = &.{"-O3"} }, // ours: the Qwen3.5 hybrids' glue, the Metal kernels' rounding points
    .{ .name = "qwen35_draft", .flags = &.{"-O3"} }, // ours: the DFlash2 drafter's glue (draft_ops.metal's rounding)
    .{ .name = "qwen35_keep", .flags = &.{"-O3"} }, // ours: a verified window's kept path applied to its stream's state
    .{ .name = "qwen_vision", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the Qwen3.5 vision tower's glue and attention
    .{ .name = "qwen35_kv8", .flags = &.{"-O3"} }, // ours: --kv-cache fp8's e4m3 attention rows for the Qwen3.5 hybrids
    .{ .name = "nemotron_norms", .flags = glue }, // ours, each with a host reference (glue_ref.zig): Nemotron's glue
    .{ .name = "nemotron_route", .flags = glue },
    .{ .name = "nemotron_mamba", .flags = glue },
    .{ .name = "nemotron_attention", .flags = glue },
    .{ .name = "nemotron_keyed", .flags = glue },
    .{ .name = "nemotron_latent", .flags = glue }, // ours: a latent MoE's wide picks, share sums and shared add
    .{ .name = "ck_lane", .flags = &.{"-O3"} }, // nvfp4/lane4.cu's A8: FP8 rows times FP8 weights (checkpoint math)
    .{ .name = "ck_experts", .flags = &.{"-O3"}, .arch_specific = true }, // ours: NVFP4 x NVFP4 experts (sm_121a)
    .{ .name = "norms", .flags = &.{"-O3"} }, // ours: RMS norm rows for every family (cuda/norms.zig)
    .{ .name = "embed_rows", .flags = &.{"-O3"} }, // ours: 4-bit affine and bf16 token rows (cuda/embed_rows.zig)
    .{ .name = "lfm2", .flags = glue }, // ours: LFM2's short convolution, QK norm and rotary, gates and label rows
    .{ .name = "vit", .flags = glue }, // ours: vision towers' LayerNorm, Linear epilogues, activations, unshuffle
    .{ .name = "tree_round", .flags = &.{"-O3"} }, // ours: verified tree rounds for every family (cuda/tree_round.zig)
    .{ .name = "act_rows", .flags = &.{"-O3"} }, // ours: SwiGLU, output gates, group sums (cuda/act_rows.zig)
    .{ .name = "gdn_prep", .flags = &.{"-O3"} }, // ours: Gated DeltaNet conv, norms, g and beta (cuda/gdn_prep.zig)
    .{ .name = "qk_rows", .flags = &.{"-O3"} }, // ours: QK norm, RoPE and the cache write (cuda/qk_rows.zig)
    .{ .name = "qmmf", .flags = &.{"-O3"} }, // nvfp4/qmmf.cu's device code (FP8G, FP4), tensorfold_nvfp4_v3
    .{ .name = "prompt16", .flags = &.{"-O3"} }, // nvfp4/prompt.cu's FP4 and FP8 tile 4, tensorfold_nvfp4_prompt_v1
    .{ .name = "tree_attention", .flags = &.{"-O3"} }, // ours: committed keys and paths (cuda/tree_attention.zig)
    .{ .name = "block_attention", .flags = &.{"-O3"} }, // ours: a drafter's block over its context (cuda/block_attention.zig)
    .{ .name = "window_attention", .flags = &.{"-O3"} }, // ours: sliding-window attention (cuda/window_attention.zig)
    .{ .name = "moe_glue", .flags = &.{"-O3"} }, // ours: sigmoid top-k and the weighted combine (cuda/moe_glue.zig)
    .{ .name = "dense_rows", .flags = &.{"-O3"} }, // ours: bf16 rows times weight rows into fp32 (cuda/dense_rows.zig)
    .{ .name = "modelopt_pack", .flags = &.{"-O3"} }, // ours: ModelOpt tensors into the lane, prompt and expert layouts
    .{ .name = "glm_kda", .flags = &.{ "-O3", "--fmad=false" } }, // GLM: the Python family's glm5_next/cuda/kda.cu
    .{ .name = "glm_glue", .flags = &.{"-O3"} }, // ours: the Python family's Triton glue by hand, its PTX op for op
    .{ .name = "glm_ops", .flags = &.{"-O3"} }, // ours: GLM's integer work the Python family did in torch (the DSA token lists)
    .{ .name = "glm_quant", .flags = torch_ops }, // ours: GLM's draft-only weight quantizers, torch at load in the Python family
    .{ .name = "exl3_linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // exl3/linear.cu's device code, tensorfold_exl3_linear_v5
    .{ .name = "exl3_experts", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // exl3/experts.cu's device code, tensorfold_exl3_experts
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    .{ .name = "roce_gather", .src = "roce_gather", .flags = &.{"-O3"} }, // b12x's one-shot RoCE all-gather, two ranks
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

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module, tokenizer: *std.Build.Module, heat: *std.Build.Module, radio: *std.Build.Module, living: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = optimize, .link_libc = true });
    // Living Weights' family-neutral learner: a family's lessons, and the server's view of them, import it
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("heat", heat);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const radio = radioModule(b, target, optimize, cuda, core, lanes);
    nemotron.addImport("radio", radio);
    nemotron.addImport("living", living);
    const qwen35 = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen35.addImport("cuda", cuda);
    qwen35.addImport("core", core);
    qwen35.addImport("lanes", lanes);
    const qwen_vision = qwenVisionModule(b, target, optimize, cuda, core, lanes);
    qwen35.addImport("qwen_vision", qwen_vision);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .qwen35 = qwen35, .qwen_vision = qwen_vision, .tokenizer = tokenizer, .heat = heat, .radio = radio, .living = living };
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
    const native = engines(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, target, optimize, strip, native.engines, mods.core));
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    runner.addImport("cuda", cuda);
    runner.addImport("core", mods.core);
    runner.addImport("lanes", mods.lanes);
    runner.addImport("nemotron", mods.nemotron);
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
    const runner_exe = b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner });
    b.installArtifact(runner_exe);
    // tf-tp-watch-check: the ranks' lines with no GPU or NCCL; a failing or dying rank stops every rank
    b.installArtifact(b.addExecutable(.{ .name = "tf-tp-watch-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/tp_watch_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "cuda", .module = cuda }} }) }));
    flashnextChecks(b, target, optimize, cuda, mods.core, mods.lanes, mods.qwen_vision);
    const controls = if (nvcc) |tool| fatbin(b, tool, version.?, .{ .name = "contract_controls", .flags = &.{"-O3"} }, sms) else null;
    @import("kernel_contract.zig").cuda(b, runner_exe, controls);
    _ = nativeServer(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, true, strip);
    // tf-qwen35-cuda: the Qwen CUDA forward's rows for a reference to compare, and its tower on picture files
    const q35 = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen35_cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    q35.addImport("cuda", cuda);
    q35.addImport("core", mods.core);
    q35.addImport("qwen35", mods.qwen35);
    b.installArtifact(b.addExecutable(.{ .name = "tf-qwen35-cuda", .root_module = q35 }));
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
            .{ .name = "lfm2", .module = lfm2Module(b, target, optimize, cuda, core) },
            .{ .name = "core", .module = core },
            .{ .name = "flashnext", .module = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision) },
            .{ .name = "glm_cuda", .module = glmCuda(b, target, optimize, cuda, core, lanes, heat) },
            .{ .name = "qwen35", .module = qwen35 },
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
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "living", .module = living }, .{ .name = "checkpoint_cli", .module = checkpointCli(b, target, .ReleaseSafe, null, m.engines, core) } },
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

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, all: *std.Build.Step, server_steps: bool) void {
    const step = b.step("test-cuda-host", "The CUDA side's host unit tests alone (no GPU work)");
    all.dependOn(step);
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = host }) })).step);
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const exl3_host = b.step("test-exl3-cuda-host", "EXL3 loader and CUDA host regressions without a GPU");
    for ([_]*std.Build.Module{ cuda, mods.core }) |m| {
        exl3_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
    const contract_arch = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_arch_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract_arch })).step);
    const writeback_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_flashnext_writeback.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) }));
    step.dependOn(&writeback_contract.step);
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
    const native_modules = engines(b, host, .debug, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    const native = native_modules.engines;
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
    const server_test = b.addRunArtifact(b.addTest(.{ .root_module = server_tests }));
    const json_test = b.addRunArtifact(b.addTest(.{ .root_module = server_json }));
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
        golden_build.step(b, host, server_tests, native_modules.api, mods.tokenizer, template);
        @import("cortheon_fuzz.zig").add(b, host, native_modules.api, mods.tokenizer, template);
    }
    // the tokenizer's own tests (memo, steps, regex, unicode) run here: no other test binary reaches them
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.qwen35, mods.qwen_vision, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), lfm2Module(b, host, .debug, cuda, mods.core), mods.radio, mods.heat, mods.tokenizer, native_modules.api, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    // the GLM-5.3-Flash CUDA family's host side
    const glm = glmCuda(b, host, .debug, cuda, mods.core, mods.lanes, mods.heat);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    // the shares a GLM-5.3-Flash rank takes (cluster/canon.zig against the Python family's)
    const shares = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon_glm_test.zig"), .target = host, .optimize = .debug });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = shares })).step);
    // the tool-call parser's tests, run beside the server's on Macs: the GLM calls the CUDA port serves
    const parse_tests = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } } });
    _ = @import("server.zig").addTo(b, parse_tests, host, .debug);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = parse_tests })).step);
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
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    // with cl.exe as host compiler, CUDA 13's headers need MSVC's conforming preprocessor, and M_SQRT2 its math define
    if (b.graph.host.result.os.tag == .windows) run.addArgs(&.{ "-Xcompiler=/Zc:preprocessor", "-D_USE_MATH_DEFINES" });
    run.addArgs(k.flags);
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| {
        // arch-specific features (the block-scaled FP4 mma) from sm_120 on; older targets build the trapping fallback
        const a = if (k.arch_specific and (std.fmt.parseInt(u32, sm, 10) catch 0) >= 120) "a" else "";
        run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    }
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
    return nativeServer(b, target, .fast, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, false, null);
}

/// SASS targets every release fatbin holds: Ampere (sm_80, sm_86), RTX 50/PRO Blackwell (sm_120), GB10 (sm_121).
pub const release_sms = "80,86,120,121";

/// Validate the complete named input set, including images unused by today's server, and each file's release targets.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const tool = b.addExecutable(.{ .name = "check-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = b.graph.host }) });
    const check = b.addRunArtifact(tool);
    check.addArg(release_sms);
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}

/// Explicit model diagnostics; these tools do not qualify a checkpoint merely by compiling.
fn flashnextChecks(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module) void {
    const flashnext = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision);
    // not installed: one embedding row through the captured _embed cubin
    const embed_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_embed_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    embed_check.addImport("cuda", cuda);
    embed_check.addImport("core", core);
    embed_check.addImport("flashnext", flashnext);
    const embed_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-embed", .root_module = embed_check }));
    const embed_model = b.option([]const u8, "flashnext_model", "Flash Next checkpoint directory for the embedding check");
    const embed_kernels = b.option([]const u8, "flashnext_kernels", "captured Triton pack for the embedding check");
    const embed_token = b.option([]const u8, "flashnext_token", "token id for the embedding check") orelse "7";
    if (embed_model != null and embed_kernels != null) embed_run.addArgs(&.{ embed_model.?, embed_kernels.?, embed_token });
    b.step("flashnext-embed", "Upload the Flash Next embedding table and run _embed for one token").dependOn(&embed_run.step);
    const prompt_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_prompt_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    prompt_check.addImport("cuda", cuda);
    prompt_check.addImport("core", core);
    prompt_check.addImport("flashnext", flashnext);
    const prompt_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-prompt", .root_module = prompt_check }));
    const prompt_ids = b.option([]const u8, "flashnext_ids", "file of prompt token ids for the Flash Next prompt check");
    const prompt_count = b.option([]const u8, "flashnext_count", "greedy tokens to emit") orelse "8";
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
    if (b.option([]const u8, "flashnext_fixtures", "capture_sparse.sh's selection fixtures folder")) |dir| select_run.addArgs(&.{dir});
    b.step("flashnext-select", "fn_qsa_select against Triton's _select / _select_tiles lists on captured score rows").dependOn(&select_run.step);
}
