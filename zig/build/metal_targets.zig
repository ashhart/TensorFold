//! Metal build targets and their shared source/library/module construction.
const std = @import("std");
const builtin = @import("builtin");
const qwen27_build = @import("qwen27.zig");
const drama3_build = @import("drama3_metal.zig");
const core_checks = @import("core_checks.zig");
const glm_checks = @import("glm_checks.zig");
const flashnext_checks = @import("flashnext_checks.zig");
const counters_build = @import("metal_counters.zig");
const exl3_build = @import("exl3.zig");
const exlr8_build = @import("exlr8.zig");
const metal_server = @import("metal_server.zig");
const contract_build = @import("kernel_contract.zig");
const image_programs = @import("metal_image_programs.zig");
const metal_lists = @import("metal_source_lists.zig");
const kernels = metal_lists.kernels;
const generated = metal_lists.generated;
const kimi_dir = metal_lists.kimi_dir;
const kimi_kernels = metal_lists.kimi_kernels;
const nemotron_packed = metal_lists.nemotron_packed;

pub fn mlxMetalStd(b: *std.Build) []const u8 {
    const os = b.graph.host.result.os;
    const major = if (os.tag == .macos) os.version_range.semver.min.major else 0;
    if (major >= 27) return "metal4.1";
    if (major >= 26) return "metal4.0";
    if (major >= 15) return "metal3.2";
    return "metal3.1";
}

/// Whether this Mac has the offline Metal compiler (Xcode's toolchain, not the Command Line Tools alone).
fn hasMetalCompiler(b: *std.Build) bool {
    b.graph.poisonCache(); // installing Xcode changes the answer
    if (b.runFallible(&.{ "xcrun", "-sdk", "macosx", "-f", "metal" }, .{ .stderr_behavior = .ignore }) == .success) return true;
    std.log.warn("skipping offline Metal shader checks: xcrun finds no metal compiler (install Xcode)", .{});
    return false;
}

pub fn metallib(b: *std.Build, name: []const u8, sources: []const []const u8, flags: []const []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg(b.fmt("{s}.metallib", .{name}));
    for (sources) |src| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal" });
        cc.addArgs(flags);
        cc.addArg("-c");
        cc.addFileArg(b.path(src));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("{s}.air", .{std.fs.path.stem(src)})));
    }
    return out;
}

pub fn kimiMetallib(b: *std.Build, flags: []const []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg("kimi.metallib");
    for (kimi_kernels) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal" });
        cc.addArgs(flags);
        cc.addPrefixedDirectoryArg("-I", b.path(kimi_dir));
        cc.addFileInput(b.path(kimi_dir ++ "/kimi_common.h"));
        cc.addArg("-c");
        cc.addFileArg(b.path(b.fmt("{s}/{s}", .{ kimi_dir, name })));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("kimi_{s}.air", .{std.fs.path.stem(name)})));
    }
    return out;
}

pub fn embedded(b: *std.Build, lib: std.Build.LazyPath, file: []const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(lib, file);
    const src = wf.add("metallib.zig", b.fmt("pub const bytes = @embedFile(\"{s}\");\n", .{file}));
    return b.createModule(.{ .root_source_file = src });
}

pub fn metalTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: *std.Build.Step) void {
    const metal_std = b.option([]const u8, "metal-std", "Metal language version (default: MLX's for this macOS)") orelse mlxMetalStd(b);
    // Without the offline compiler the tests embed empty metallibs; the programs that load them stop instead.
    const shaders = hasMetalCompiler(b);
    const no_shaders = b.addWriteFiles().add("no-shaders.metallib", "");
    const no_compiler: ?*std.Build.Step = if (shaders) null else &b.addFail("this program embeds compiled Metal shaders: install Xcode").step;

    // MLX's mx.fast.metal_kernel JIT: MTLCompileOptions mathMode safe, fp32 functions fast (the default), its...
    const mlx_flags = [_][]const u8{ b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast" };
    const lib = metallib(b, "tensorfold", &kernels, &mlx_flags);
    const lib_install = b.addInstallFile(lib, "lib/tensorfold.metallib");
    b.step("metallib", "Compile zig/kernels/metal into tensorfold.metallib").dependOn(&lib_install.step);

    // Control: the same Nemotron kernels under fast math, which the bit-exact check must be able to tell apart.
    const fast_lib = metallib(b, "nemotron_fastmath", generated, &[_][]const u8{ b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=fast", "-fmetal-math-fp32-functions=fast" });
    b.getInstallStep().dependOn(&b.addInstallFile(fast_lib, "lib/nemotron_fastmath.metallib").step);
    b.getInstallStep().dependOn(&lib_install.step);

    const mods = metalEngineModules(b, target, optimize, draft_ids);
    const metal = mods.metal;
    const kernels_mod = embedded(b, if (shaders) lib else no_shaders, "tensorfold.metallib");
    counters_build.targets(b, target, optimize, metal, mods.sources, test_step);

    // The engines registry: the families this platform's binaries serve; the CLI reads it for models/info/pull.
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    mods.engine.addImport("engine_api", api);
    const engines = b.createModule(.{
        .root_source_file = b.path("zig/src/native/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = mods.engine } },
    });
    engines.addImport("living", mods.living);
    engines.addImport("kernel_sources", mods.sources);
    inline for ([_][]const u8{ "inkling2_metal", "inkling2_metal_ref", "inkling2_glue_plan", "inkling2_loop_plan", "inkling2_tree_plan" }) |name|
        engines.addImport(name, mods.engine.import_table.get(name).?);
    const drama3_lib = drama3_build.metallib(b, metal_std);
    b.getInstallStep().dependOn(&b.addInstallFile(drama3_lib, "lib/drama3.metallib").step);
    const drama3_embed = if (shaders) drama3_lib else no_shaders;
    drama3_build.addNative(b, target, optimize, metal, engines, drama3_embed);
    drama3_build.targets(b, target, optimize, metal, drama3_embed, test_step, no_compiler);

    const programs = [_]struct { name: []const u8, path: []const u8, about: []const u8 }{
        .{ .name = "tf-metal-tests", .path = "zig/tests/gpu_tests.zig", .about = "GPU runtime tests" },
        .{ .name = "tf-dispatch-bench", .path = "zig/tests/dispatch_bench.zig", .about = "Dispatch boundary benchmark" },
        .{ .name = "tf-kernel-exact", .path = "zig/tests/kernel_exact.zig", .about = "Bit-exact check of a kernel against an MLX oracle case" },
    };
    for (programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "metallib", .module = kernels_mod } },
        });
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        const step = b.step(p.name, p.about);
        step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        if (no_compiler) |fail| step.dependOn(fail);
    }

    // Kimi K3: the family module, its metallib, and its GPU check and benchmark programs.
    const kimi_lib = kimiMetallib(b, &mlx_flags);
    b.getInstallStep().dependOn(&b.addInstallFile(kimi_lib, "lib/kimi.metallib").step);
    const kimi = kimiModule(b, target, optimize, metal, mods.engine, mods.lanes);
    const kimi_bytes = embedded(b, if (shaders) kimi_lib else no_shaders, "kimi.metallib");
    engines.addImport("kimi_k3", kimi);
    engines.addImport("kimi_metallib", kimi_bytes);
    const kimi_programs = [_]struct { name: []const u8, path: []const u8, about: []const u8 }{
        .{ .name = "tf-k3-check", .path = "zig/tests/kimi_k3/check.zig", .about = "Kimi K3 kernels and layers on synthetic weights at real shapes" },
        .{ .name = "tf-k3-serve-check", .path = "zig/tests/kimi_k3/serve.zig", .about = "Kimi native lane adapter on a cut-down synthetic Metal graph" },
        .{ .name = "tf-k3-bench", .path = "zig/tests/kimi_k3/bench.zig", .about = "Kimi K3 kernel bandwidth" },
    };
    for (kimi_programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kimi_k3", .module = kimi }, .{ .name = "kimi_metallib", .module = kimi_bytes }, .{ .name = "lanes", .module = mods.lanes } },
        });
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        const step = b.step(p.name, p.about);
        step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        if (no_compiler) |fail| step.dependOn(fail);
    }

    // Kimi's tiktoken tokenizer and its parity program (tools/zig/k3_tokenizer_parity.py).
    const tiktoken = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tiktoken.zig"), .target = target, .optimize = optimize });
    const tok_mod = b.createModule(.{
        .root_source_file = b.path("zig/tests/kimi_k3/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "tiktoken", .module = tiktoken }},
    });
    const tok_exe = b.addExecutable(.{ .name = "tf-k3-tokenizer", .root_module = tok_mod });
    b.installArtifact(tok_exe);
    b.step("tf-k3-tokenizer", "Kimi K3's tiktoken tokenizer over parity cases").dependOn(&b.addInstallArtifact(tok_exe, .{}).step);

    // The cluster layer: inventory, membership, planner, loading and converged rounds over the lane core.
    const lanes = mods.lanes;
    const fabric = mods.fabric;
    const cluster = b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/cluster.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "core", .module = mods.engine }, .{ .name = "canon", .module = @import("cuda_engines.zig").canonModule(b, target, optimize) } },
    });
    // The engine: kernel sources compiled at run time (MLX's options), the families, the lane core.
    const sources = mods.sources;
    // Nemotron's packed kernels, compiled offline and embedded: the runtime fallback when this macOS's compiler...
    const packed_lib = metallib(b, "nemotron_packed", &nemotron_packed, &mlx_flags);
    b.getInstallStep().dependOn(&b.addInstallFile(packed_lib, "lib/nemotron_packed.metallib").step);
    sources.addImport("nemotron_packed_metallib", embedded(b, if (shaders) packed_lib else no_shaders, "nemotron_packed.metallib"));
    const engine = mods.engine;
    // FlashNext's CPU contracts, with the family module's imports so the replay graph type-checks
    const flash_host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_host.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "nemotron_draft_ids", .module = draft_ids }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric } },
    }) });
    b.step("compile-flashnext-host", "Compile FlashNext CPU metadata contracts without running them").dependOn(&flash_host.step);
    const flash_host_run = b.addRunArtifact(flash_host);
    b.step("test-flashnext-host", "Run FlashNext CPU metadata contracts without Metal").dependOn(&flash_host_run.step);
    test_step.dependOn(&flash_host_run.step);
    // Pack builder cases that do not need the replay graph.
    const pack_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_pack_cases.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    test_step.dependOn(&b.addRunArtifact(pack_cases).step);
    const pack_cache_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_pack_cache_cases.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    const pack_cache_run = b.addRunArtifact(pack_cache_cases);
    b.step("test-flashnext-pack-cache", "CPU-only atomic Flash Next pack cache lifecycle").dependOn(&pack_cache_run.step);
    test_step.dependOn(&pack_cache_run.step);
    const engine_programs = [_]image_programs.Program{
        .{ .name = "tf-qwen35-check", .path = "zig/tests/qwen35_check.zig", .about = "Qwen3.5-2B checkpoint and native operations" },
        .{ .name = "tf-qwen35-forward", .path = "zig/tests/qwen35_forward.zig", .about = "Qwen3.5-2B native teacher-forced logits" },
        .{ .name = "tf-qwen35-exact", .path = "zig/tests/qwen35_exact.zig", .about = "Qwen3.5-2B forward and committed-state exactness" },
        .{ .name = "tf-lfm3-forward", .path = "zig/tests/lfm3_forward.zig", .about = "LFM3 native teacher-forced BF16 logits" },
        .{ .name = "tf-lfm3-vision", .path = "zig/tests/lfm3_vision.zig", .about = "LFM3-VL native tower and projector outputs" },
        .{ .name = "tf-lfm3-exact", .path = "zig/tests/lfm3_exact.zig", .about = "LFM3 prompt, window, snapshot and concurrent cache equality" },
        .{ .name = "tf-lfm3-profile", .path = "zig/tests/lfm3_profile.zig", .about = "LFM3 calibrated GPU class costs with final-logit byte identity" },
        .{ .name = "tf-lfm3-dual-project", .path = "zig/tests/lfm3_dual_project.zig", .about = "Actual checkpoint paired projection byte comparison" },
        .{ .name = "tf-lfm3-draw-tune", .path = "zig/tests/lfm3_draw_tune.zig", .about = "Diagnostic exact CPU greedy reduction timing" },
        .{ .name = "tf-lfm3-row-tune", .path = "zig/tests/lfm3_row_tune.zig", .about = "Diagnostic byte-preserving single-row projection launch sweep" },
        .{ .name = "tf-lfm3-project", .path = "zig/tests/lfm3_project.zig", .about = "LFM3 production projection outputs on saved BF16 input rows" },
        .{ .name = "tf-lfm2-forward", .path = "zig/tests/lfm2_forward.zig", .about = "LFM2-MoE native teacher-forced logits" },
        .{ .name = "tf-lfm2-bench", .path = "zig/tests/lfm2_bench.zig", .about = "LFM2-MoE prompt-chunk and one-row decode timing" },
        .{ .name = "tf-lfm2-exact", .path = "zig/tests/lfm2_exact.zig", .about = "LFM2-MoE prompt chunks, windows and committed-state exactness" },
        .{ .name = "tf-lfm2-fp64", .path = "zig/tests/lfm2_fp64.zig", .about = "LFM2 projection classes and attention on the tensor units, the tiles and the twin, each against float64" },
        .{ .name = "tensorfold", .path = "zig/src/main.zig", .about = "The native engine's command line" },
        .{ .name = "tf-nemotron-fixtures", .path = "zig/tests/nemotron_fixtures.zig", .about = "Nemotron kernels against the Python engine's captured ops" },
        .{ .name = "tf-nemotron-bench", .path = "zig/tests/nemotron_bench.zig", .about = "One-row projection kernels timed by tile count" },
        .{ .name = "tf-nemotron-wide", .path = "zig/tests/nemotron_wide.zig", .about = "Window costs by kernel class and expert overlap of drafted, sibling and stream rows" },
        .{ .name = "tf-nemotron-prefill-check", .path = "zig/tests/nemotron_prefill_check.zig", .about = "A prompt chunk layer by layer against prefill_dump.py's rows" },
        .{ .name = "tf-nemotron-dense", .path = "zig/tests/nemotron_dense.zig", .about = "Dense projection schedules bit-checked against the engine's kernels, then timed" },
        .{ .name = "tf-nemotron-experts", .path = "zig/tests/nemotron_experts.zig", .about = "Routed-expert pass shapes bit-checked against the engine's kernels, then timed by window width" },
        .{ .name = "tf-nemotron-sample-check", .path = "zig/tests/nemotron_sample_check.zig", .about = "tf_sample_full against its host reference on synthetic rows, timed beside tf_gpu_sample" },
        .{ .name = "tf-train-check", .path = "zig/tests/train_check.zig", .about = "The learning step's gradients against the engine's own forward by finite differences, on the real model" },
        .{ .name = "tf-train-pack", .path = "zig/tests/train_pack.zig", .about = "Training passes holding several sequences against one a pass, on the real model: differences and time" },
        .{ .name = "tf-pocket-fit-check", .path = "zig/tests/pocket_fit_check.zig", .about = "A dumped pocket lesson's read rows fitted as the learner fits them, timed, each fit's numbers and weights written" },
        .{ .name = "tf-flashnext-run", .path = "zig/tests/flashnext_run.zig", .about = "Flash Next one-row greedy steps on the Python engine's recorded kernels, against its tokens" },
        .{ .name = "tf-flashnext-pack", .path = "zig/tests/flashnext_pack.zig", .about = "Flash Next's three weight packs built from a checkpoint, byte-compared with a Python dump" },
        .{ .name = "tf-flashnext-think", .path = "zig/tests/flashnext_think.zig", .about = "Flash Next's verdict kernels under thinking budgets on a fake target, against the lane core's close" },
        .{ .name = "tf-grid-sync-bench", .path = "zig/tests/grid_sync_bench.zig", .about = "A GPU-wide barrier in one persistent dispatch against dependent relaunches" },
        .{ .name = "tf-segments-probe", .path = "zig/tests/segments_probe.zig", .about = "A two-segment chunk behind a held first queue: the GPU watchdog without the probe, a pass with it" },
        .{ .name = "tf-weight-read-check", .path = "zig/tests/weight_read_check.zig", .about = "Check native file reads and failed-read cleanup", .c_source = "zig/tests/pread_fault.c" },
        .{ .name = "tf-glm-run", .path = "zig/tests/glm_run.zig", .about = "GLM-5.3-Flash greedy replies at each draft depth against depth 0 and reference tokens" },
        .{ .name = "tf-glm-vision", .path = "zig/tests/glm_vision.zig", .about = "GLM-5.3-Flash's image tower on an image or a reference dump's patches, compared with the dump" },
        .{ .name = "tf-glm-video", .path = "zig/tests/glm_video.zig", .about = "GLM-5.3-Flash's video preparation on a clip (libtfvideo), written for tools/zig/glm_video_fixtures.py" },
        .{ .name = "tf-qwen-vision", .path = "zig/tests/qwen_vision.zig", .about = "The Qwen-VL tower on picture files: patches, grids and merged rows written for a float32 reference" },
    } ++ image_programs.rows ++ [_]image_programs.Program{
        .{ .name = "tf-glm-attn-probe", .path = "zig/tests/glm_attn_probe.zig", .about = "GLM-5.3-Flash's dense latent attention kernels on given inputs (scores, probabilities, outputs)" },
        .{ .name = "tf-moe-bench", .path = "zig/tests/moe_bench.zig", .about = "Routed-expert prompt kernels from .metal files on a skewed routing: chained speed, bits against the first" },
        .{ .name = "tf-nemotron-forward", .path = "zig/tests/nemotron_forward.zig", .about = "Nemotron (Lightning or Bolt) native teacher-forced logits" },
        .{ .name = "tf-mm-bench", .path = "zig/tests/mm_bench.zig", .about = "Prompt matmul kernels from .metal files: chained speed at one shape, bits against the first, error against fp64" },
        .{ .name = "tf-sq-bench", .path = "zig/tests/sq_bench.zig", .about = "Row-exact 4-bit prompt tiles before the M5: chained speed at one shape, every output's bits against the first" },
        .{ .name = "tf-gemma4-kernels", .path = "zig/tests/gemma4/kernels.zig", .about = "Gemma 4 dense decode kernels against the reference's dumped fixtures, bit for bit by window width" },
        .{ .name = "tf-gemma4-glue", .path = "zig/tests/gemma4/glue.zig", .about = "Gemma 4 dense glue kernels against the reference's dumped fixtures: embed, norm, q/k/v, attention, tails, softcap" },
        .{ .name = "tf-gemma4-run", .path = "zig/tests/gemma4/run.zig", .about = "Gemma 4 dense greedy decode, tokens and top-two margins for the oracle's check" },
        .{ .name = "tf-gemma4-lanes", .path = "zig/tests/gemma4/lanes.zig", .about = "Gemma 4 dense on the lane core: plain, copy-drafted and shared-round streams against the oracle's tokens" },
        .{ .name = "tf-gemma4-vision", .path = "zig/tests/gemma4/vision.zig", .about = "Gemma 4's image tower against gemma4_vision_fixtures.py's dumps: patches bit for bit, each stage's error beside torch bf16's" },
        .{ .name = "tf-gemma4-bench", .path = "zig/tests/gemma4/bench.zig", .about = "Gemma 4 dense step costs: projections alone by rows against whole forwards" },
        .{ .name = "tf-gemma4-no-tensor", .path = "zig/tests/gemma4/no_tensor.zig", .about = "Gemma 4's Metal kernels on a Mac without tensor units: what builds, and the row path's bits" },
        .{ .name = "tf-spark-run", .path = "zig/tests/spark25/run.zig", .about = "Spark2.5 on Metal: greedy tokens, every position's logits or a prompt's last logits, for the reference's check" },
        .{ .name = "tf-minimax-run", .path = "zig/tests/minimax_m3/run.zig", .about = "MiniMax-M3.1-Flash greedy tokens, every position's logits or a prompt's last logits, for the reference's check" },
        .{ .name = "tf-minimax-vision", .path = "zig/tests/minimax_m3/vision.zig", .about = "MiniMax-M3.1-Flash's image tower alone: an image's patches and text rows, for tools/vision_ref.py's check" },
        .{ .name = "tf-llama-run", .path = "zig/tests/llama/run.zig", .about = "Llama on Metal: greedy tokens, every position's logits or a prompt's last logits, for the reference's check" },
        .{ .name = "tf-flint-run", .path = "zig/tests/flint/run.zig", .about = "Flint greedy tokens, every position's logits or a prompt's last logits, for the reference's check" },
        .{ .name = "tf-ricursive-run", .path = "zig/tests/ricursive/run.zig", .about = "Ripple on Metal: greedy tokens, every position's logits, a prompt's last logits or forced steps, for the reference's check" },
        .{ .name = "tf-qwen36-row-seam", .path = "zig/tests/qwen36_row_seam.zig", .about = "Model-free fixture: the row projection's bias sums stay fp32 at every width" },
    };
    // the tokenizer for the programs that tokenize prompts themselves (the engine module has none)
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for (engine_programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "core", .module = engine }, .{ .name = "cluster", .module = cluster }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "native_engines", .module = engines }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, exlr8_build.cliImport(b, target, optimize, engine) },
        });
        if (std.mem.eql(u8, p.name, "tensorfold")) mod.addImport("glm_tfq3_check", engine);
        if (p.c_source) |file| {
            mod.addCSourceFile(.{ .file = b.path(file), .flags = &.{"-std=c11"} });
        }
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        b.step(p.name, p.about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    glm_checks.sampler(b, target, optimize, metal, lanes, sources);

    @import("h3_metal.zig").targets(b, target, optimize, metal, sources, test_step);
    // The core row projection (chips without tensor units) on synthetic matrices; runs on any Mac's GPU
    const row_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/row_projection.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const row_check = b.addExecutable(.{ .name = "tf-row-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/row_projection_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "row_projection", .module = row_mod } },
    }) });
    b.step("tf-row-check", "The core row projection on synthetic 4-bit matrices: widths bit-equal, a CPU reference, indexed experts").dependOn(&b.addInstallArtifact(row_check, .{}).step);
    core_checks.rowWidths(b, target, optimize, metal, row_mod);
    const row_tests = b.addRunArtifact(b.addTest(.{ .root_module = row_mod }));
    test_step.dependOn(&row_tests.step);
    b.step("test-row-projection-host", "Row projection shapes and the kernel chosen at each window width").dependOn(&row_tests.step);
    core_checks.tileQmm(b, target, optimize, metal, sources, test_step);
    core_checks.kvPages(b, target, optimize, metal, sources, test_step);
    exl3_build.targets(b, target, optimize, metal, test_step);
    exlr8_build.metalTargets(b, target, optimize, metal, mods.sources, test_step);
    exlr8_build.glmTargets(b, target, optimize, metal, mods.engine);

    // The checkpoint CLI: models, info and pull, against a synthetic cache and a fake hub, at ReleaseSafe as it ships.
    const cli_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/tests.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, optimize, mods.engine), .{ .name = "core", .module = mods.engine } },
    }) });
    cli_tests.root_module.addImport("glm_tfq3_check", mods.engine);
    const run_cli_tests = b.addRunArtifact(cli_tests);
    test_step.dependOn(&run_cli_tests.step);
    b.step("test-cli", "The checkpoint CLI's host tests").dependOn(&run_cli_tests.step);

    // The core lane, shared-context and Qwen3.8-27B programs and host tests.
    qwen27_build.targets(b, target, optimize, metal, sources, engine, lanes, mods.qwen27, test_step);
    core_checks.targets(b, target, optimize, metal, engine);
    contract_build.metal(b, target, optimize, metal, engine, sources, kimi_bytes, test_step);
    glm_checks.targets(b, target, optimize, metal, engine, sources, test_step);
    flashnext_checks.targets(b, target, optimize, metal, engine);

    _ = metal_server.nativeServer(b, target, optimize, metal, engine, mods.living, api, engines, build_options, test_step, no_compiler);

    const cluster_metal = b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/metal_sink.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "cluster", .module = cluster } },
    });
    const cluster_exe = b.addExecutable(.{ .name = "tf-cluster", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "cluster", .module = cluster }},
    }) });
    b.installArtifact(cluster_exe);
    b.step("tf-cluster", "tensorfold cluster / serve --cluster / node").dependOn(&b.addInstallArtifact(cluster_exe, .{}).step);
    // Kimi K3's checkpoint validator: the cluster's header reader cross-checked with the family's own specs.
    const k3_validate = b.addExecutable(.{ .name = "tf-k3-validate", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/kimi_k3/validate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "kimi_k3", .module = kimi }, .{ .name = "cluster", .module = cluster } },
    }) });
    b.installArtifact(k3_validate);
    b.step("tf-k3-validate", "Kimi K3's checkpoint: index, headers, coverage and every tensor against the family").dependOn(&b.addInstallArtifact(k3_validate, .{}).step);
    const k3_cluster = b.addExecutable(.{ .name = "tf-k3-cluster", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/kimi_k3/cluster.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kimi_k3", .module = kimi }, .{ .name = "kimi_metallib", .module = kimi_bytes }, .{ .name = "cluster", .module = cluster } },
    }) });
    b.installArtifact(k3_cluster);
    const k3_cluster_step = b.step("tf-k3-cluster", "Kimi K3 tensor- and expert-parallel ranks against one node, through the cluster's canon");
    k3_cluster_step.dependOn(&b.addInstallArtifact(k3_cluster, .{}).step);
    if (no_compiler) |fail| k3_cluster_step.dependOn(fail);
    const tp2 = b.addExecutable(.{ .name = "tf-tp2-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/tp2_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "fabric", .module = fabric } },
    }) });
    b.step("tf-tp2-bench", "TP=2's exchange as the GPU sees it, between two Macs over MCDMA").dependOn(&b.addInstallArtifact(tp2, .{}).step);
    const cluster_tests = b.step("test-cluster", "Cluster tests: fake nodes and fabric, K3 planning (TF_K3_DIR), Metal sink");
    cluster_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cluster })).step);
    cluster_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cluster_metal })).step);
    // canon.zig is its own module, so a cluster-rooted test does not run these.
    cluster_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = @import("cuda_engines.zig").canonModule(b, target, optimize) })).step);

    // These module tests include Metal device checks when the macOS host has a GPU.
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = metal })).step);
    const helpers = b.createModule(.{
        .root_source_file = b.path("zig/tests/common.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "metallib", .module = kernels_mod } },
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = helpers })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine })).step);
    const round_depth = b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"round_depth"} }));
    b.step("test-nemotron-round-depth-host", "Priced head depth, nonlanding levels and fixed-depth controls without devices").dependOn(&round_depth.step);
    const lfm2_tests = b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"lfm2_moe"} }));
    b.step("test-lfm2", "The LFM2-MoE family's tests: config, draws, state and its shared rounds on this Mac's GPU").dependOn(&lfm2_tests.step);
    const lfm2_dspark = b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"lfm2 dspark:"} }));
    b.step("test-lfm2-dspark", "LFM2's taps for a drafter: the tap kernel on this Mac's GPU").dependOn(&lfm2_dspark.step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes })).step);
    // the fake fabric maps windows with mmap (the Macs' RDMA, not Windows); verbs is its own module, with its own root
    if (target.result.os.tag != .windows) {
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fabric })).step);
        const verbs_tests = b.addRunArtifact(b.addTest(.{ .root_module = @import("tp.zig").verbsModule(b, target, optimize) }));
        b.step("test-verbs", "verbs.zig and verbs_abi.zig, outside the fabric module").dependOn(&verbs_tests.step);
        test_step.dependOn(&verbs_tests.step);
    }
    const kimi_test = b.step("test-k3", "Kimi K3's host-side unit tests (no GPU work)");
    for ([_]*std.Build.Module{ kimi, tiktoken }) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        kimi_test.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}

pub fn metalEngineModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module) struct { metal: *std.Build.Module, lanes: *std.Build.Module, fabric: *std.Build.Module, sources: *std.Build.Module, engine: *std.Build.Module, living: *std.Build.Module, qwen27: qwen27_build.Shared } {
    const metal = b.createModule(.{
        .root_source_file = b.path("zig/src/metal/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    if (target.result.os.tag == .macos) {
        metal.linkFramework("Metal", .{});
        metal.linkFramework("Foundation", .{});
        metal.linkFramework("IOSurface", .{}); // the Neural Engine's prompt share passes IOSurfaces
        metal.linkSystemLibrary("objc", .{});
    }
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = optimize, .link_libc = true });
    fabric.addImport("verbs", @import("tp.zig").verbsModule(b, target, optimize));
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const inkling2_metal = @import("inkling2_sources.zig").module(b, target, optimize, sources);
    const inkling2_metal_ref = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/ref.zig"), .target = target, .optimize = optimize });
    inkling2_metal_ref.addImport("metal", metal);
    const inkling2_glue_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/glue_plan.zig"), .target = target, .optimize = optimize });
    const inkling2_loop_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/loop_plan.zig"), .target = target, .optimize = optimize });
    const inkling2_tree_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/tree_plan.zig"), .target = target, .optimize = optimize });
    // Living Weights' family-neutral learner, which Nemotron's lessons and the server's view of them share
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const engine = b.createModule(.{
        .root_source_file = b.path("zig/src/tensorfold.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "nemotron_draft_ids", .module = draft_ids }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "living", .module = living } },
    });
    // the DeepSeek V4.1 Metal files shared with its package reach the EXL3 kernels as `@import("core").exl3_projection`
    engine.addImport("core", engine);
    engine.addImport("inkling2_metal", inkling2_metal);
    engine.addImport("inkling2_metal_ref", inkling2_metal_ref);
    engine.addImport("inkling2_glue_plan", inkling2_glue_plan);
    engine.addImport("inkling2_loop_plan", inkling2_loop_plan);
    engine.addImport("inkling2_tree_plan", inkling2_tree_plan);
    engine.addImport("engram_hash_core", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target, .optimize = optimize }));
    engine.addImport("config_number", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/config_number.zig") }));
    return .{ .metal = metal, .lanes = lanes, .fabric = fabric, .sources = sources, .engine = engine, .living = living, .qwen27 = qwen27_build.engineModules(b, target, metal, sources, engine) };
}

pub fn distMetalServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, sdk: []const u8) *std.Build.Step.Compile {
    const m = metalEngineModules(b, target, .fast, draft_ids);
    // the release binary embeds the same prebuilt packed-kernel metallib as the development build
    const flags = [_][]const u8{ b.fmt("-std={s}", .{mlxMetalStd(b)}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast" };
    m.sources.addImport("nemotron_packed_metallib", embedded(b, metallib(b, "nemotron_packed", &nemotron_packed, &flags), "nemotron_packed.metallib"));
    // An explicit deployment target is a cross-build even on macOS. Supply the SDK search paths.
    m.metal.addSystemFrameworkPath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk, "System/Library/Frameworks" })));
    m.metal.addLibraryPath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk, "usr/lib" })));
    m.metal.addSystemIncludePath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk, "usr/include" })));
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .fast, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = m.lanes }} });
    m.engine.addImport("engine_api", api);
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal.zig"), .target = target, .optimize = .fast, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = m.metal }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = m.engine } } });
    engines.addImport("living", m.living);
    engines.addImport("kernel_sources", m.sources);
    inline for ([_][]const u8{ "inkling2_metal", "inkling2_metal_ref", "inkling2_glue_plan", "inkling2_loop_plan", "inkling2_tree_plan" }) |name|
        engines.addImport(name, m.engine.import_table.get(name).?);
    drama3_build.addNative(b, target, .fast, m.metal, engines, drama3_build.metallib(b, mlxMetalStd(b)));
    engines.addImport("kimi_k3", kimiModule(b, target, .fast, m.metal, m.engine, m.lanes));
    engines.addImport("kimi_metallib", embedded(b, kimiMetallib(b, &flags), "kimi.metallib"));
    return metal_server.nativeServer(b, target, .fast, m.metal, m.engine, m.living, api, engines, build_options, null, null);
}

pub fn kimiModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const kimi = b.createModule(.{
        .root_source_file = b.path("zig/src/families/kimi_k3/kimi_k3.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = b.createModule(.{
            .root_source_file = b.path("zig/src/families/kimi_k3/core_safetensors.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "core", .module = engine }},
        }) } },
    });
    kimi.addImport("lanes", lanes);
    return kimi;
}
