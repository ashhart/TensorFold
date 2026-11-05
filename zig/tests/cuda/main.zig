//! GPU test runner for the CUDA runtime: `tf-cuda-test <command> [args]`, PASS/FAIL/RESULT lines, exit 1 on failure.

const std = @import("std");
const q27_back_tests = @import("q27_back_tests.zig");
const q27_gemm_tests = @import("q27_gemm_tests.zig");
const cuda = @import("cuda");
const check = @import("check.zig");
const dump = @import("byte_dump.zig");
const runtime_tests = @import("runtime_tests.zig");
const bench = @import("bench.zig");
const oracle_tests = @import("oracle_tests.zig");
const libs_tests = @import("libs_tests.zig");
const rank_scatter_tests = @import("rank_scatter_tests.zig");
const roce_gather_tests = @import("roce_gather_tests.zig");
const sample_tests = @import("sample_tests.zig");
const glue_tests = @import("glue_tests.zig");
const window_profile = @import("window_profile.zig");
const tree_accept = @import("tree_accept.zig");
const resume_tests = @import("resume_tests.zig");
const prompt_cuts = @import("prompt_cuts.zig");
const shared_fork = @import("shared_fork.zig");
const chunk_costs = @import("chunk_costs.zig");
const grouped_tests = @import("grouped_tests.zig");
const fp8_tests = @import("fp8_tests.zig");
const mxfp8_tests = @import("mxfp8_tests.zig");
const projection_f32_tests = @import("projection_f32_tests.zig");
const fp8_experts_tests = @import("fp8_experts_tests.zig");
const nvfp4_experts_tests = @import("nvfp4_experts_tests.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const glm_ref_tests = @import("glm_ref_tests.zig");
const affine8_tests = @import("affine8_tests.zig");
const native_fp_tests = @import("native_fp_tests.zig");
const affine4_decode_tests = @import("affine4_decode_tests.zig");
const w4a8_tests = @import("w4a8_tests.zig");
const glm_exl3_tests = @import("glm_exl3_tests.zig");
const glm_kda_pad_check = @import("glm_kda_pad_check.zig");
const glm_vision_tests = @import("glm_vision_tests.zig");
const glm_mm_fixed = @import("glm_mm_fixed_tests.zig");
const glm_mm_bench = @import("glm_mm_bench.zig");
const glm_index_check = @import("glm_index_check.zig");
const carveout_tests = @import("carveout_tests.zig");
const nvfp4_tests = @import("nvfp4_tests.zig");
const modelopt_tests = @import("modelopt_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const exl3_tp2_tests = @import("exl3_tp2_tests.zig");
const exl3_plan_bench = @import("exl3_plan_bench.zig");
const exl3_contract = @import("contract_exl3.zig");
const exl3_candidate_tests = @import("exl3_candidate_tests.zig");
const experts3_tests = @import("experts3_tests.zig");
const moe_tests = @import("moe_tests.zig");
const ck_tests = @import("ck_tests.zig");
const train_pack = @import("train_pack.zig");
const exlr8_tests = @import("exlr8_tests.zig");
const exlr8_dense_tests = @import("exlr8_dense_tests.zig");
const norms_tests = @import("norms_tests.zig");
const norm_rows_tests = @import("norm_rows_tests.zig");
const norm_tail_tests = @import("norm_tail_tests.zig");
const embed_tests = @import("embed_tests.zig");
const pocket_tests = @import("pocket_tests.zig");
const recapture_tests = @import("recapture_tests.zig");
const contract_tests = @import("contract_tests.zig");
const gb10_many_contract = @import("contract_gb10_many.zig");
const qmm_prefill_many_contract = @import("contract_qmm_prefill_many.zig");
const qmm_prefill_tile22_contract = @import("contract_qmm_prefill_tile22.zig");
const glue_contract = @import("contract_flashnext_glue.zig");
const lfm2_attention_tests = @import("lfm2_attention_tests.zig");
const lfm2_kernel_tests = @import("lfm2_kernel_tests.zig");
const lfm2_vision_tests = @import("lfm2_vision_tests.zig");
const radio_tests = @import("radio_tests.zig");
const glm_lanes_tests = @import("glm_lanes_tests.zig");
const glm_pair_tests = @import("glm_pair_tests.zig");
const glm_split_tests = @import("glm_split_tests.zig");
const glm_solo_tests = @import("glm_solo_tests.zig");
const glm_paged_tests = @import("glm_paged_tests.zig");
const glm_kv_host_slots = @import("glm_kv_host_slots.zig");
const glm_tap_tests = @import("glm_tap_tests.zig");
const glm_synth = @import("glm_synth.zig");
const tree_round_tests = @import("tree_round_tests.zig");
const tree_tests = @import("tree_tests.zig");
const block_tests = @import("block_tests.zig");
const act_rows_tests = @import("act_rows_tests.zig");
const gdn_prep_tests = @import("gdn_prep_tests.zig");
const qk_rows_tests = @import("qk_rows_tests.zig");
const window_tests = @import("window_tests.zig");
const window_rows_tests = @import("window_rows_tests.zig");
const window_pages512_tests = @import("window_pages512_tests.zig");
const window_bench = @import("window_bench.zig");
const moe_glue_tests = @import("moe_glue_tests.zig");
const dense_rows_tests = @import("dense_rows_tests.zig");

const experts_rows_tests = @import("experts_rows_tests.zig");
const route_rows_tests = @import("route_rows_tests.zig");
const flashnext_model_rank_tests = @import("flashnext_model_rank_tests.zig");
const flashnext_fidelity_tests = @import("flashnext_fidelity_tests.zig");
const flashnext_window_tests = @import("flashnext_window_tests.zig");
const flashnext_windows_tests = @import("flashnext_windows_tests.zig");
const flashnext_mtp_parity_tests = @import("flashnext_mtp_parity_tests.zig");
const flashnext_score_tests = @import("flashnext_score_tests.zig");
const flashnext_rank_tests = @import("flashnext_rank_tests.zig");
const flashnext_draft_affine_tests = @import("flashnext_draft_affine_tests.zig");
const flashnext_shared_tests = @import("flashnext_shared_tests.zig");
const flashnext_nvfp4_tp_tests = @import("flashnext_nvfp4_tp_tests.zig");
const flashnext_fp8_experts_tests = @import("flashnext_fp8_experts_tests.zig");
const nvfp4_fact_tests = @import("nvfp4_fact_tests.zig");
const nvfp4_b32_tests = @import("nvfp4_b32_tests.zig");
const lane_half_tests = @import("lane_half_tests.zig");
const flashnext_dense_hc_tests = @import("flashnext_dense_hc_tests.zig");
const flashnext_hc_tests = @import("flashnext_hc_tests.zig");
const flashnext_attn_tests = @import("flashnext_attn_tests.zig");
const exl3_pack_tests = @import("flashnext_exl3/pack/exl3_pack_tests.zig");
const exl3_prefill_tests = @import("flashnext_exl3/prefill/exl3_prefill_tests.zig");
const flashnext_exl3_tests = @import("flashnext_exl3/model/flashnext_exl3_tests.zig");
const flashnext_exl3_tp_tests = @import("flashnext_exl3/tp/flashnext_exl3_tp_tests.zig");
const flashnext_exl3_engine_tests = @import("flashnext_exl3/model/flashnext_exl3_engine_tests.zig");
const flashnext_exl3_tp_engine_tests = @import("flashnext_exl3/tp/flashnext_exl3_tp_engine_tests.zig");
const ggml_tests = @import("ggml_tests.zig");
const paged_tests = @import("paged_tests.zig");
const glm_loader_gate = @import("glm_loader_gate.zig");
const glm_fp8g_shares = @import("glm_fp8g_shares.zig");

const gdn_back_tests = @import("gdn_back_tests.zig");
const q4_back_tests = @import("q4_back_tests.zig");
const moe_back_tests = @import("moe_back_tests.zig");
const fn_moe_back_tests = @import("fn_moe_back_tests.zig");
const attn_back_tests = @import("attn_back_tests.zig");
const fn_hc_back_tests = @import("fn_hc_back_tests.zig");
const fn_pocket_tests = @import("fn_pocket_tests.zig");
const kolibri_tests = @import("kolibri_tests.zig");
const dory_tests = @import("dory_tests.zig");
const dory_serve = @import("dory_serve_tests.zig");
const kolibri_layer_tests = @import("kolibri_layer_tests.zig");
const aot_fallback_tests = @import("aot_fallback_tests.zig");
const qwen_draft_frames = @import("qwen_draft_frames.zig");

const usage = @import("runner_usage.zig").text;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    const cmd = args[1];
    const rest = args[2..];

    try dump.begin(init.gpa, cmd);
    run(gpu, cmd, rest) catch |e| {
        if (std.mem.eql(u8, cmd, "cached-plans-tp2")) cuda.tp_watch.fatal(@errorName(e));
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, e });
        try dump.end(init.gpa, cmd, false);
        return 1;
    };
    try dump.end(init.gpa, cmd, true);
    return 0;
}

fn arg(rest: []const [:0]const u8, i: usize) ![]const u8 {
    if (i >= rest.len) {
        std.debug.print("{s}", .{usage});
        return error.MissingArgument;
    }
    return rest[i];
}

/// The contract glue's device path, compiled with the runner: the device reaches prepare through the vtable.
fn compileContractGlue() !void {
    const cores = std.Thread.getCpuCount() catch 1;
    if (cores != 0) return;
    var g: glue_contract.Glue = undefined;
    g.kind = .chain;
    const kernels = [_]@import("core").contract.gates.Kernel{g.kernel()};
    for (kernels) |k| _ = try k.vt.prepare(k.ctx, .{ .rows = 0, .n = 0, .k = 0 }, .random, 0);
}

fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "qwen4")) {
        if (rest.len != 0) return error.Qwen4DeviceArguments;
        return @import("qwen4_tests.zig").run(gpu);
    }
    if (std.mem.eql(u8, cmd, "compile-contract-glue")) return compileContractGlue();
    if (std.mem.eql(u8, cmd, "qwen-draft-frames")) {
        if (rest.len != 4) return error.FrameGateArguments;
        return qwen_draft_frames.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    }
    if (std.mem.eql(u8, cmd, "experts-rows")) return experts_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "route-rows")) return route_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-fidelity-single")) return flashnext_fidelity_tests.runSingle(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), false);
    if (std.mem.eql(u8, cmd, "flashnext-fidelity-single-serial")) return flashnext_fidelity_tests.runSingle(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), true);
    if (std.mem.eql(u8, cmd, "flashnext-fidelity-rank")) return flashnext_fidelity_tests.run(gpu, try arg(rest, 0), try std.fmt.parseInt(usize, try arg(rest, 1), 10), try std.fmt.parseInt(u16, try arg(rest, 2), 10), try arg(rest, 3), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "flashnext-windows")) return flashnext_windows_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "flashnext-mtp-parity")) return flashnext_mtp_parity_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "flashnext-score")) return flashnext_score_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "flashnext-window-rank")) return flashnext_window_tests.run(gpu, try arg(rest, 0), try std.fmt.parseInt(usize, try arg(rest, 1), 10), try std.fmt.parseInt(u16, try arg(rest, 2), 10), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "flashnext-model-rank")) return flashnext_model_rank_tests.run(gpu, try arg(rest, 0), try std.fmt.parseInt(usize, try arg(rest, 1), 10), try std.fmt.parseInt(u16, try arg(rest, 2), 10), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "flashnext-rank")) return flashnext_rank_tests.run(gpu, try std.fmt.parseInt(usize, try arg(rest, 0), 10), try std.fmt.parseInt(u16, try arg(rest, 1), 10), if (rest.len > 2) rest[2] else "sum");
    if (std.mem.eql(u8, cmd, "flashnext-draft-affine")) return flashnext_draft_affine_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-shared")) return flashnext_shared_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-nvfp4-tp")) return flashnext_nvfp4_tp_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-fp8-experts")) return flashnext_fp8_experts_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "moetile")) return @import("moetile_tests.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "moetile-contract")) return @import("contract_qmmf_shared.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "moetile-bench")) return @import("moetile_tests.zig").bench(gpu);
    if (std.mem.eql(u8, cmd, "nvfp4-fact")) return nvfp4_fact_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "nvfp4-b32")) return nvfp4_b32_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "qmmf-prompt")) return @import("qmmf_prompt_tests.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "qmmf-prompt-contract")) return @import("contract_qmmf_prompt.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "qmmf-prompt-repro")) return @import("qmmf_prompt_tests.zig").repro(gpu);
    if (std.mem.eql(u8, cmd, "qmmf-prompt-bench")) return @import("qmmf_prompt_tests.zig").bench(gpu);
    if (std.mem.eql(u8, cmd, "lane-half")) return lane_half_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-dense-hc")) return flashnext_dense_hc_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-hc")) return flashnext_hc_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-attn")) return flashnext_attn_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-gdn-heads")) {
        if (rest.len > 1) return error.GdnHeadsArguments;
        const heads = @import("flashnext_gdn_heads_tests.zig");
        return if (rest.len == 1) heads.runBaseline(gpu, rest[0]) else heads.run(gpu);
    }
    if (std.mem.eql(u8, cmd, "flashnext-tp-heads")) return @import("flashnext_tp_heads_tests.zig").run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "exl3-prefill-contract")) return @import("contract_flashnext_exl3_prefill.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "exl3-promptd-contract")) return @import("contract_flashnext_exl3_promptd.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "exl3-tppf-contract")) return @import("contract_flashnext_exl3_tppf.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "exl3-kvt-contract")) return @import("contract_flashnext_exl3_kvt.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-exl3")) return flashnext_exl3_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-kvtier")) return @import("flashnext_exl3/kv/exl3_kvtier_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "flashnext-exl3-tp")) return flashnext_exl3_tp_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "flashnext-exl3-engine")) return flashnext_exl3_engine_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "flashnext-exl3-tpe")) return flashnext_exl3_tp_engine_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "flashnext-pages")) return @import("contract_fn_pages.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-attn-lanes")) return @import("contract_fn_attn_lanes.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-readout-cut")) return @import("flashnext_readout_cut.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "info")) return runtime_tests.info(gpu);
    if (std.mem.eql(u8, cmd, "smoke")) return runtime_tests.smoke(gpu);
    if (std.mem.eql(u8, cmd, "shared-context")) return runtime_tests.sharedContext(gpu);
    if (std.mem.eql(u8, cmd, "graph")) return runtime_tests.graphs(gpu);
    if (std.mem.eql(u8, cmd, "overhead") or std.mem.eql(u8, cmd, "overhead-pdl")) {
        const n = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 1000;
        const reps = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 20;
        return bench.overhead(gpu, n, reps, std.mem.eql(u8, cmd, "overhead-pdl"));
    }
    if (std.mem.eql(u8, cmd, "launch-ex")) return runtime_tests.launchEx(gpu);
    if (std.mem.eql(u8, cmd, "ptx")) return runtime_tests.ptx(gpu);
    if (std.mem.eql(u8, cmd, "symbols")) return runtime_tests.symbols(gpu);
    if (std.mem.eql(u8, cmd, "cublaslt")) return libs_tests.cublaslt(gpu);
    if (std.mem.eql(u8, cmd, "nccl")) return libs_tests.nccl(gpu);
    if (std.mem.eql(u8, cmd, "rank-scatter")) {
        const rank = try std.fmt.parseInt(usize, try arg(rest, 0), 10);
        const world = try std.fmt.parseInt(usize, try arg(rest, 1), 10);
        return rank_scatter_tests.run(gpu, rank, world, if (rest.len > 2) rest[2] else null);
    }
    if (std.mem.eql(u8, cmd, "roce-gather")) return roce_gather_tests.run(gpu, try std.fmt.parseInt(usize, try arg(rest, 0), 10));
    if (std.mem.eql(u8, cmd, "rank-sum")) return rank_scatter_tests.sum(gpu);
    if (std.mem.eql(u8, cmd, "nccl-pair")) return libs_tests.ncclPair(gpu, try std.fmt.parseInt(c_int, try arg(rest, 0), 10), try std.fmt.parseInt(c_int, try arg(rest, 1), 10), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "gdn-replay")) return oracle_tests.gdnReplay(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "gdn-tree")) return oracle_tests.gdnTree(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "triton")) return oracle_tests.tritonKernel(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "fp8-lane")) return fp8_tests.lane(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "fp8-experts")) return fp8_experts_tests.experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "nvfp4")) return nvfp4_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "nvfp4-experts")) return nvfp4_experts_tests.experts(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "projection-f32")) return projection_f32_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "mxfp8")) return mxfp8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "quant-ref")) return quant_ref_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-ref")) return glm_ref_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "affine8")) return affine8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "native-fp")) return native_fp_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "affine4-decode")) return affine4_decode_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "w4a8")) return w4a8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-exl3")) return glm_exl3_tests.run_(gpu);
    if (std.mem.eql(u8, cmd, "glm-kda-pad-split")) return glm_kda_pad_check.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-mm-fixed")) return glm_mm_fixed.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-mm-bench")) return glm_mm_bench.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-index-check")) return glm_index_check.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-kv-host-slots")) return glm_kv_host_slots.run(gpu);
    if (std.mem.eql(u8, cmd, "glm-vision")) return glm_vision_tests.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "modelopt")) return modelopt_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "moe")) return moe_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "moe-live")) return @import("moe_live_tests.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "moe-router-bench")) return @import("moe_router_bench.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "experts3")) return experts3_tests.run(gpu, if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 0);
    if (std.mem.eql(u8, cmd, "ggml")) return ggml_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "ggml-bench")) return ggml_tests.bench(gpu);
    if (std.mem.eql(u8, cmd, "checkpoint")) return ck_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "exlr8")) return exlr8_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "exlr8-dense")) return exlr8_dense_tests.run(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "exlr8-bench")) return exlr8_tests.bench(gpu, try arg(rest, 0), if (rest.len > 1) rest[1] else "home", if (rest.len > 2) try std.fmt.parseInt(u16, rest[2], 10) else 40, if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 4);
    if (std.mem.eql(u8, cmd, "exl3")) return exl3_tests.run(gpu, if (rest.len > 0) rest[0] else "none");
    if (std.mem.eql(u8, cmd, "exl3-pack")) return exl3_pack_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-ple2")) return @import("flashnext_exl3/model/exl3_ple2_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-draft")) return @import("flashnext_exl3/tp/exl3_draft_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-pool")) return @import("flashnext_exl3/offload/exl3_pool_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-prefill")) return exl3_prefill_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-tpprefill")) return @import("flashnext_exl3/tp/exl3_tpprefill_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "nccl-lat")) return exl3_pack_tests.ncclLatency(gpu, rest);
    if (std.mem.eql(u8, cmd, "exl3-tiles")) return exl3_candidate_tests.run(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "exl3-shapes")) return exl3_plan_bench.profileShapes(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "exl3-midm")) return exl3_plan_bench.midm(gpu);
    if (std.mem.eql(u8, cmd, "exl3-plans")) return exl3_plan_bench.run(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "exl3-contract")) return exl3_contract.run(gpu);
    if (std.mem.eql(u8, cmd, "exl3-tp2")) return exl3_tp2_tests.run(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2) rest[2] else "none");
    if (std.mem.eql(u8, cmd, "sample")) return sample_tests.draws(gpu);
    if (std.mem.eql(u8, cmd, "topk")) return sample_tests.topk(gpu);
    if (std.mem.eql(u8, cmd, "q35-argmax")) return sample_tests.q35Argmax(gpu);
    if (std.mem.eql(u8, cmd, "norms")) return norms_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "norm-rows")) return norm_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "norm-tail")) return norm_tail_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "embed-rows")) return embed_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "tree-round")) return tree_round_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "act-rows")) return act_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "gdn-prep")) return gdn_prep_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "qk-rows")) return qk_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "glue")) return glue_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "lfm2-kernels")) return lfm2_kernel_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "gdn-back")) return gdn_back_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "q4-back")) return q4_back_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "moe-back")) return moe_back_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "attn-back")) return attn_back_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "fn-hc-back")) return fn_hc_back_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "fn-moe-back")) return fn_moe_back_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "fn-pocket")) return fn_pocket_tests.run(gpu, if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 0);
    if (std.mem.eql(u8, cmd, "fn-back")) {
        try gdn_back_tests.run(gpu);
        try q4_back_tests.run(gpu);
        try moe_back_tests.run(gpu);
        try fn_moe_back_tests.run(gpu);
        try attn_back_tests.run(gpu);
        try fn_hc_back_tests.run(gpu);
        return;
    }
    if (std.mem.eql(u8, cmd, "pocket"))
        return pocket_tests.run(gpu, if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 0);
    if (std.mem.eql(u8, cmd, "lfm2-attention")) return lfm2_attention_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "lfm2-vision")) return lfm2_vision_tests.run(gpu, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "radio")) return radio_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "radio-processor")) return radio_tests.processor(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "radio-kernels")) return radio_tests.kernels(gpu);
    if (std.mem.eql(u8, cmd, "radio-attention")) return radio_tests.attention(gpu);
    if (std.mem.eql(u8, cmd, "window-profile")) return window_profile.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "tree-accept")) return tree_accept.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "chunk-costs")) return chunk_costs.run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "resume")) return resume_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "marks")) return resume_tests.insideChunks(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "prompt-cuts")) return prompt_cuts.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "prompt-segments")) return prompt_cuts.segments(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "cached-plans-tp2")) return @import("shared_plans.zig").tp2(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "shared-fork")) return shared_fork.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "recapture")) return recapture_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "paged")) return paged_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "grouped-plan")) return grouped_tests.plan(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "carveout")) {
        const mib = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else cuda.carveout.default_bytes >> 20;
        return carveout_tests.run(gpu, if (rest.len > 1) rest[1] else cuda.carveout.default_card, mib);
    }
    if (std.mem.eql(u8, cmd, "q27-back")) return q27_back_tests.run(gpu, .all);
    if (std.mem.eql(u8, cmd, "q27-gemm")) return q27_gemm_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "gdn-back-27b")) return gdn_back_tests.runShape(gpu, 5120, 64);
    if (std.mem.eql(u8, cmd, "kernel-contract")) return contract_tests.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "gb10-grouped-contract")) return gb10_many_contract.runAll(gpu);
    if (std.mem.eql(u8, cmd, "qmm-prefill-many-contract")) return qmm_prefill_many_contract.run(gpu);
    if (std.mem.eql(u8, cmd, "qmm-prefill-tile22-contract")) return qmm_prefill_tile22_contract.run(gpu);
    if (std.mem.eql(u8, cmd, "qmm-prefill-many-bench")) return qmm_prefill_many_contract.bench(gpu);
    if (std.mem.eql(u8, cmd, "mtp-rows")) return @import("contract_qwen_mtp_rows.zig").alone(gpu);
    if (std.mem.eql(u8, cmd, "qwen-pages")) return @import("contract_qwen_pages.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "qwen-norm-shift")) return @import("qwen_norm_shift_tests.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "clef")) return @import("clef_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "qwen-pending")) return @import("contract_qwen_pending.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "dspine-table")) return @import("dspine_table_tests.zig").runAll(gpu);
    if (std.mem.eql(u8, cmd, "minimax-contract")) return @import("contract_minimax.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "rope-yarn")) return @import("contract_qwen_rope_yarn.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "kernel-contract-limits")) return contract_tests.fatbins(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "pocket-contract")) return @import("contract_pocket.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "flashnext-modelopt-contract")) return @import("contract_flashnext_modelopt.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "moss-contract")) return @import("contract_moss_image.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "tree")) return tree_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "block")) return block_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "window")) return window_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "window-rows")) return window_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "window-pages-512")) return window_pages512_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "window-bench")) return window_bench.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "moe-glue")) return moe_glue_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "dense-rows")) return dense_rows_tests.run(gpu);
    if (std.mem.eql(u8, cmd, "train-pack")) return train_pack.run(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 24);
    if (std.mem.eql(u8, cmd, "glm-lanes-cover")) return glm_lanes_tests.coverCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-paged-check")) return glm_paged_tests.pagedCheck(gpu, try arg(rest, 0), try std.fmt.parseInt(c_int, try arg(rest, 1), 10), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "glm-shared-check")) return glm_lanes_tests.sharedCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-pair-check")) return glm_pair_tests.pairCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-split-check")) return glm_split_tests.splitCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-solo-check")) return glm_solo_tests.soloCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-tap-split-check")) return glm_tap_tests.tapSplitCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    if (std.mem.eql(u8, cmd, "glm-synth")) {
        const kind = if (rest.len > 1) std.meta.stringToEnum(glm_synth.Kind, rest[1]) orelse return error.UnknownCommand else .nvfp4;
        return glm_synth.run(gpu, try arg(rest, 0), kind);
    }
    if (std.mem.eql(u8, cmd, "glm-loader-gate")) return glm_loader_gate.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "glm-fp8g-shares")) return glm_fp8g_shares.run(gpu);
    if (std.mem.startsWith(u8, cmd, "dory-")) return dory_serve.run(gpu, cmd, rest);
    if (std.mem.eql(u8, cmd, "gemma4-states")) return @import("gemma4_states_tests.zig").states(gpu, rest);
    if (std.mem.eql(u8, cmd, "spark25-pages")) return @import("spark25_pages_tests.zig").pages(gpu, rest);
    if (std.mem.eql(u8, cmd, "gemma4-pages")) return @import("gemma4_pages_tests.zig").pages(gpu, rest);
    if (std.mem.eql(u8, cmd, "gemma4-ple")) return @import("gemma4_ple_tests.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "gemma4-e4b-force")) return @import("gemma4_e4b_force.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "dory")) return dory_tests.greedy(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "kolibri-reference")) return kolibri_layer_tests.reference(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "kolibri-layers")) return kolibri_layer_tests.layers(gpu, try arg(rest, 0), try arg(rest, 1));
    if (std.mem.eql(u8, cmd, "kolibri-full")) return kolibri_tests.full(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2));
    if (std.mem.eql(u8, cmd, "kolibri-pages")) return @import("kolibri_pages_tests.zig").pages(gpu, rest);
    if (std.mem.eql(u8, cmd, "kolibri-greedy")) return kolibri_tests.greedy(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    if (std.mem.eql(u8, cmd, "kolibri-logits")) return kolibri_tests.logits(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), if (rest.len > 3) rest[3] else null);
    if (std.mem.eql(u8, cmd, "aot-fallback")) return aot_fallback_tests.run(gpu, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "glm-lanes-concurrent")) return glm_lanes_tests.concurrentCheck(gpu, try arg(rest, 0), try arg(rest, 1), try std.fmt.parseInt(c_int, try arg(rest, 2), 10), try std.fmt.parseInt(c_int, try arg(rest, 3), 10), try arg(rest, 4));
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn hasFlag(flags: []const [:0]const u8, name: []const u8) bool {
    for (flags) |f| if (std.mem.eql(u8, f, name)) return true;
    return false;
}
