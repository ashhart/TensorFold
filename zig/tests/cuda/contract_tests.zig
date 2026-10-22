//! tf-cuda-test kernel-contract: the registered CUDA kernels through gates 1-4, the planted controls, and gate 5.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const options = @import("contract_options");
const check = @import("check.zig");
const rows_k = @import("contract_rows.zig");
const tail_k = @import("contract_norms_tail.zig");
const router_k = @import("contract_router.zig");
const mm_k = @import("contract_mm.zig");
const mx_k = @import("contract_mxfp8.zig");
const half_k = @import("contract_lane_half.zig");
const wide_k = @import("contract_projection_f32.zig");
const rank_k = @import("contract_rank.zig");
const writeback_k = @import("contract_flashnext_writeback.zig");
const hcnorm_k = @import("contract_hc_normalize.zig");
const hcmix_k = @import("contract_hc_mix.zig");
const utility_k = @import("contract_utility.zig");
const best_k = @import("contract_row_best.zig");
const qsa_k = @import("contract_qsa_select.zig");
const h3_k = @import("contract_h3.zig");
const h3_prep = @import("contract_h3_prep.zig");
const h3_topk_k = @import("contract_h3_topk.zig");
const drama3_k = @import("contract_drama3.zig");
const ck_k = @import("contract_ck.zig");
const q35_bf16_k = @import("contract_qwen_bf16.zig");
const e3_k = @import("contract_experts3.zig");
const nvfp4_fact_k = @import("contract_nvfp4_fact.zig");
const scan_k = @import("contract_scan.zig");
const lane8_k = @import("contract_ck_lane.zig");
const latent_k = @import("contract_latent.zig");
const vit_k = @import("contract_vit_glue.zig");
const act_round_k = @import("contract_act_round.zig");
const f32w_k = @import("contract_glm_f32w.zig");

const w4a8_k = @import("contract_w4a8.zig");

const round_k = @import("contract_tree_round.zig");

const act_k = @import("contract_act.zig");
const vit_g4_group = @import("contract_gemma_group.zig");
const gdn_k = @import("contract_gdn.zig");
const qk_k = @import("contract_qk.zig");
const wrows_k = @import("contract_window_rows.zig");
const glm_pages_k = @import("contract_glm_pages.zig");
const glm_index_k = @import("contract_glm_index.zig");
const glm_attn_k = @import("contract_glm_attn.zig");
const glm = @import("glm_cuda");
const embed_k = @import("contract_embed.zig");

const tree_k = @import("contract_tree_attn.zig");
const block_k = @import("contract_block_attn.zig");
const q35_k = @import("contract_q35_attn.zig");

const window_k = @import("contract_window.zig");
const moe_k = @import("contract_moe.zig");
const qkr_k = @import("contract_qk_round.zig");
const fact_k = @import("contract_flashnext_fact.zig");
const route_k = @import("contract_route_rows.zig");
const fexp_k = @import("contract_flashnext_fexp.zig");
const glue_k = @import("contract_flashnext_glue.zig");

const contract = core.contract;
const g = contract.gates;
const kk = cuda.kernels;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const h3_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3").*;
};

/// Gates 1-4 on every registered CUDA kernel and each control at its own gate, then gate 5 on this GPU.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    if (!kk.available) return error.BuiltWithoutKernels;
    if (!options.controls) {
        std.debug.print("FAIL kernel-contract: the controls are built only with -Dnvcc; without them no gate is proven\n", .{});
        return error.TestFailed;
    }
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();
    var mods: [8]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.nemotron_route, kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, &controls.bytes, kk.experts, kk.nemotron_mamba }, 0..) |img, i| {
        mods[i] = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const cap = try gpu.ctx.capability();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var affine4 = try cuda.qlinear.Affine4Kernels.resolve(gpu.d, mods[1], mods[2], mods[3], mods[4], sms, cap == 121);
    defer affine4.deinit();
    var split = try cuda.qlinear.Split.initFor(gpu.d, 2); // GB10 serves without it; other GPUs take it for K slices
    defer split.deinit();
    var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var lane = try cuda.qmmf.Lane.load(gpu.d, cap / 10); // sm_90 up: K slices meet in a cluster
    defer lane.unload();
    var lane_reduce = try cuda.qmmf.Lane.load(gpu.d, 8); // no clusters: the slices meet in reduce_kernel
    defer lane_reduce.unload();
    var rank_sum = try cuda.rank_scatter.RankScatter.load(gpu.d);
    defer rank_sum.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    var vit = try cuda.vit.Kernels.load(gpu.d);
    defer vit.unload();
    var glue: [12]vit_k.Glue = undefined;
    for (&glue, 0..) |*item, i| item.* = .{
        .rig = .{ .gpu = gpu, .s = s },
        .k = &vit,
        .act = @fromBackingInt(@intCast(i % 4)),
        .residual = i >= 4,
        .in_place = i >= 8,
    };
    defer for (&glue) |*item| item.deinit();
    for (&glue) |*item| try kernels.append(a, item.kernel());
    var wrong_vit: vit_k.Glue = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &vit, .act = .none, .residual = true, .in_place = false, .wrong_bias_round = true, .ctl = try mods[5].function("tf_ctl_vit_unrounded_bias") };
    defer wrong_vit.deinit();
    try kernels.append(a, wrong_vit.kernel());
    var w4a8_mod = try cuda.Module.load(gpu.d, kk.w4a8_prefill);
    defer w4a8_mod.unload();
    const w4a8_fns = try cuda.w4a8.Kernels.resolve(w4a8_mod);
    const w4a8_configs = [_]struct { kind: w4a8_k.Kind, fault: w4a8_k.Fault }{
        .{ .kind = .quantize, .fault = .none },       .{ .kind = .prefill, .fault = .none },
        .{ .kind = .quantize, .fault = .guard },      .{ .kind = .prefill, .fault = .guard },
        .{ .kind = .quantize, .fault = .wrong_code }, .{ .kind = .quantize, .fault = .wrong_scale },
        .{ .kind = .quantize, .fault = .wrong_sum },  .{ .kind = .quantize, .fault = .narrow_sum },
        .{ .kind = .prefill, .fault = .wrong_code },  .{ .kind = .prefill, .fault = .narrow_epilogue },
    };
    var w4a8_cases: [w4a8_configs.len]w4a8_k.Case = undefined;
    for (&w4a8_cases, w4a8_configs) |*case, cfg| case.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = cfg.kind, .fault = cfg.fault, .kernels = w4a8_fns, .arena = .init(a) };
    defer for (&w4a8_cases) |*case| case.deinit();
    for (&w4a8_cases) |*case| try kernels.append(a, case.kernel());
    var q35_bf16 = try cuda.qwen35_bf16_rows.Kernels.load(gpu.d);
    defer q35_bf16.unload();
    const Q35Bf16Case = q35_bf16_k.Case(@TypeOf(q35_bf16));
    var q35_cases: [q35_bf16_k.entries.len]Q35Bf16Case = undefined;
    for (&q35_cases, q35_bf16_k.entries) |*item, entry| item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kernels = &q35_bf16, .entry = entry, .ctl = if (entry.fault == .half) try mods[5].function("tf_ctl_q35_dense_half") else null };
    defer for (&q35_cases) |*item| item.deinit();
    for (&q35_cases) |*item| try kernels.append(a, item.kernel());
    var row_list: [7]rows_k.Row = undefined;
    var nrows: usize = 0;
    defer for (row_list[0..nrows]) |*r| r.deinit();
    for (std.enums.values(rows_k.Which)) |w| {
        row_list[nrows] = try rows_k.Row.init(gpu, s, &norms, mods[5], w);
        nrows += 1;
    }
    for (row_list[0..nrows]) |*r| try kernels.append(a, r.kernel());
    const tail_which = comptime std.enums.values(tail_k.Which);
    var tails: [tail_which.len]tail_k.Tail = undefined;
    for (tail_which, &tails) |w, *t| t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .norms = &norms, .ctl = if (tail_k.control(w) != null) try mods[5].function(tail_k.symbol(w)) else null };
    defer for (&tails) |*t| t.deinit();
    for (&tails) |*t| try kernels.append(a, t.kernel());
    var logits = router_k.Logits{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[0].function("tf_nemo_router") };
    defer logits.deinit();
    var logits_tiles = router_k.Logits{ .rig = .{ .gpu = gpu, .s = s }, .f = logits.f, .tiles = try mods[0].function("tf_nemo_router_tiles") };
    defer logits_tiles.deinit();
    var topk = router_k.TopK{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[0].function("tf_nemo_topk") };
    defer topk.deinit();
    var latent = try nemotron.latent.Fns.load(gpu.d);
    defer latent.unload();
    var latent_topk = router_k.LatentTopK{ .rig = .{ .gpu = gpu, .s = s }, .fns = &latent };
    defer latent_topk.deinit();
    try kernels.append(a, logits.kernel());
    try kernels.append(a, logits_tiles.kernel());
    try kernels.append(a, topk.kernel());
    try kernels.append(a, latent_topk.kernel());
    var scan = scan_k.Scan{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[7].function("tf_nemo_scan"), .host = .init(a) };
    defer scan.deinit();
    try kernels.append(a, scan.kernel());
    var latent_sums: [2]latent_k.Sums = undefined;
    for (std.enums.values(latent_k.Kind), &latent_sums) |kind, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .fns = &latent };
    defer for (&latent_sums) |*x| x.deinit();
    for (&latent_sums) |*x| try kernels.append(a, x.kernel());
    const kinds = comptime std.enums.values(mm_k.Kind);
    var mm: [kinds.len]mm_k.Mm = undefined;
    for (kinds, &mm) |kind, *m| {
        const by_lane: ?*const cuda.qmmf.Lane = switch (kind) {
            .lane_f32 => &lane,
            .lane_f32_reduce => &lane_reduce,
            else => null,
        };
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .sm = cap, .affine4 = &affine4, .split = &split, .prompt = &prompt, .lane = by_lane };
    }
    defer for (&mm) |*m| m.deinit();
    for (&mm) |*m| {
        const k = m.kernel() orelse {
            std.debug.print("N/A  {s}: cluster path needs sm_90 or newer, this GPU is sm_{d}\n", .{ mm_k.spec(m.kind).name, cap });
            continue;
        };
        if (cap < 90 and (m.kind == .lane_gemv or m.kind == .lane_gemv_split))
            std.debug.print("N/A  {s} replacement bits: cluster baseline needs sm_90; gate 3 uses the float64 bound\n", .{k.spec.name});
        try kernels.append(a, k);
    }
    var verify_controls: [6]mm_k.Mm = undefined;
    for ([_]mm_k.Kind{ .verify32_bits, .verify32_fp64, .verify64_bits, .verify64_fp64, .verify128_bits, .verify128_fp64 }, &verify_controls) |kind, *m| {
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .sm = cap, .affine4 = &affine4, .truncate_verify = true };
    }
    defer for (&verify_controls) |*m| m.deinit();
    for (&verify_controls) |*m| if (m.kernel()) |k| try kernels.append(a, k);
    var mx = [_]mx_k.Product{
        .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .clusters = true },
        .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane_reduce, .clusters = false },
    };
    defer for (&mx) |*m| m.deinit();
    for (&mx) |*m| try kernels.append(a, m.kernel());
    var half_lane = lane;
    var halves: [6]half_k.Half = undefined;
    for (std.enums.values(half_k.Kind), 0..) |kind, i| for ([_]cuda.qmmf.Out{ .bf16, .f32 }, 0..) |format, j| {
        halves[i * 2 + j] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &half_lane, .kind = kind, .format = format };
    };
    defer for (&halves) |*h| h.deinit();
    for (&halves) |*h| try kernels.append(a, h.kernel());
    var wide: [wide_k.modes.len * 2]wide_k.Product = undefined;
    for (wide_k.modes, 0..) |mode, i| {
        wide[i * 2] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .mode = mode, .clusters = true };
        wide[i * 2 + 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane_reduce, .mode = mode, .clusters = false };
    }
    defer for (&wide) |*m| m.deinit();
    for (&wide) |*m| try kernels.append(a, m.kernel());
    var narrow: [wide_k.bf16_modes.len * 2 + 1]wide_k.Product = undefined; // bf16 out (the 27B's ModelOpt layers)
    for (wide_k.bf16_modes, 0..) |mode, i| {
        narrow[i * 2] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .mode = mode, .clusters = true, .out_kind = .bf16 };
        narrow[i * 2 + 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane_reduce, .mode = mode, .clusters = false, .out_kind = .bf16 };
    }
    narrow[narrow.len - 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .mode = .fp8, .clusters = true, .out_kind = .bf16, .prompt = &prompt };
    defer for (&narrow) |*m| m.deinit();
    for (&narrow) |*m| try kernels.append(a, m.kernel());
    var rank = rank_k.Rank{ .rig = .{ .gpu = gpu, .s = s }, .sum = &rank_sum };
    defer rank.deinit();
    try kernels.append(a, rank.kernel());
    var hc = try cuda.Module.load(gpu.d, kk.flashnext_hc);
    defer hc.unload();
    var writebacks: [3]writeback_k.Writeback = undefined;
    for (std.enums.values(writeback_k.Mode), &writebacks) |mode, *w| {
        const symbol: [:0]const u8 = switch (mode) {
            .sums => "fn_hc_writeback0",
            .moe => "fn_hc_writeback_moe",
            .slices => "fn_hc_writeback_slices",
        };
        w.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try hc.function(symbol), .mode = mode };
    }
    defer for (&writebacks) |*w| w.deinit();
    for (&writebacks) |*w| try kernels.append(a, w.kernel());
    var hcnorms: [2]hcnorm_k.Normalize = undefined;
    for (std.enums.values(hcnorm_k.Kind), &hcnorms) |kind, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .cols_fn = try hc.function("fn_hc_normalize_cols"), .old_fn = try hc.function("fn_hc_normalize") };
    defer for (&hcnorms) |*x| x.deinit();
    for (&hcnorms) |*x| try kernels.append(a, x.kernel());
    var hcmixes: [4]hcmix_k.Mix = undefined;
    for (std.enums.values(hcmix_k.Kind), &hcmixes) |kind, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .new_fn = try hc.function("fn_hc_dense_mix8"), .old_fn = try hc.function("fn_hc_dense_mix"), .ctl_fn = if (kind == .sums_control) try mods[5].function("tf_ctl_hc_mix8_bf16") else null };
    defer for (&hcmixes) |*x| x.deinit();
    for (&hcmixes) |*x| try kernels.append(a, x.kernel());
    var batch = try cuda.Module.load(gpu.d, kk.flashnext_batch);
    defer batch.unload();
    var sums: [2]utility_k.Sum = undefined;
    var shifts: [2]utility_k.Shift = undefined;
    for ([_]bool{ false, true }, &sums, &shifts) |variant, *sum, *shift| {
        sum.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_sum_slices"), .bf16 = variant };
        shift.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_shift_window"), .padded = variant };
    }
    defer for (&sums) |*sum| sum.deinit();
    defer for (&shifts) |*shift| shift.deinit();
    for (&sums) |*sum| try kernels.append(a, sum.kernel());
    for (&shifts) |*shift| try kernels.append(a, shift.kernel());
    var mtp = try cuda.Module.load(gpu.d, kk.flashnext_mtp);
    defer mtp.unload();
    var best = best_k.Best{ .rig = .{ .gpu = gpu, .s = s }, .f = try mtp.function("fn_row_best") };
    defer best.deinit();
    try kernels.append(a, best.kernel());
    var selections: [2]qsa_k.Select = undefined;
    for ([_]bool{ false, true }, &selections) |padded, *selection|
        selection.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_qsa_select"), .padded = padded };
    defer for (&selections) |*selection| selection.deinit();
    for (&selections) |*selection| try kernels.append(a, selection.kernel());
    var h3_mod = try cuda.Module.load(gpu.d, &h3_image.bytes);
    defer h3_mod.unload();
    const h3_which = comptime std.enums.values(h3_k.Which);
    var h3_list: [h3_which.len]h3_k.H3 = undefined;
    for (h3_which, &h3_list) |w, *m| {
        const f = try h3_mod.function(h3_k.symbol(w));
        if (w == .attention) try f.allowDynamicShared(h3_prep.attention_shared);
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = f, .arena = std.heap.ArenaAllocator.init(a) };
    }
    defer for (&h3_list) |*m| m.deinit();
    for (&h3_list) |*m| try kernels.append(a, m.kernel());
    var h3_topk = h3_topk_k.TopK{ .rig = .{ .gpu = gpu, .s = s }, .f = try h3_mod.function("h3_topk") };
    defer h3_topk.deinit();
    try kernels.append(a, h3_topk.kernel());
    var d3_mod = try cuda.Module.load(gpu.d, kk.drama3);
    defer d3_mod.unload();
    var d3_norm = try drama3_k.Norm.init(gpu, s, d3_mod);
    defer d3_norm.deinit();
    try kernels.append(a, d3_norm.kernel());
    var ck: ?cuda.ck.Ck = null;
    var ck_reduce: ?cuda.ck.Ck = null; // no clusters: the FP8 slices meet in tf_ck8_reduce
    if (cap >= 120) {
        ck = try cuda.ck.Ck.load(gpu.d, cap / 10, sms);
        ck_reduce = try cuda.ck.Ck.load(gpu.d, 8, sms);
    } else std.debug.print("N/A  ck experts: the block-scaled FP4 mma needs sm_120a, this GPU is sm_{d}\n", .{cap});
    defer if (ck) |*q| q.unload();
    defer if (ck_reduce) |*q| q.unload();
    var packer = try cuda.modelopt.Packer.load(gpu.d);
    defer packer.unload();
    const ck_kinds = comptime std.enums.values(ck_k.Kind);
    var ck_experts: [ck_kinds.len]ck_k.Experts = undefined;
    var lanes: [2]lane8_k.Lane = undefined;
    var nck: usize = 0;
    var nlane: usize = 0;
    defer for (ck_experts[0..nck]) |*x| x.deinit();
    defer for (lanes[0..nlane]) |*x| x.deinit();
    if (ck) |*q| {
        for (ck_kinds) |kind| {
            ck_experts[nck] = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .ck = q, .router = try cuda.grouped.Router.resolve(mods[6]), .latent = &latent };
            try kernels.append(a, ck_experts[nck].kernel());
            nck += 1;
        }
        for ([_]lane8_k.Kind{ .cluster, .reduce }, [_]*const cuda.ck.Ck{ q, &ck_reduce.? }) |kind, by| {
            lanes[nlane] = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .ck = by, .packer = &packer };
            try kernels.append(a, lanes[nlane].kernel());
            nlane += 1;
        }
    }
    var e3_mods = try e3_k.Loaded.init(gpu);
    defer e3_mods.deinit();
    var e3_splits: [std.enums.values(e3_k.Fault).len]e3_k.Split = undefined;
    for (std.enums.values(e3_k.Fault), &e3_splits) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &e3_mods.k, .fault = f };
    defer for (&e3_splits) |*x| x.deinit();
    for (&e3_splits) |*x| try kernels.append(a, x.kernel());
    var nvfp4 = try cuda.experts.Nvfp4Experts.load(gpu.d, sms);
    defer nvfp4.unload();
    var nvfp4_facts: [3]nvfp4_fact_k.Experts = undefined;
    for (std.enums.values(nvfp4_fact_k.Kind), &nvfp4_facts) |kind, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .k = &nvfp4, .router = try cuda.grouped.Router.resolve(mods[6]) };
    defer for (&nvfp4_facts) |*x| x.deinit();
    for (&nvfp4_facts) |*x| try kernels.append(a, x.kernel());
    var round_mod = try cuda.Module.load(gpu.d, kk.tree_round);
    defer round_mod.unload();
    const round_fns = try cuda.tree_round.Kernels.resolve(round_mod);
    var rounds: [std.enums.values(round_k.Which).len]round_k.Round = undefined;
    for (std.enums.values(round_k.Which), &rounds) |w, *r| r.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .k = &round_fns };
    defer for (&rounds) |*r| r.deinit();
    for (&rounds) |*r| try kernels.append(a, r.kernel());
    var act_rows = try cuda.act_rows.ActRows.load(gpu.d);
    defer act_rows.unload();
    var acts: [std.enums.values(act_k.Which).len]act_k.Act = undefined;
    for (std.enums.values(act_k.Which), &acts) |w, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .k = &act_rows, .arena = .init(a) };
    defer for (&acts) |*x| x.deinit();
    for (&acts) |*x| try kernels.append(a, x.kernel());
    var act_rounds: [act_round_k.entries.len]act_round_k.Act = undefined;
    for (act_round_k.entries, &act_rounds) |e, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .act = &act_rows, .ctl = if (e.keep != null) try mods[5].function("tf_ctl_act_round") else null };
    defer for (&act_rounds) |*x| x.deinit();
    for (&act_rounds) |*x| try kernels.append(a, x.kernel());
    var gemma_ops: @import("contract_gemma_ops.zig").Group = .{};
    defer gemma_ops.deinit();
    try gemma_ops.init(gpu, s, &act_rows, &vit, &round_fns, mods[5]);
    try gemma_ops.append(a, &kernels);
    var vit_g4 = try vit_g4_group.Group.init(gpu, s, &vit, mods[5]);
    defer vit_g4.deinit();
    try vit_g4.append(a, &kernels);
    var gdn_prep = try cuda.gdn_prep.GdnPrep.load(gpu.d);
    defer gdn_prep.unload();
    var gdns: [9]gdn_k.Gdn = undefined;
    for (&gdns, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .dk = @as(usize, 64) << @intCast(i / 3), .part = @fromBackingInt(@intCast(i % 3)), .k = &gdn_prep, .arena = .init(a) };
    defer for (&gdns) |*x| x.deinit();
    for (&gdns) |*x| try kernels.append(a, x.kernel());
    var qk_rows = try cuda.qk_rows.QkRows.load(gpu.d);
    defer qk_rows.unload();
    var qks: [qk_k.forms.len]qk_k.Qk = undefined;
    for (&qks, qk_k.forms) |*x, f| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &qk_rows, .arena = .init(a) };
    defer for (&qks) |*x| x.deinit();
    for (&qks) |*x| try kernels.append(a, x.kernel());
    var tree_paths = try cuda.tree_attention.TreeAttention.load(gpu.d);
    defer tree_paths.unload();
    var paths = tree_k.Paths{ .rig = .{ .gpu = gpu, .s = s }, .ta = &tree_paths };
    defer paths.deinit();
    try kernels.append(a, paths.kernel());
    var trees: [tree_k.variants.len]tree_k.Tree = undefined;
    var ntrees: usize = 0;
    defer for (trees[0..ntrees]) |*x| {
        x.ta.unload();
        x.deinit();
    };
    for (tree_k.variants) |v| {
        trees[ntrees] = .{ .rig = .{ .gpu = gpu, .s = s }, .v = v, .ta = try cuda.tree_attention.TreeAttention.load(gpu.d) };
        ntrees += 1;
    }
    for (&trees) |*x| try kernels.append(a, x.kernel());
    var block_attn = try cuda.block_attention.BlockAttention.load(gpu.d);
    defer block_attn.unload();
    var blocks: [2]block_k.Block = .{ .{ .rig = .{ .gpu = gpu, .s = s }, .ba = &block_attn, .causal = false }, .{ .rig = .{ .gpu = gpu, .s = s }, .ba = &block_attn, .causal = true } };
    defer for (&blocks) |*x| x.deinit();
    for (&blocks) |*x| try kernels.append(a, x.kernel());
    var q35_mod = try cuda.Module.load(gpu.d, kk.qwen35);
    defer q35_mod.unload();
    var q35_kv8 = try cuda.Module.load(gpu.d, kk.qwen35_kv8);
    defer q35_kv8.unload();
    var q35s: [q35_k.forms.len]q35_k.Q35 = undefined;
    for (q35_k.forms, &q35s) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .partial = try if (f.cache == .fp8) q35_kv8.function("tf_q35_attn_partial8") else q35_mod.function("tf_q35_attn_partial"), .merge = try q35_mod.function("tf_q35_attn_merge") };
    defer for (&q35s) |*x| x.deinit();
    for (&q35s) |*x| try kernels.append(a, x.kernel());
    var window_fns = try cuda.window_attention.WindowAttention.load(gpu.d);
    defer window_fns.unload();
    var windows: [window_k.cfgs.len]window_k.Window = undefined;
    for (window_k.cfgs, &windows) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .g = &window_fns };
    defer for (&windows) |*x| x.deinit();
    for (&windows) |*x| try kernels.append(a, x.kernel());
    var moe_fns = try cuda.moe_glue.MoeGlue.load(gpu.d);
    defer moe_fns.unload();
    var topks: [moe_k.routes.len]moe_k.TopK = undefined;
    for (moe_k.routes, &topks) |r, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .route = r, .g = &moe_fns };
    defer for (&topks) |*x| x.deinit();
    for (&topks) |*x| try kernels.append(a, x.kernel());
    var combines: [moe_k.combos.len]moe_k.Combine = undefined;
    for (moe_k.combos, &combines) |m, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .x = m, .g = &moe_fns };
    defer for (&combines) |*x| x.deinit();
    for (&combines) |*x| try kernels.append(a, x.kernel());
    var dense_fns = try cuda.dense_rows.DenseRows.load(gpu.d);
    defer dense_fns.unload();
    var denses: [moe_k.dense_tiles.len]moe_k.DenseRows = undefined;
    for (&denses, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .tile = i, .g = &dense_fns };
    defer for (&denses) |*x| x.deinit();
    for (&denses) |*x| try kernels.append(a, x.kernel());
    var embed_rows = try cuda.embed_rows.EmbedRows.load(gpu.d);
    defer embed_rows.unload();
    var embeds: [embed_k.entries.len]embed_k.Embed = undefined;
    for (embed_k.entries, &embeds) |e, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .e = &embed_rows, .ctl = if (e.keep != null) try mods[5].function("tf_ctl_embed_affine") else null };
    defer for (&embeds) |*x| x.deinit();
    for (&embeds) |*x| try kernels.append(a, x.kernel());
    var qkrs: [qkr_k.forms.len]qkr_k.Qk = undefined;
    for (&qkrs, qkr_k.forms) |*x, f| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &qk_rows, .ctl = if (f.keep != null) try mods[5].function("tf_ctl_qk_round") else null, .arena = .init(a) };
    defer for (&qkrs) |*x| x.deinit();
    for (&qkrs) |*x| try kernels.append(a, x.kernel());
    const fparts = @import("flashnext").batch_state;
    const ftk = @import("flashnext").tile_kernels;
    var qmm_mod = try cuda.Module.load(gpu.d, kk.qmm);
    defer qmm_mod.unload();
    var fact_mod = try cuda.Module.load(gpu.d, kk.flashnext_fact);
    defer fact_mod.unload();
    const flane = try qmm_mod.function(fparts.cubin_qmm);
    try flane.allowDynamicShared(fparts.qmm_smem);
    const fprompt = try fact_mod.function(ftk.cubin_fact);
    try fprompt.allowDynamicShared(ftk.fact_smem);
    const ffold = try fact_mod.function(ftk.cubin_fact_fold);
    try ffold.allowDynamicShared(ftk.fact_smem);
    var facts: [2]fact_k.Fact = undefined;
    for (std.enums.values(fact_k.Kind), &facts) |kind, *m| m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .lane = flane, .prompt = fprompt, .fold = ffold, .sum = try batch.function("fn_sum_slices") };
    defer for (&facts) |*m| m.deinit();
    for (&facts) |*m| try kernels.append(a, m.kernel());
    var route = try cuda.route_rows.RouteRows.load(gpu.d);
    defer route.unload();
    var routes: [3]route_k.Logits = undefined;
    for (std.enums.values(route_k.Kind), &routes) |kind, *r| r.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .route = &route };
    defer for (&routes) |*r| r.deinit();
    for (&routes) |*r| try kernels.append(a, r.kernel());
    var rows_mod = try cuda.Module.load(gpu.d, kk.experts_rows);
    defer rows_mod.unload();
    var fexp_mod = try cuda.Module.load(gpu.d, kk.flashnext_fexp);
    defer fexp_mod.unload();
    var pack_mod = try cuda.Module.load(gpu.d, kk.experts_pack);
    defer pack_mod.unload();
    const fexp_fns: fexp_k.Fns = .{
        .up = try fexp_mod.function(ftk.cubin_fexp_up),
        .down = try fexp_mod.function(ftk.cubin_fexp_down),
        .rows_up = try rows_mod.function(cuda.experts.RowsExperts.symbols.up),
        .rows_down = try rows_mod.function(cuda.experts.RowsExperts.symbols.down),
        .pack = try pack_mod.function(fexp_k.pack_symbol),
    };
    try fexp_fns.up.allowDynamicShared(ftk.fexp_smem);
    try fexp_fns.down.allowDynamicShared(ftk.fexp_smem);
    var fexps: [2]fexp_k.Fexp = undefined;
    for (std.enums.values(fexp_k.Kind), &fexps) |kind, *m| m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = &fexp_fns };
    defer for (&fexps) |*m| m.deinit();
    for (&fexps) |*m| try kernels.append(a, m.kernel());
    var attn = try cuda.Module.load(gpu.d, kk.flashnext_attn);
    defer attn.unload();
    var hcf_mod = try cuda.Module.load(gpu.d, kk.flashnext_hcf);
    defer hcf_mod.unload();
    var fglue = try @import("flashnext").Glue.load(gpu.d);
    defer fglue.deinit();
    const glue_fns: glue_k.Fns = .{
        .tpr = try batch.function("fn_gdn_chain_tpr"),
        .rows_to = try batch.function("fn_gdn_rows_to"),
        .tile = try attn.function("_ZN8qsa_tile6scoresILi8ELi4EEEvPK13__nv_bfloat16S3_PKiPfiiii"),
        .scores = try attn.function("fn_qsa_scores"),
        .hcf_down = try hcf_mod.function(ftk.cubin_hcf_down),
        .hcf_up = try hcf_mod.function(ftk.cubin_hcf_up),
        .front = try batch.function("fn_gdn_front"),
        .norm = try batch.function("fn_gdn_norm"),
        .fused = try batch.function("fn_gdn_fused"),
    };
    try glue_fns.fused.allowDynamicShared(fparts.gdn_fused_smem);
    try glue_fns.tpr.allowDynamicShared(fparts.chain_tpr_smem);
    try glue_fns.hcf_down.allowDynamicShared(ftk.hcf_down_smem);
    try glue_fns.hcf_up.allowDynamicShared(ftk.hcf_up_smem);
    var glues: [4]glue_k.Glue = undefined;
    for (std.enums.values(glue_k.Kind), &glues) |kind, *m| m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = &glue_fns, .glue = &fglue };
    defer for (&glues) |*m| m.deinit();
    for (&glues) |*m| try kernels.append(a, m.kernel());
    var wrows: [wrows_k.cfgs.len]wrows_k.Rows = undefined;
    for (&wrows, wrows_k.cfgs) |*x, f| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .g = &window_fns, .ctl = mods[5], .arena = .init(a) };
    defer for (&wrows) |*x| x.deinit();
    for (&wrows) |*x| try kernels.append(a, x.kernel());
    var glm_glue = try cuda.Module.load(gpu.d, kk.glm_glue);
    defer glm_glue.unload();
    var f32ws: [6]f32w_k.F32w = undefined;
    for (std.enums.values(f32w_k.Which), &f32ws) |w, *x| x.* = .init(.{ .gpu = gpu, .s = s }, w, try glm_glue.function(switch (w) {
        .norm_y, .norm_sums => "tf_glm_rmsnorm_4096_f32w",
        else => "tf_glm_hc_finish_f32w",
    }));
    defer for (&f32ws) |*x| x.deinit();
    for (&f32ws) |*x| try kernels.append(a, x.kernel());
    var bf16s: [f32w_k.bf16_entries.len]f32w_k.F32w = undefined;
    for (f32w_k.bf16_entries, &bf16s) |e, *x| {
        x.* = .init(.{ .gpu = gpu, .s = s }, e.which, try glm_glue.function(e.symbol));
        x.width = e.width;
        x.baseline = try mods[5].function(e.old);
    }
    defer for (&bf16s) |*x| x.deinit();
    for (&bf16s) |*x| try kernels.append(a, x.kernel());
    var f32w_control: f32w_k.F32w = .init(.{ .gpu = gpu, .s = s }, .norm_y, try glm_glue.function("tf_glm_rmsnorm_4096_f32w"));
    f32w_control.bf16_gamma = true;
    defer f32w_control.deinit();
    try kernels.append(a, f32w_control.kernel());
    const glm_fns = try glm.glue.Kernels.load(glm_glue);
    var glm_lat = glm_pages_k.LatWrite{ .rig = .{ .gpu = gpu, .s = s }, .f = glm_fns.lat_write };
    defer glm_lat.deinit();
    try kernels.append(a, glm_lat.kernel());
    var glm_pool = glm_pages_k.PoolKeys{ .rig = .{ .gpu = gpu, .s = s }, .f = glm_fns.pool_keys };
    defer glm_pool.deinit();
    try kernels.append(a, glm_pool.kernel());
    var glm_scores = glm_index_k.Scores{ .rig = .{ .gpu = gpu, .s = s }, .f = glm_fns.scores };
    defer glm_scores.deinit();
    try kernels.append(a, glm_scores.kernel());
    var glm_ops = try cuda.Module.load(gpu.d, kk.glm_ops);
    defer glm_ops.unload();
    var glm_tokens = glm_index_k.Tokens{ .rig = .{ .gpu = gpu, .s = s }, .f = try glm_ops.function("tf_glm_dsa_tokens") };
    defer glm_tokens.deinit();
    try kernels.append(a, glm_tokens.kernel());
    var glm_chunks: [4]glm_attn_k.Chunks = undefined;
    for (&glm_chunks, [_]u32{ 16, 16, 32, 32 }, [_]bool{ false, true, false, true }) |*x, hb, sparse|
        x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = if (hb == 16) glm_fns.chunks_16 else glm_fns.chunks_32, .merge = glm_fns.merge, .form = .{ .hb = hb, .sparse = sparse } };
    defer for (&glm_chunks) |*x| x.deinit();
    for (&glm_chunks) |*x| try kernels.append(a, x.kernel());
    var glm_controls = @import("contract_glm_controls.zig").Controls.init(gpu, s, glm_fns, glm_tokens.f);
    defer glm_controls.deinit();
    try glm_controls.append(a, &kernels);
    var t: g.Tally = .{};
    try @import("contract_dump.zig").run(a, kernels.items);
    const t0 = check.now(gpu.io);
    try g.runAll(a, kernels.items, 0x6b63, &t);
    try @import("gdn_back_tests.zig").run(gpu);
    try @import("q4_back_tests.zig").run(gpu);
    try @import("moe_back_tests.zig").run(gpu);
    try @import("attn_back_tests.zig").run(gpu);
    try @import("fn_hc_back_tests.zig").run(gpu);
    try @import("fn_moe_back_tests.zig").run(gpu);
    try @import("fn_pocket_tests.zig").run(gpu, 0);
    std.debug.print("PASS gradient component suites (FP64 and finite-difference checks); resource receipts still required\n", .{});
    const t1 = check.now(gpu.io);
    try @import("contract_qwen_served.zig").run(gpu);
    try @import("contract_qwen_learning.zig").numerical(gpu);
    const lim = try live(gpu);
    const t2 = check.now(gpu.io);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}

/// The named embedded image and gate 5: in the contract's own file, so the gates' registrations stay short.
pub const image = @import("contract_limits.zig").image;
pub const live = @import("contract_limits.zig").live;
pub const fatbins = @import("contract_limits.zig").fatbins;
