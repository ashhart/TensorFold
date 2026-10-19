//! tf-cuda-test kernel-contract: the registered CUDA kernels through gates 1-4, the planted controls, and gate 5.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const options = @import("contract_options");
const check = @import("check.zig");
const rows_k = @import("contract_rows.zig");
const router_k = @import("contract_router.zig");
const mm_k = @import("contract_mm.zig");
const mx_k = @import("contract_mxfp8.zig");
const wide_k = @import("contract_projection_f32.zig");
const rank_k = @import("contract_rank.zig");
const writeback_k = @import("contract_flashnext_writeback.zig");
const utility_k = @import("contract_utility.zig");
const best_k = @import("contract_row_best.zig");
const qsa_k = @import("contract_qsa_select.zig");
const h3_k = @import("contract_h3.zig");
const h3_prep = @import("contract_h3_prep.zig");
const h3_topk_k = @import("contract_h3_topk.zig");
const drama3_k = @import("contract_drama3.zig");
const ck_k = @import("contract_ck.zig");
const scan_k = @import("contract_scan.zig");
const lane8_k = @import("contract_ck_lane.zig");
const latent_k = @import("contract_latent.zig");

const round_k = @import("contract_tree_round.zig");

const act_k = @import("contract_act.zig");
const gdn_k = @import("contract_gdn.zig");
const qk_k = @import("contract_qk.zig");

const tree_k = @import("contract_tree_attn.zig");
const block_k = @import("contract_block_attn.zig");

const contract = core.contract;
const g = contract.gates;
const limits = contract.limits;
const registry = contract.registry;
const kk = cuda.kernels;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const h3_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3").*;
};

/// The named embedded image; only contract_controls may select the planted controls.
pub fn image(name: []const u8) error{UnknownContractImage}![]const u8 {
    if (@import("contract_deepseek_images.zig").image(name)) |bytes| return bytes;
    if (std.mem.eql(u8, name, "h3")) return &h3_image.bytes;
    inline for (.{ "tree_round", "act_rows", "gdn_prep", "qk_rows", "norms", "moe", "qwen36_mtp", "nemotron_attention", "nemotron_route", "nemotron_latent", "qmm_group", "lane_gemv", "prompt16", "qmmf", "ck_experts", "rank_sum", "prefill_attention", "vit", "embed_rows", "route_rows", "flashnext_gdn", "flashnext_batch", "flashnext_hc", "flashnext_attn", "flashnext_mtp", "qwen_vision", "exl3_linear", "exl3_experts", "exl3_candidate", "experts3", "experts3_prefill", "drama3", "drama3_fp8", "drama3_codec", "glm_mm", "glm_dflash", "glm_glue", "nemotron_mamba", "sample", "qwen35", "qwen35_keep", "qwen35_kv8", "qwen35_draft", "ck_lane", "gdn_train", "train_q4", "train_attn", "flashnext_train", "train", "attn_train", "qwen27_train", "tree_attention", "block_attention" }) |n| if (std.mem.eql(u8, name, n)) return @field(kk, n);
    if (std.mem.eql(u8, name, "contract_controls")) return if (options.controls) &controls.bytes else &.{};
    std.debug.print("FAIL kernel-contract: unknown image name: {s}\n", .{name});
    return error.UnknownContractImage;
}

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
    var split = try cuda.qlinear.Split.init(gpu.d); // GB10 serves without it; other GPUs take it for K slices
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
    var row_list: [7]rows_k.Row = undefined;
    var nrows: usize = 0;
    defer for (row_list[0..nrows]) |*r| r.deinit();
    for (std.enums.values(rows_k.Which)) |w| {
        row_list[nrows] = try rows_k.Row.init(gpu, s, &norms, mods[5], w);
        nrows += 1;
    }
    for (row_list[0..nrows]) |*r| try kernels.append(a, r.kernel());
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
    var verify_controls: [2]mm_k.Mm = undefined;
    for ([_]mm_k.Kind{ .verify32_bits, .verify32_fp64 }, &verify_controls) |kind, *m| {
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
    var wide: [wide_k.modes.len * 2]wide_k.Product = undefined;
    for (wide_k.modes, 0..) |mode, i| {
        wide[i * 2] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .mode = mode, .clusters = true };
        wide[i * 2 + 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane_reduce, .mode = mode, .clusters = false };
    }
    defer for (&wide) |*m| m.deinit();
    for (&wide) |*m| try kernels.append(a, m.kernel());
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
    var t: g.Tally = .{};
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

/// Gate 5 on this GPU: the table's row for it against the driver, each record against the loaded function.
pub fn live(gpu: check.Gpu) !usize {
    var n = try registry.checkCuda(gpu.gpa, &registry.all_cuda, false);
    std.debug.print("{s} limits: {d} registered CUDA launches against sm_80/86/89/120/121, {d} problems\n", .{ if (n == 0) "PASS" else "FAIL", registry.all_cuda.len, n });
    const sm: u32 = @intCast(try gpu.ctx.capability());
    const at = for (limits.sms, 0..) |t, i| {
        if (t.sm == sm) break i;
    } else {
        std.debug.print("N/A  limits live: sm_{d} is not a target the records cover\n", .{sm});
        return n;
    };
    const t = limits.sms[at];
    const dev = [_]struct { []const u8, cuda.abi.DeviceAttribute, u32 }{
        .{ "most shared bytes a block (opt-in)", .max_shared_memory_per_block_optin, t.smem_optin },
        .{ "threads a block", .max_threads_per_block, t.threads },
    };
    for (dev) |d| {
        const got: u32 = @intCast(try gpu.ctx.attribute(d[1]));
        if (got != d[2]) n += 1;
        std.debug.print("{s} limits live sm_{d} {s}: the driver says {d}, the table {d}\n", .{ if (got == d[2]) "PASS" else "FAIL", sm, d[0], got, d[2] });
    }
    var said: ?limits.Nvcc = null;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        if (!control and registry.targetDecision(r, t, r.usage[at]) != .applicable) continue;
        const bytes = try image(r.launch.image);
        const rel = try built(gpu.gpa, bytes, sm);
        const set = registry.setFor(rel);
        if (said == null or !std.meta.eql(said.?, set)) records("live", rel, set);
        said = set;
        const want = registry.usageFor(r, rel)[at];
        var mod = try cuda.Module.load(gpu.d, bytes);
        defer mod.unload();
        const sym = try gpu.gpa.dupeSentinel(u8, r.launch.symbol, 0);
        defer gpu.gpa.free(sym);
        const f = try mod.function(sym);
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const static: u32 = @intCast(try f.attribute(.shared_size_bytes));
        const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
        const fits = r.launch.threads <= most and static + r.launch.dynamic <= t.smem_optin;
        const current = want.regs == regs and want.static == static;
        if (!control and !(fits and current)) n += 1;
        if (control and fits) n += 1;
        const word = if (control) (if (fits) "MISSED" else "RED ") else if (fits and current) "PASS" else "FAIL";
        std.debug.print("{s} limits live {s}: {d} threads launched, the driver allows {d}; {d} registers, {d} + {d} shared bytes (recorded {d}, {d})\n", .{ word, r.launch.name, r.launch.threads, most, regs, static, r.launch.dynamic, want.regs, want.static });
    }
    return n;
}

/// The nvcc release that built a fatbin's image for `sm`, else the one its images agree on (null: none names one).
fn built(a: std.mem.Allocator, bytes: []const u8, sm: u32) !?limits.Nvcc {
    return contract.cubin.releaseFor(a, bytes, sm);
}

/// Says which compiler's records an audit compares against, and why.
fn records(what: []const u8, rel: ?limits.Nvcc, set: limits.Nvcc) void {
    if (rel) |v| {
        const own = std.meta.eql(v, set);
        std.debug.print("INFO limits {s}: images built by nvcc {d}.{d}, checked against nvcc {d}.{d}'s records{s}\n", .{ what, v.major, v.minor, set.major, set.minor, if (own) "" else " (the newest set at or below it)" });
    } else std.debug.print("INFO limits {s}: the images name no nvcc release; checked against nvcc {d}.{d}'s records\n", .{ what, set.major, set.minor });
}

/// tf-cuda-test kernel-contract-limits DIR: each record against the fatbins in DIR, every SM image they hold.
pub fn fatbins(gpu: check.Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    var n: usize = 0;
    var said: ?limits.Nvcc = null;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        const any_target = for (limits.sms, 0..) |target, index| {
            if (registry.targetDecision(r, target, r.usage[index]) == .applicable) break true;
        } else false;
        if (!control and !any_target) {
            std.debug.print("N/A limits {s}: no applicable target\n", .{r.launch.name});
            continue;
        }
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, r.launch.image });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
        defer a.free(bytes);
        const imgs = try contract.cubin.images(a, bytes);
        defer contract.cubin.freeImages(a, imgs);
        var red: usize = 0;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (limits.sms, 0..) |t, i| {
            const decision = registry.targetDecision(r, t, r.usage[i]);
            if (!control and decision != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, decision });
                continue;
            }
            const img = for (imgs) |im| {
                if (im.sm == t.sm) break im;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no sm_{d} image (build with -Dsm=80,86,89,120,121)\n", .{ r.launch.name, path, t.sm });
                n += 1;
                continue;
            };
            const ks = try contract.cubin.kernels(a, img.elf, t.sm);
            defer a.free(ks);
            const k = for (ks) |kv| {
                if (std.mem.eql(u8, kv.name, r.launch.symbol)) break kv;
            } else return error.SymbolNotInImage;
            const rel = try contract.cubin.release(a, img.elf);
            const set = registry.setFor(rel);
            if (said == null or !std.meta.eql(said.?, set)) records("fatbins", rel, set);
            said = set;
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(a);
            const measured = registry.targetDecision(r, t, k.usage);
            if (!control and measured != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, measured });
                continue;
            }
            red += try limits.cuda(r.launch, k.usage, t, &out, a);
            const current = std.meta.eql(k.usage, registry.usageFor(r, rel)[i]);
            if (!control and !current) n += 1;
            std.debug.print("{s} limits sm_{d} {s}: {d} registers, {d} static shared bytes, bound {d} (record {s}){s}", .{ if (out.items.len > 0) (if (control) "RED " else "FAIL") else "PASS", t.sm, r.launch.name, k.usage.regs, k.usage.static, k.usage.bound, if (current) "current" else "stale", if (out.items.len > 0) ": " else "\n" });
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ k.usage.regs, k.usage.static, k.usage.bound });
        }
        std.debug.print("RECORD {s}: .usage = .{{ {s}}}\n", .{ r.launch.name, line.items });
        if (control and red == 0) n += 1;
        if (!control) n += red;
    }
    try check.expect(n == 0, "kernel-contract-limits: {d} problems", .{n});
    check.pass("kernel-contract-limits: every registered launch fits sm_80, sm_86, sm_89, sm_120 and sm_121, every control red", .{});
}
