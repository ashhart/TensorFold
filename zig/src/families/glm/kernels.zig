//! GLM-5.3-Flash pipelines compiled at runtime: the Python family's kernels (generated), our glue and latent attention.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const moe_route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const mm_groups = @import("mm_groups.zig");
const hc = @import("../../core/hc.zig");
const wide = @import("wide.zig");

/// The route's shape (config.zig refuses other checkpoints).
pub const route_shape: moe_route.Shape = .{ .hidden = 4096, .experts = 288, .topk = 8 };
/// The MLA indexer's key gate as the same gemv_t (x [rows, 4096] times the packed [128, 4096] gate, bf16 out).
pub const igate_shape: moe_route.Shape = .{ .hidden = 4096, .experts = 128, .topk = 8, .out_bf16 = true };

/// The hyper-connection boundary's shape (config.zig refuses other checkpoints).
pub const hc_shape: hc.Shape = .{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000, .topk = route_shape.topk };

pub const max_rows = 16;
pub const MtpPrepArgs = extern struct { rows: u32, width: u32, eps: f32 };
pub const HeadTop1Args = extern struct { k: u32, n: u32, rows: u32, pairs: u32 };

pub const Kernels = struct {
    source_hash: u64, // every compiled source, hashed: part of a learned prompt state's identity
    qmv_kda_in: mtl.Pipeline,
    qmv_kda_out: mtl.Pipeline, // also the MTP's eh_proj (K 8192, N 4096)
    qmv_x: mtl.Pipeline,
    qmv_qr: mtl.Pipeline,
    qmv_mla_out: mtl.Pipeline,
    qmv_dense_gu: mtl.Pipeline,
    qmv_dense_down: mtl.Pipeline,
    qmv_head: mtl.Pipeline,
    qmv_kda_in_tp: mtl.Pipeline, // TP2: the KDA input projection's rows of one Mac's heads
    qmvp_kda_out: mtl.Pipeline, // TP2: the KDA out-projection over one Mac's heads, fp32 partials
    qmv_head_tp: mtl.Pipeline, // TP2: the head's rows of one Mac's half of the vocabulary
    qmv_qr_tp: mtl.Pipeline, // TP2: q_b's rows of one Mac's MLA heads, the indexer's queries whole
    qmvp_mla_out: mtl.Pipeline, // TP2: the MLA out-projection over one Mac's heads, fp32 partials
    sparse_attention_tp: mtl.Pipeline, // TP2: the sparse kernel over one Mac's 32 heads
    qmv_dense_gu_tp: mtl.Pipeline, // TP2: the dense MLP's gate and up rows of one Mac's half
    qmvp_dense_down: mtl.Pipeline, // TP2: its down projection over that half, fp32 partials
    gemv_t_igate: mtl.Pipeline,
    gemv_t_values: mtl.Pipeline,
    gemv_scores_lt4: mtl.Pipeline,
    gemv_scores_le32: mtl.Pipeline,
    gemv_scores: mtl.Pipeline,
    hc_expand_10: mtl.Pipeline,
    hc_core: hc.Pipes, // core/hc.zig's names: the expand (or first read) with partial sums, split, in place, parts apart, MoE
    kda_rows: mtl.Pipeline,
    kda_rows_tp: mtl.Pipeline, // TP2: the fused step over one Mac's 32 heads
    kda_window: mtl.Pipeline, // windows of 2+ rows: kda_rows' values, the rows' gates, convolutions and norms at once
    kda_window_tp: mtl.Pipeline,
    kda_parts: mtl.Pipeline, // all 64 heads on two threadgroups each (their value columns), raw outputs and gates
    kda_norm: mtl.Pipeline, // then each row's head normed and gated: kda_rows' values
    kda_split: bool = false, // 2+ row steps take kda_parts and kda_norm (an Ultra's GPU idles at 64 groups)
    router: [max_rows]mtl.Pipeline, // by window rows (RR = 1 .. 16)
    moe_route: mtl.Pipeline,
    shared_gateup: mtl.Pipeline, // the shared expert's kernels at one output a simdgroup: four times the threadgroups, the same sums
    shared_down: mtl.Pipeline,
    moe_gateup_0: mtl.Pipeline, // the routed experts and the shared one (the last slot) in one gather
    moe_down_0: mtl.Pipeline,
    moe_gateup_0_one: mtl.Pipeline, // the same at two outputs a simdgroup, for one-row rounds (more weight reads in flight)
    moe_down_0_one: mtl.Pipeline,
    moe_gateup_pick: mtl.Pipeline, // one row: each routed group replays the route select's picks, no select launch
    moe_down_pick: mtl.Pipeline, // and the down projections of the picks in pick order
    combine_0: mtl.Pipeline, // their combine: the shared expert's row read from the last slot
    moe_gateup_2: mtl.Pipeline,
    moe_down_2: mtl.Pipeline,
    moe_combine: mtl.Pipeline,
    moe_gateup_2h: mtl.Pipeline, // expert parallel by rows: half of every expert's gate/up rows
    moe_down_2h: mtl.Pipeline, // and its down inputs, fp32 partials
    sparse_attention: mtl.Pipeline,
    cast_f32: mtl.Pipeline,
    rms: mtl.Pipeline,
    layer_norm: mtl.Pipeline,
    absorb: mtl.Pipeline,
    unabsorb: mtl.Pipeline,
    scale: mtl.Pipeline,
    mla_norms: mtl.Pipeline, // an MLA layer's latent key norm, indexer key norm and indexer weights in one launch
    pool: mtl.Pipeline,
    stream_mean: mtl.Pipeline,
    tap_mean_ring: mtl.Pipeline,
    add: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    argmax: mtl.Pipeline,
    sample: mtl.Pipeline, // the shared tf_sample_full (nemotron_sample.metal): the head's keyed draw
    index_select: mtl.Pipeline,
    index_scores_nax: mtl.Pipeline, // a prompt chunk's index scores on the tensor units (glm_index_nax.metal)
    index_decode: mtl.Pipeline, // and decode rows' (a row's heads as the op's rows)
    select_mark: mtl.Pipeline, // decode rows' key lists past CAP blocks: the blocks at or above a sampled floor (glm_select.metal)
    select_merge: mtl.Pipeline, // and the selection among them, glm_index_select's bytes
    exp_f32: mtl.Pipeline,
    streams: mtl.Pipeline,
    copy_u32: mtl.Pipeline,
    route_rows: mtl.Pipeline,
    act2: mtl.Pipeline,
    dense_indices: mtl.Pipeline,
    softmax: mtl.Pipeline,
    embed: mtl.Pipeline,
    latent_scores: mtl.Pipeline,
    latent_values: mtl.Pipeline,
    latent_scores_rows: mtl.Pipeline, // a window's dense rows at once: each row the one-row kernels' bits
    latent_values_rows: mtl.Pipeline,
    softmax_rows: mtl.Pipeline,
    route_logits: mtl.Pipeline,
    route_select: mtl.Pipeline,
    igate_logits: mtl.Pipeline,
    route_logits_one: mtl.Pipeline, // both for a one-row window: row 0's sums alone (glm_route_one.metal)
    igate_logits_one: mtl.Pipeline,
    kda_prep: mtl.Pipeline, // a prompt chunk's KDA layer (glm_kda_prompt.metal): conv, norms, decay and beta
    kda_scan: mtl.Pipeline, // the recurrence alone
    kda_post: mtl.Pipeline, // the output norm and gate
    kda_prep_tp: mtl.Pipeline, // TP2: the three passes over one Mac's 32 heads
    kda_scan_tp: mtl.Pipeline,
    kda_post_tp: mtl.Pipeline,
    sparse_nax: mtl.Pipeline, // a prompt chunk's sparse MLA attention on the tensor units (glm_sparse_nax.metal)
    sparse_split: mtl.Pipeline, // decode rows' sparse MLA over key blocks on the tensor units
    sparse_combine: mtl.Pipeline, // and the blocks combined in order
    absorb_nax: mtl.Pipeline, // and its absorb (glm_absorb_nax.metal)
    mtp_prep_4b: mtl.Pipeline, // fused embed + enorm + hnorm (glm_mtp_opt.metal)
    mtp_prep_8b: mtl.Pipeline,
    head_top1_4b_one: mtl.Pipeline, // fused 4/64 greedy LM head + local top-1 + reduce
    head_top1_4b: mtl.Pipeline,
    reduce_top1: mtl.Pipeline,
    sparse_nax_tp: mtl.Pipeline, // TP2: both over one Mac's 32 heads
    absorb_nax_tp: mtl.Pipeline,
    mm_bf16: affine_mm.Pipes, // prompt chunks' 4-bit g64 matmuls on the tensor units (core/affine_mm.zig): dense, gathers
    mm_groups: mm_groups.Pipes, // the routed gathers walked in column groups (mm_groups.zig): mm_bf16's bits
    mm_f32: affine_mm.Pipes, // and with fp32 out: expert parallel by rows' down partials
    wide: ?wide.Pipes = null, // the products at other widths (wide.zig), for a checkpoint that has one
    m3_fast: bool, // pre-M5 Metal dispatch geometry (32x64 affine_mm, 4-sg KDA scan, 32x64/64 index NAX)

    pub fn deinit(k: *Kernels) void {
        const info = @typeInfo(Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Pipeline) @field(k, name).deinit() else if (T == ?wide.Pipes) {
                if (k.wide) |*w| w.deinit();
            } else if (T != u64 and T != bool) for (&@field(k, name)) |*p| p.deinit();
        }
    }

    /// The other widths' pipelines, once: their sources join the hash a learned state's identity reads.
    pub fn widen(k: *Kernels, gpa: std.mem.Allocator, device: mtl.Device) !void {
        if (k.wide != null) return;
        k.wide = try wide.load(gpa, device);
        k.source_hash ^= k.wide.?.hash;
    }

    /// TFQ3's pipelines, only for a checkpoint with TFQ3 files: its library, then the q8 KDA pair on main's KDA source.
    pub fn enableTfq3(k: *Kernels, gpa: std.mem.Allocator, device: mtl.Device) !void {
        try k.widen(gpa, device);
        const w = &k.wide.?;
        if (w.tfq3 != null) return;
        var t = try wide.tfq3.Pipes.init(gpa, device, wide.glue_math);
        errdefer t.deinit();
        const m3_def = if (k.m3_fast) "#define TF_GLM_M3_FAST 1\n" else "";
        const src = try std.mem.concat(gpa, u8, &.{ comptime kernelOf("kda_rows").source, m3_def, sources.glm_kda_prompt, sources.glm_kda_window, sources.glm_kda_parts, sources.glm_tfq3.kda_q8 });
        defer gpa.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const parts = try mtl.Pipeline.init(device, lib, "glm_kda_parts_q8", false);
        errdefer parts.deinit();
        t.kda_q8 = .{ parts, try mtl.Pipeline.init(device, lib, "glm_kda_window_q8", false) };
        t.hash ^= std.hash.Wyhash.hash(0, src);
        w.tfq3 = t;
        k.source_hash ^= t.hash;
    }

    /// Main's KDA pipeline, unless `check.kdaQ8` names a TFQ3 layer's q8 parts or window (8-bit/64 gates).
    pub fn kdaSelect(k: *const Kernels, gate: wide.KdaGate, tfq3_q8: bool, tp: u32, rows: u32) struct { pipe: mtl.Pipeline, parts: bool } {
        const q8 = if (k.wide) |*w| if (w.tfq3) |*t| t.kda_q8 else null else null;
        if (gate == .fused) switch (wide.tfq3.check.kdaQ8(tfq3_q8, true, q8 != null, k.kda_split, tp, rows)) {
            .parts => return .{ .pipe = q8.?[0], .parts = true },
            .window => return .{ .pipe = q8.?[1], .parts = false },
            .none => {},
        };
        const parts = gate == .generated and k.kda_split and tp == 1 and rows > 1;
        const pipe = switch (gate) {
            .generated => if (parts) k.kda_parts else if (rows > 1) (if (tp > 1) k.kda_window_tp else k.kda_window) else if (tp > 1) k.kda_rows_tp else k.kda_rows,
            .pre, .fused => |p| p,
        };
        return .{ .pipe = pipe, .parts = parts };
    }
};

/// The generated kernel `key` (tools/zig/gen_glm_kernels.py) and the field its first function fills.
const generated = [_]struct { key: []const u8, field: []const u8 }{
    .{ .key = "qmv_4096_24896", .field = "qmv_kda_in" },
    .{ .key = "qmv_8192_4096", .field = "qmv_kda_out" },
    .{ .key = "qmv_4096_2208", .field = "qmv_x" },
    .{ .key = "qmv_1536_20480", .field = "qmv_qr" },
    .{ .key = "qmv_16384_4096", .field = "qmv_mla_out" },
    .{ .key = "qmv_4096_24576", .field = "qmv_dense_gu" },
    .{ .key = "qmv_12288_4096", .field = "qmv_dense_down" },
    .{ .key = "qmv_4096_154880", .field = "qmv_head" },
    .{ .key = "qmv_4096_12576", .field = "qmv_kda_in_tp" },
    .{ .key = "qmvp_4096_4096", .field = "qmvp_kda_out" },
    .{ .key = "qmv_4096_77440", .field = "qmv_head_tp" },
    .{ .key = "qmv_1536_12288", .field = "qmv_qr_tp" },
    .{ .key = "qmvp_8192_4096", .field = "qmvp_mla_out" },
    .{ .key = "sparse_attention_tp", .field = "sparse_attention_tp" },
    .{ .key = "qmv_4096_12288", .field = "qmv_dense_gu_tp" },
    .{ .key = "qmvp_6144_4096", .field = "qmvp_dense_down" },
    .{ .key = "gemv_t_igate", .field = "gemv_t_igate" },
    .{ .key = "gemv_t_values", .field = "gemv_t_values" },
    .{ .key = "gemv_scores_lt4", .field = "gemv_scores_lt4" },
    .{ .key = "gemv_scores_le32", .field = "gemv_scores_le32" },
    .{ .key = "gemv_scores", .field = "gemv_scores" },
    .{ .key = "hc_expand_10", .field = "hc_expand_10" },
    .{ .key = "kda_rows", .field = "kda_rows" },
    .{ .key = "kda_rows_tp", .field = "kda_rows_tp" },
    .{ .key = "router", .field = "router" },
    .{ .key = "moe_route", .field = "moe_route" },
    .{ .key = "moe_gateup_2", .field = "moe_gateup_2" },
    .{ .key = "moe_down_2", .field = "moe_down_2" },
    .{ .key = "moe_combine", .field = "moe_combine" },
    .{ .key = "moe_gateup_2h", .field = "moe_gateup_2h" },
    .{ .key = "moe_down_2h", .field = "moe_down_2h" },
    .{ .key = "sparse_attention", .field = "sparse_attention" },
};

const glue = [_]struct { name: [:0]const u8, field: []const u8 }{
    .{ .name = "glm_cast_f32", .field = "cast_f32" },           .{ .name = "glm_rms", .field = "rms" },
    .{ .name = "glm_layer_norm", .field = "layer_norm" },       .{ .name = "glm_absorb", .field = "absorb" },
    .{ .name = "glm_unabsorb", .field = "unabsorb" },           .{ .name = "glm_scale", .field = "scale" },
    .{ .name = "glm_pool", .field = "pool" },                   .{ .name = "glm_stream_mean", .field = "stream_mean" },
    .{ .name = "glm_tap_mean_ring", .field = "tap_mean_ring" }, .{ .name = "glm_add", .field = "add" },
    .{ .name = "glm_swiglu", .field = "swiglu" },               .{ .name = "glm_argmax", .field = "argmax" },
    .{ .name = "glm_index_select", .field = "index_select" },   .{ .name = "glm_exp_f32", .field = "exp_f32" },
    .{ .name = "glm_streams", .field = "streams" },             .{ .name = "glm_copy_u32", .field = "copy_u32" },
    .{ .name = "glm_route_rows", .field = "route_rows" },       .{ .name = "glm_act2", .field = "act2" },
    .{ .name = "glm_dense_indices", .field = "dense_indices" }, .{ .name = "glm_mla_norms", .field = "mla_norms" },
};

const Job = struct {
    device: mtl.Device,
    source: []const u8,
    names: []const [:0]const u8,
    out: []mtl.Pipeline,
    failed: bool = false,

    fn run(job: *Job) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = mtl.Library.fromSource(job.device, job.source, mtl.CompileOptions.mlx()) catch {
            job.failed = true;
            return;
        };
        defer lib.deinit();
        for (job.names, job.out) |name, *p| {
            p.* = mtl.Pipeline.init(job.device, lib, name, false) catch {
                job.failed = true;
                return;
            };
        }
    }
};

/// Generated kernel `key`'s template at other parameters, under `name`.
pub fn instance(comptime key: []const u8, comptime params: []const u8, comptime name: []const u8) []const u8 {
    const f = comptime kernelOf(key).functions[0];
    return "\ntemplate [[host_name(\"" ++ name ++ "\")]] [[kernel]] decltype(" ++ f ++ "<" ++ params ++ ">) " ++ f ++ "<" ++ params ++ ">;\n";
}

/// Generated kernel `key`'s source.
pub fn generatedSource(comptime key: []const u8) []const u8 {
    return kernelOf(key).source;
}

/// Generated kernel `key`'s first function.
pub fn generatedFunction(comptime key: []const u8) [:0]const u8 {
    return kernelOf(key).functions[0];
}

fn kernelOf(comptime key: []const u8) sources.glm.Kernel {
    @setEvalBranchQuota(20000);
    for (sources.glm.all) |k| if (comptime std.mem.eql(u8, k.key, key)) return k;
    @compileError("no generated GLM kernel " ++ key);
}

/// The KDA step's 64 heads on two threadgroups each for 2+ rows on an Ultra's GPU (faster there), not on a Max's.
pub fn splitKda(gpu: []const u8) bool {
    return std.mem.indexOf(u8, gpu, "Ultra") != null;
}

/// `src` with `old` replaced by `new` once: error.KernelCompile when the generated text no longer has it.
fn edit(gpa: std.mem.Allocator, src: []const u8, old: []const u8, new: []const u8) ![]u8 {
    if (std.mem.count(u8, src, old) != 1) return error.KernelCompile;
    return std.mem.replaceOwned(u8, gpa, src, old, new);
}

/// One row's experts with no select launch: gate/up groups replay the select's picks, slot u takes pick u (same slots).
fn pickSources(gpa: std.mem.Allocator, route_core: []const u8, gu: []const u8, down: []const u8) ![2][]u8 {
    const member = "  if (!shared && u >= UCOUNT[0]) return;\n  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];";
    const by_pick = "  const int pick = shared ? (m < R ? m * TOPK : -1) : u;";
    const g1 = try edit(gpa, gu, member, by_pick);
    defer gpa.free(g1);
    const g2 = try edit(gpa, g1, "  const size_t e = shared ? 0 : size_t(UIDS[u]);", "  const size_t e = shared ? 0 : size_t(tf_route_pick_at(RLOGITS, RBIAS, u, lane, u == TOPK - 1 && threadgroup_position_in_grid.y == 0, RPICK, RWTS, RSCALE));");
    defer gpa.free(g2);
    const g3 = try edit(gpa, g2, "  const device int32_t* UIDS [[buffer(11)]],\n  const device int32_t* UMEM [[buffer(12)]],\n  const device int32_t* UCOUNT [[buffer(13)]],", "  const device float* RLOGITS [[buffer(11)]],\n  const device float* RBIAS [[buffer(12)]],\n  device int32_t* RPICK [[buffer(13)]],");
    defer gpa.free(g3);
    const g4 = try edit(gpa, g3, "  device bfloat16_t* ACT [[buffer(15)]],", "  device bfloat16_t* ACT [[buffer(15)]],\n  device float* RWTS [[buffer(16)]],\n  constant float& RSCALE [[buffer(17)]],");
    defer gpa.free(g4);
    const gate_up = try std.mem.concat(gpa, u8, &.{ route_core, g4 });
    errdefer gpa.free(gate_up);
    return .{ gate_up, try edit(gpa, down, member, by_pick) };
}

const m16_names = affine_mm.names ++ mm_groups.names;

pub fn mtpPrep(k: *const Kernels, e: mtl.ComputeEncoder, bits: u8, ids: anytype, emb: anytype, we: anytype, wh: anytype, h: anytype, out: anytype, rows: u32, width: u32, eps: f32) void {
    e.setPipeline(if (bits == 4) k.mtp_prep_4b else k.mtp_prep_8b);
    inline for (.{ ids, emb.w, emb.s, emb.b, we, wh, h, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(MtpPrepArgs{ .rows = rows, .width = width, .eps = eps }, 8);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(((width + 3) / 4 + 31) / 32 * 32, 1, 1));
}

pub fn headTop1_4b(k: *const Kernels, e: mtl.ComputeEncoder, in: anytype, q: anytype, n: u32, scratch_pairs: anytype, picks: anytype, rows: u32) void {
    const pairs: u32 = if (rows == 1) (n + 7) / 8 else (n + 3) / 4;
    const args = HeadTop1Args{ .k = q.k, .n = n, .rows = rows, .pairs = pairs };
    e.setPipeline(if (rows == 1) k.head_top1_4b_one else k.head_top1_4b);
    inline for (.{ in, q.w, q.s, q.b }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(args, 4);
    e.setBuffer(scratch_pairs.buf, scratch_pairs.off, 5);
    e.dispatchGroups(mtl.Size.of(pairs, 1, 1), mtl.Size.of(if (rows == 1) 128 else 32 * rows, 1, 1));
    e.setPipeline(k.reduce_top1);
    e.setBuffer(scratch_pairs.buf, scratch_pairs.off, 0);
    e.setBuffer(picks.buf, picks.off, 1);
    e.setValue(args, 2);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

pub fn canHeadTop1(allow: bool, tp: u32, m_row: u32, n: u32, hw: anytype, rows: u32) bool {
    return allow and tp == 1 and m_row == 0 and n % 4 == 0 and hw.more == 0 and hw.bits == 4 and hw.group == 64 and hw.k >= 512 and hw.k % 512 == 0 and rows > 0 and rows <= max_rows;
}

pub fn tryHeadTop1(k: *const Kernels, e: mtl.ComputeEncoder, in: anytype, hw: anytype, logits: anytype, picks: anytype, rows: u32, n: u32, tp: u32, m_row: u32, allow: bool) bool {
    if (!canHeadTop1(allow, tp, m_row, n, hw, rows)) return false;
    headTop1_4b(k, e, in, hw, n, logits, picks, rows);
    return true;
}

/// Compile every library on worker threads.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !*Kernels {
    const k = try gpa.create(Kernels);
    errdefer gpa.destroy(k);
    try assemble(gpa, device, k, {});
    return k;
}

/// Visit the loader's assembled libraries without making pipelines or submitting commands.
pub fn visitSources(gpa: std.mem.Allocator, device: mtl.Device, visitor: anytype) !void {
    var k: Kernels = undefined;
    try assemble(gpa, device, &k, visitor);
}

fn assemble(gpa: std.mem.Allocator, device: mtl.Device, k: *Kernels, visitor: anytype) !void {
    k.wide = null;
    const m3_fast = !device.tensorUnits();
    k.m3_fast = m3_fast;
    var jobs: [generated.len + 29]Job = undefined;
    inline for (generated, 0..) |g, i| {
        const src = comptime kernelOf(g.key);
        const FT = @FieldType(Kernels, g.field);
        const single = FT == mtl.Pipeline;
        if (src.functions.len != (if (single) 1 else @typeInfo(FT).array.len)) @compileError("GLM kernel " ++ g.key ++ ": functions and pipelines differ");
        const out: []mtl.Pipeline = if (single) @as(*[1]mtl.Pipeline, &@field(k, g.field)) else &@field(k, g.field);
        jobs[i] = .{ .device = device, .source = src.source, .names = src.functions, .out = out };
    }
    var glue_out: [glue.len]mtl.Pipeline = undefined;
    const glue_names = comptime blk: {
        var n: [glue.len][:0]const u8 = undefined;
        for (glue, 0..) |g, i| n[i] = g.name;
        break :blk n;
    };
    jobs[generated.len] = .{ .device = device, .source = sources.glm_glue, .names = &glue_names, .out = &glue_out };
    var sample_out: [1]mtl.Pipeline = undefined;
    jobs[generated.len + 1] = .{ .device = device, .source = sources.nemotron_sample, .names = &.{"tf_sample_full"}, .out = &sample_out };
    jobs[generated.len + 2] = .{ .device = device, .source = sources.ops_softmax, .names = &.{"tf_softmax_bf16"}, .out = @as(*[1]mtl.Pipeline, &k.softmax) };
    jobs[generated.len + 3] = .{ .device = device, .source = sources.ops_embed_norm, .names = &.{"tf_embed_b4_g64"}, .out = @as(*[1]mtl.Pipeline, &k.embed) };
    const attn_src = try frags.source(device, gpa, sources.glm_attn);
    defer gpa.free(attn_src);
    var attn_out: [4]mtl.Pipeline = undefined;
    jobs[generated.len + 4] = .{ .device = device, .source = attn_src, .names = &.{ "glm_latent_scores", "glm_latent_values", "glm_latent_scores_rows", "glm_latent_values_rows" }, .out = &attn_out };
    const route_core = try moe_route.source(gpa, route_shape);
    defer gpa.free(route_core);
    const route_src = try std.mem.concat(gpa, u8, &.{ route_core, sources.glm_route_one });
    defer gpa.free(route_src);
    var route_out: [3]mtl.Pipeline = undefined;
    jobs[generated.len + 5] = .{ .device = device, .source = route_src, .names = &.{ moe_route.names[0], moe_route.names[1], "glm_route_logits_one" }, .out = &route_out };
    const igate_core = try moe_route.source(gpa, igate_shape);
    defer gpa.free(igate_core);
    const igate_src = try std.mem.concat(gpa, u8, &.{ igate_core, sources.glm_route_one });
    defer gpa.free(igate_src);
    var igate_out: [2]mtl.Pipeline = undefined;
    jobs[generated.len + 6] = .{ .device = device, .source = igate_src, .names = &.{ moe_route.names[0], "glm_route_logits_one" }, .out = &igate_out };
    const m16_src = try mm_groups.source(device, gpa); // affine_mm.names, then the column-group gathers
    defer gpa.free(m16_src);
    const m32_src = try affine_mm.source(device, gpa, .{ .out_f32 = true, .glm_m3_fast = m3_fast, .exact_nonfinite = true });
    defer gpa.free(m32_src);
    var m16_out: [m16_names.len]mtl.Pipeline = undefined;
    const grouped_from: usize = if (m3_fast) affine_mm.names.len else 0;
    jobs[generated.len + 7] = .{ .device = device, .source = m16_src, .names = m16_names[grouped_from..], .out = m16_out[grouped_from..] };
    jobs[generated.len + 8] = .{ .device = device, .source = m32_src, .names = &affine_mm.names, .out = &k.mm_f32 };
    const m3_def = if (m3_fast) "#define TF_GLM_M3_FAST 1\n" else "";
    const kda_src = try std.mem.concat(gpa, u8, &.{ comptime kernelOf("kda_rows").source, m3_def, sources.glm_kda_prompt, sources.glm_kda_window, sources.glm_kda_parts });
    defer gpa.free(kda_src);
    var kda_out: [10]mtl.Pipeline = undefined;
    jobs[generated.len + 9] = .{ .device = device, .source = kda_src, .names = &.{ "glm_kda_prep", "glm_kda_scan", "glm_kda_post", "glm_kda_prep_tp", "glm_kda_scan_tp", "glm_kda_post_tp", "glm_kda_window", "glm_kda_window_tp", "glm_kda_parts", "glm_kda_norm" }, .out = &kda_out };
    const sparse_src = try frags.source(device, gpa, sources.glm_sparse_nax);
    defer gpa.free(sparse_src);
    var sparse_out: [3]mtl.Pipeline = undefined;
    jobs[generated.len + 10] = .{ .device = device, .source = sparse_src, .names = &.{ "glm_sparse_nax", "glm_sparse_split", "glm_sparse_combine" }, .out = &sparse_out };
    const absorb_src = try frags.source(device, gpa, sources.glm_absorb_nax);
    defer gpa.free(absorb_src);
    jobs[generated.len + 11] = .{ .device = device, .source = absorb_src, .names = &.{"glm_absorb_nax"}, .out = @as(*[1]mtl.Pipeline, &k.absorb_nax) };
    const hc_src = try hc.source(gpa, hc_shape);
    defer gpa.free(hc_src);
    jobs[generated.len + 12] = .{ .device = device, .source = hc_src, .names = &hc.names, .out = &k.hc_core };
    const tp_heads = "#define GLM_HEADS 32\n"; // TP2: one Mac's MLA heads
    const sparse_tp_raw = try std.mem.concat(gpa, u8, &.{ tp_heads, sources.glm_sparse_nax });
    defer gpa.free(sparse_tp_raw);
    const sparse_tp_src = try frags.source(device, gpa, sparse_tp_raw);
    defer gpa.free(sparse_tp_src);
    jobs[generated.len + 13] = .{ .device = device, .source = sparse_tp_src, .names = &.{"glm_sparse_nax"}, .out = @as(*[1]mtl.Pipeline, &k.sparse_nax_tp) };
    const absorb_tp_raw = try std.mem.concat(gpa, u8, &.{ tp_heads, sources.glm_absorb_nax });
    defer gpa.free(absorb_tp_raw);
    const absorb_tp_src = try frags.source(device, gpa, absorb_tp_raw);
    defer gpa.free(absorb_tp_src);
    jobs[generated.len + 14] = .{ .device = device, .source = absorb_tp_src, .names = &.{"glm_absorb_nax"}, .out = @as(*[1]mtl.Pipeline, &k.absorb_nax_tp) };
    const index_raw = try std.mem.concat(gpa, u8, &.{ m3_def, sources.glm_index_nax });
    defer gpa.free(index_raw);
    const index_src = try frags.source(device, gpa, index_raw);
    defer gpa.free(index_src);
    var index_out: [2]mtl.Pipeline = undefined;
    jobs[generated.len + 15] = .{ .device = device, .source = index_src, .names = &.{ "glm_index_scores_nax", "glm_index_decode" }, .out = &index_out };
    const shared_gu_src = comptime kernelOf("moe_gateup_1").source ++ instance("moe_gateup_1", "4096, 2048, 1, 8, 16, 128, 1, 4, 16, 8", "glm_shared_gateup");
    const shared_down_src = comptime kernelOf("moe_down_1").source ++ instance("moe_down_1", "2048, 4096, 1, 8, 16, 128, 1, 4, 16, 8", "glm_shared_down");
    jobs[generated.len + 16] = .{ .device = device, .source = shared_gu_src, .names = &.{"glm_shared_gateup"}, .out = @as(*[1]mtl.Pipeline, &k.shared_gateup) };
    jobs[generated.len + 17] = .{ .device = device, .source = shared_down_src, .names = &.{"glm_shared_down"}, .out = @as(*[1]mtl.Pipeline, &k.shared_down) };
    const pick_slot = "const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);"; // the shared expert: grid z rows * topk, past the routed ids
    const gu0_src = try std.mem.replaceOwned(u8, gpa, comptime kernelOf("moe_gateup_2").source ++ instance("moe_gateup_2", "4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8", "glm_moe_gateup_0"), pick_slot, "const int u = PART == 1 ? MAXU : (int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK ? MAXU : int(threadgroup_position_in_grid.z));");
    defer gpa.free(gu0_src);
    const down0_src = try std.mem.replaceOwned(u8, gpa, comptime kernelOf("moe_down_2").source ++ instance("moe_down_2", "2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8", "glm_moe_down_0"), pick_slot, "const int u = PART == 1 ? MAXU : (int(threadgroup_position_in_grid.z) == int(ACT_shape[0]) * TOPK ? MAXU : int(threadgroup_position_in_grid.z));");
    defer gpa.free(down0_src);
    if (std.mem.indexOf(u8, gu0_src, pick_slot) != null or std.mem.indexOf(u8, down0_src, pick_slot) != null) return error.KernelCompile;
    jobs[generated.len + 18] = .{ .device = device, .source = gu0_src, .names = &.{"glm_moe_gateup_0"}, .out = @as(*[1]mtl.Pipeline, &k.moe_gateup_0) };
    jobs[generated.len + 19] = .{ .device = device, .source = down0_src, .names = &.{"glm_moe_down_0"}, .out = @as(*[1]mtl.Pipeline, &k.moe_down_0) };
    const gu0_one = try std.mem.replaceOwned(u8, gpa, gu0_src, "<4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8>", "<4096, 2048, 2, 8, 16, 128, 0, 4, 16, 8>");
    defer gpa.free(gu0_one);
    const down0_one = try std.mem.replaceOwned(u8, gpa, down0_src, "<2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8>", "<2048, 4096, 2, 8, 16, 128, 0, 4, 16, 8>");
    defer gpa.free(down0_one);
    if (std.mem.eql(u8, gu0_one, gu0_src) or std.mem.eql(u8, down0_one, down0_src)) return error.KernelCompile;
    jobs[generated.len + 20] = .{ .device = device, .source = gu0_one, .names = &.{"glm_moe_gateup_0"}, .out = @as(*[1]mtl.Pipeline, &k.moe_gateup_0_one) };
    jobs[generated.len + 21] = .{ .device = device, .source = down0_one, .names = &.{"glm_moe_down_0"}, .out = @as(*[1]mtl.Pipeline, &k.moe_down_0_one) };
    const picks = try pickSources(gpa, route_core, gu0_one, down0_one);
    defer gpa.free(picks[0]);
    defer gpa.free(picks[1]);
    jobs[generated.len + 25] = .{ .device = device, .source = picks[0], .names = &.{"glm_moe_gateup_0"}, .out = @as(*[1]mtl.Pipeline, &k.moe_gateup_pick) };
    jobs[generated.len + 26] = .{ .device = device, .source = picks[1], .names = &.{"glm_moe_down_0"}, .out = @as(*[1]mtl.Pipeline, &k.moe_down_pick) };
    const comb = comptime kernelOf("moe_combine");
    const comb_a = try std.mem.replaceOwned(u8, gpa, comb.source, "const device bfloat* y = Y + size_t(r) * TOPK * D + d;", "const device bfloat* y = Y + size_t(r) * (TOPK + 1) * D + d;");
    defer gpa.free(comb_a);
    const comb_src = try std.mem.replaceOwned(u8, gpa, comb_a, "+ YS[size_t(r) * D + d];", "+ y[size_t(TOPK) * D];");
    defer gpa.free(comb_src);
    if (std.mem.eql(u8, comb_src, comb.source)) return error.KernelCompile; // the generated combine changed: its edit no longer applies
    jobs[generated.len + 22] = .{ .device = device, .source = comb_src, .names = comb.functions, .out = @as(*[1]mtl.Pipeline, &k.combine_0) };
    var select_out: [2]mtl.Pipeline = undefined;
    jobs[generated.len + 23] = .{ .device = device, .source = sources.glm_select, .names = &.{ "glm_select_mark", "glm_select_merge" }, .out = &select_out };
    const softmax_rows_src = comptime sources.ops_softmax ++ sources.glm_softmax_rows;
    jobs[generated.len + 24] = .{ .device = device, .source = softmax_rows_src, .names = &.{"glm_softmax_rows"}, .out = @as(*[1]mtl.Pipeline, &k.softmax_rows) };
    var mtp_out: [5]mtl.Pipeline = undefined;
    jobs[generated.len + 27] = .{ .device = device, .source = sources.glm_mtp_opt, .names = &.{ "glm_mtp_prep_4b", "glm_mtp_prep_8b", "glm_head_top1_4b_one", "glm_head_top1_4b", "glm_reduce_top1" }, .out = &mtp_out };

    const m3_src = if (m3_fast) try affine_mm.source(device, gpa, .{ .glm_m3_fast = true, .exact_nonfinite = true }) else try gpa.dupe(u8, "");
    defer gpa.free(m3_src);
    var m3_out: affine_mm.Pipes = undefined;
    if (m3_fast) jobs[generated.len + 28] = .{ .device = device, .source = m3_src, .names = &affine_mm.names, .out = &m3_out };
    const active_jobs = jobs[0 .. jobs.len - @intFromBool(!m3_fast)];
    if (comptime @TypeOf(visitor) != void) {
        for (active_jobs) |j| try visitor.raw("glm", j.names[0], j.source);
        return;
    }
    var sources_seen = std.hash.Wyhash.init(0x6b);
    for (active_jobs) |j| sources_seen.update(j.source);
    k.source_hash = sources_seen.final();
    var next = std.atomic.Value(usize).init(0);
    const Worker = struct {
        fn run(all: []Job, counter: *std.atomic.Value(usize)) void {
            while (true) {
                const i = counter.fetchAdd(1, .monotonic);
                if (i >= all.len) return;
                all[i].run();
            }
        }
    };
    var threads: [8]?std.Thread = @splat(null);
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, Worker.run, .{ active_jobs, &next }) catch null;
    Worker.run(active_jobs, &next);
    for (threads) |t| if (t) |th| th.join();
    for (active_jobs) |j| if (j.failed) return error.KernelCompile;
    inline for (glue, 0..) |g, i| @field(k, g.field) = glue_out[i];
    k.sample = sample_out[0]; // the shared sampler's own library: glm_glue's table never named it
    k.latent_scores = attn_out[0];
    k.latent_values = attn_out[1];
    k.latent_scores_rows = attn_out[2];
    k.latent_values_rows = attn_out[3];
    k.mm_bf16 = if (m3_fast) m3_out else m16_out[0..affine_mm.names.len].*;
    k.mm_groups = m16_out[affine_mm.names.len..].*;
    k.route_logits = route_out[0];
    k.route_select = route_out[1];
    k.route_logits_one = route_out[2];
    k.igate_logits = igate_out[0];
    k.igate_logits_one = igate_out[1];
    k.kda_prep = kda_out[0];
    k.kda_scan = kda_out[1];
    k.kda_post = kda_out[2];
    k.kda_prep_tp = kda_out[3];
    k.kda_scan_tp = kda_out[4];
    k.kda_post_tp = kda_out[5];
    k.kda_window = kda_out[6];
    k.kda_window_tp = kda_out[7];
    k.kda_parts = kda_out[8];
    k.kda_norm = kda_out[9];
    k.kda_split = splitKda(std.mem.span(device.name()));
    k.sparse_nax = sparse_out[0];
    k.sparse_split = sparse_out[1];
    k.sparse_combine = sparse_out[2];
    k.index_scores_nax = index_out[0];
    k.index_decode = index_out[1];
    k.select_mark = select_out[0];
    k.select_merge = select_out[1];
    k.mtp_prep_4b = mtp_out[0];
    k.mtp_prep_8b = mtp_out[1];
    k.head_top1_4b_one = mtp_out[2];
    k.head_top1_4b = mtp_out[3];
    k.reduce_top1 = mtp_out[4];
}

test "glm_tap_mean_ring unique source anchor in glm_glue.metal" {
    try std.testing.expect(std.mem.indexOf(u8, sources.glm_glue, "kernel void glm_tap_mean_ring(") != null);
    try std.testing.expect(std.mem.indexOf(u8, sources.glm_glue, "GlmTapRingArgs") != null);
}
