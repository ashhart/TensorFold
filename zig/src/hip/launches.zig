//! The model-free kernels launched from Zig on the embedded code objects, one module a source group.

const std = @import("std");
const runtime = @import("runtime.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const caps = @import("caps.zig");
const affine = @import("launches/affine.zig");

const util = @import("launches/util.zig");

pub const Error = util.Error;

const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const I = util.I;
const CI = util.CI;
const Dim3 = util.Dim3;
const Args = util.Args;
const Triple = util.Triple;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;

const norms = @import("launches/norms.zig");
const conv = @import("launches/conv.zig");
const rope = @import("launches/rope.zig");
const moe = @import("launches/moe.zig");
const attention = @import("launches/attention.zig");
const recurrence = @import("launches/recurrence.zig");
const plan = @import("launches/plan.zig");
const elementwise = @import("launches/elementwise.zig");
const draw = @import("launches/draw.zig");

pub const ConvArgs = conv.ConvArgs;
pub const PlanArgs = plan.PlanArgs;
pub const PlanRef = plan.PlanRef;
pub const PlanKeep = plan.Keep;
pub const gdn_chunk = recurrence.gdn_chunk;
pub const GdnScratch = recurrence.GdnScratch;
pub const gdnScratch = recurrence.gdnScratch;

pub const Launcher = struct {
    r: *const runtime.Runtime,
    mods: [kernels.group_count]Module,
    rms: Triple,
    conv_decode: Function,
    conv_rows: Function,
    rope_decode: Function,
    moe_router: [2]Function, // fp16, bf16
    moe_select: Function,
    moe_act: [2]Function,
    moe_combine: Triple,
    gdn_gate: Triple,
    gated_delta_tile: [2]Function, // dk 128, 16
    gated_delta_wave: [2]Function,
    score_keys: Triple, // by cache kind
    apply_values: Triple,
    causal: Triple,
    fa_prefill: Triple,
    fa_wide: [2]Function, // the 64-row prefill tile, fp16 and bf16 caches
    fa_paged: [2]Function, // the same tile over a stream's pages
    gdn_chunked: [5]Function, // prep, kt, wy, h, o
    softmax_stats: Function,
    sum_partials: Function,
    op: Ops,
    /// decode.hip's kernels: a round's few rows, and the launches that merge several small ones.
    dec: Decode,
    /// plan.hip's kernels and the DeltaNet's over a round's plan.
    plan: PlanFns,
    /// decode.hip's merged launches are on (off keeps the launches they replace).
    fuse: bool,
    /// The 64-row prefill attention tile is on.
    wide: bool,
    /// The DeltaNet prefill runs chunked (off keeps the one-token recurrence).
    chunked: bool,
    /// The MLX affine products and the registry that picks their tiles.
    affine: affine.Kernels,

    const PlanFns = struct {
        kv_write: Function,
        score: [2]Function, // fp16, bf16
        apply: [2]Function,
        keep: Function,
        gather: Function,
        gdn: [2]Function, // dk 128, 16
        gdn_replay: [2]Function, // dk 128, 16
        page_write: Function,
        page_gather: Function,
    };

    const Decode = struct {
        router: [2]Function, // fp16, bf16
        tail: Function,
        conv_split: Function,
        rms2: Function,
        gnorm_out: Function,
        select: Function,
    };

    const Ops = struct {
        embed_rows: Function,
        embed_dense: Function,
        cast: Function,
        silu_mul: Function,
        add: Function,
        attn_gate: Function,
        gnorm_silu: Function,
        copy_cols: Function,
        rope_prefill: Function,
        kv_write: Function,
        kv_write_at: Function,
        argmax: Function,
        topk: Function,
        conv_prefill: Function,
        gdn_gate_prefill: Function,
        dense_rows: Function,
        moe_route: Function,
        router_rows: [2]Function, // fp16, bf16
        router_tile: [2]Function,
        router_small: [2]Function,
        rms_rows: Function,
        qk_rope: Function,
    };

    /// Which kernels a run takes where two give the same bits; every one defaults on.
    pub const Choices = struct { fuse: bool = true, wide: bool = true, chunked: bool = true, products: affine.Choices = .{} };

    /// Loads the code objects on the current device and resolves every kernel the launchers use.
    pub fn load(r: *const runtime.Runtime, choices: Choices, images: [kernels.group_count][]align(8) const u8) Error!Launcher {
        var l: Launcher = undefined;
        l.r = r;
        var loaded: usize = 0;
        errdefer for (l.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            l.mods[i] = try Module.load(r, img);
            loaded += 1;
        }
        const ops = l.mods[@backingInt(kernels.Group.ops)];
        const act = l.mods[@backingInt(kernels.Group.act)];
        const att = l.mods[@backingInt(kernels.Group.attention)];
        const gd = l.mods[@backingInt(kernels.Group.gated_delta)];
        const pre = l.mods[@backingInt(kernels.Group.prefill)];
        const dec = l.mods[@backingInt(kernels.Group.decode)];
        const pl = l.mods[@backingInt(kernels.Group.plan)];
        l.fuse = choices.fuse;
        l.wide = choices.wide;
        l.chunked = choices.chunked;
        l.dec = .{ .router = .{ try dec.function("tf_router_decode_f16"), try dec.function("tf_router_decode_bf16") }, .tail = try dec.function("tf_tail"), .conv_split = try dec.function("tf_conv_split"), .rms2 = try dec.function("tf_rms2"), .gnorm_out = try dec.function("tf_gnorm_out"), .select = try dec.function("tf_select_decode") };
        l.plan = .{
            .kv_write = try pl.function("tf_plan_kv_write"),
            .score = .{ try pl.function("tf_plan_score_f16"), try pl.function("tf_plan_score_bf16") },
            .apply = .{ try pl.function("tf_plan_apply_f16"), try pl.function("tf_plan_apply_bf16") },
            .keep = try pl.function("tf_plan_keep"),
            .gather = try pl.function("tf_plan_gather"),
            .gdn = .{ try gd.function("tf_gdn_plan_128"), try gd.function("tf_gdn_plan_16") },
            .gdn_replay = .{ try gd.function("tf_gdn_replay_128"), try gd.function("tf_gdn_replay_16") },
            .page_write = try pl.function("tf_page_write"),
            .page_gather = try pl.function("tf_page_gather"),
        };
        l.fa_wide = .{ try pre.function("tf_fa_wide_f16"), try pre.function("tf_fa_wide_bf16") };
        l.fa_paged = .{ try pre.function("tf_fa_paged_f16"), try pre.function("tf_fa_paged_bf16") };
        const gp = l.mods[@backingInt(kernels.Group.gdn_prefill)];
        l.gdn_chunked = .{ try gp.function("tf_gdn_prep"), try gp.function("tf_gdn_kt"), try gp.function("tf_gdn_wy"), try gp.function("tf_gdn_h"), try gp.function("tf_gdn_o") };
        l.op = .{
            .embed_rows = try ops.function("tf_embed_rows"),
            .embed_dense = try ops.function("tf_embed_dense"),
            .cast = try ops.function("tf_cast"),
            .silu_mul = try ops.function("tf_silu_mul"),
            .add = try ops.function("tf_add"),
            .attn_gate = try ops.function("tf_attn_gate"),
            .gnorm_silu = try ops.function("tf_gnorm_silu"),
            .copy_cols = try ops.function("tf_copy_cols"),
            .rope_prefill = try ops.function("tf_rope_prefill"),
            .kv_write = try ops.function("tf_kv_write"),
            .kv_write_at = try ops.function("tf_kv_write_at"),
            .argmax = try ops.function("tf_argmax"),
            .topk = try ops.function("tf_topk"),
            .conv_prefill = try ops.function("tf_conv_prefill"),
            .gdn_gate_prefill = try ops.function("tf_gdn_gate_prefill"),
            .dense_rows = try ops.function("tf_dense_rows"),
            .moe_route = try ops.function("tf_moe_route"),
            .router_rows = .{ try ops.function("tf_router_rows_f16"), try ops.function("tf_router_rows_bf16") },
            .router_tile = .{ try ops.function("tf_router_tile_f16"), try ops.function("tf_router_tile_bf16") },
            .router_small = .{ try ops.function("tf_router_small_f16"), try ops.function("tf_router_small_bf16") },
            .rms_rows = try ops.function("tf_rms_rows"),
            .qk_rope = try ops.function("tf_qk_rope"),
        };
        const ns = "_ZN2tf4rocm";
        l.rms = .{
            try act.function(ns ++ "10rms_kernelIfEEvPKT_PKfPS2_if"),
            try act.function(ns ++ "10rms_kernelI6__halfEEvPKT_PKfPS3_if"),
            try act.function(ns ++ "10rms_kernelI12hip_bfloat16EEvPKT_PKfPS3_if"),
        };
        l.conv_decode = try act.function("tf_conv_decode");
        l.conv_rows = try act.function("tf_conv_rows");
        l.rope_decode = try act.function("tf_rope_decode");
        l.moe_router = .{
            try act.function(ns ++ "17moe_router_kernelI6__halfEEvPKT_PKfPfii"),
            try act.function(ns ++ "17moe_router_kernelI12hip_bfloat16EEvPKT_PKfPfii"),
        };
        l.moe_select = try act.function("tf_moe_select");
        l.moe_act = .{
            try act.function(ns ++ "14moe_act_kernelI6__halfEEvPKfPT_ifx"),
            try act.function(ns ++ "14moe_act_kernelI12hip_bfloat16EEvPKfPT_ifx"),
        };
        l.moe_combine = .{
            try act.function(ns ++ "18moe_combine_kernelIfEEvPKfS3_PT_ii"),
            try act.function(ns ++ "18moe_combine_kernelI6__halfEEvPKfS4_PT_ii"),
            try act.function(ns ++ "18moe_combine_kernelI12hip_bfloat16EEvPKfS4_PT_ii"),
        };
        l.gdn_gate = .{
            try act.function(ns ++ "15gdn_gate_kernelIfEEvPKT_S4_PKfS6_PfS7_ii"),
            try act.function(ns ++ "15gdn_gate_kernelI6__halfEEvPKT_S5_PKfS7_PfS8_ii"),
            try act.function(ns ++ "15gdn_gate_kernelI12hip_bfloat16EEvPKT_S5_PKfS7_PfS8_ii"),
        };
        l.gated_delta_tile = .{
            try gd.function(ns ++ "16gated_delta_tileILi128ELi8ELi16EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
            try gd.function(ns ++ "16gated_delta_tileILi16ELi8ELi16EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
        };
        l.gated_delta_wave = .{
            try gd.function(ns ++ "18gated_delta_kernelILi128EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
            try gd.function(ns ++ "18gated_delta_kernelILi16EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
        };
        l.score_keys = .{
            try att.function("_Z10score_keysI6__halfEvPKfPKT_PfiiiifixxxPKi"),
            try att.function("_Z10score_keysI12hip_bfloat16EvPKfPKT_PfiiiifixxxPKi"),
            try att.function("_Z10score_keysIfEvPKfPKT_PfiiiifixxxPKi"),
        };
        l.apply_values = .{
            try att.function("_Z12apply_valuesI6__halfEvPKfS2_PKT_PfiiiiixxxPKi"),
            try att.function("_Z12apply_valuesI12hip_bfloat16EvPKfS2_PKT_PfiiiiixxxPKi"),
            try att.function("_Z12apply_valuesIfEvPKfS1_PKT_PfiiiiixxxPKi"),
        };
        l.causal = .{
            try att.function(ns ++ "13causal_kernelI6__halfEEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(ns ++ "13causal_kernelI12hip_bfloat16EEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(ns ++ "13causal_kernelIfEEvPKfPKT_S6_Pfiiiiifixxxxxx"),
        };
        l.fa_prefill = .{
            try att.function(ns ++ "10fa_prefillI6__halfLi64EEEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(ns ++ "10fa_prefillI12hip_bfloat16Li64EEEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(ns ++ "10fa_prefillIfLi32EEEvPKfPKT_S6_Pfiiiiifixxxxxx"),
        };
        l.softmax_stats = try att.function("tf_softmax_stats");
        l.sum_partials = try att.function("tf_sum_partials");
        const tiles = l.mods[@backingInt(kernels.Group.affine_tiles)];
        const dot2 = l.mods[@backingInt(kernels.Group.affine_dot2)];
        l.affine = try affine.Kernels.load(tiles, dot2, caps.Caps.of(kernels.arch) orelse return error.Invalid, choices.products);
        try l.affine.fillByteLut(r, dot2);
        return l;
    }

    pub fn unload(l: *Launcher) void {
        for (&l.mods) |*m| m.unload();
    }

    pub const tf_rms = norms.tf_rms;
    pub const tf_rms2 = norms.tf_rms2;
    pub const tf_gnorm_out = norms.tf_gnorm_out;
    pub const tf_tail = norms.tf_tail;
    pub const tf_conv_decode = conv.tf_conv_decode;
    pub const tf_conv_rows = conv.tf_conv_rows;
    pub const tf_conv_split = conv.tf_conv_split;
    pub const tf_conv_prefill = conv.tf_conv_prefill;
    pub const tf_qk_rope = rope.tf_qk_rope;
    pub const tf_rope_decode = rope.tf_rope_decode;
    pub const tf_rope_prefill = rope.tf_rope_prefill;
    pub const routerTile = moe.routerTile;
    pub const routerWith = moe.routerWith;
    pub const routerWindow = moe.routerWindow;
    pub const tf_moe_router = moe.tf_moe_router;
    pub const tf_moe_select = moe.tf_moe_select;
    pub const tf_moe_act = moe.tf_moe_act;
    pub const tf_moe_combine = moe.tf_moe_combine;
    pub const tf_moe_route = moe.tf_moe_route;
    pub const tf_causal = attention.tf_causal;
    pub const tf_attn_gate = attention.tf_attn_gate;
    pub const tf_kv_write = attention.tf_kv_write;
    pub const tf_kv_write_at = attention.tf_kv_write_at;
    pub const tf_argmax_rows = draw.tf_argmax_rows;
    pub const tf_topk_rows = draw.tf_topk_rows;
    pub const pagedCausal = attention.pagedCausal;
    pub const tf_gdn_gate = recurrence.tf_gdn_gate;
    pub const tf_gated_delta = recurrence.tf_gated_delta;
    pub const planKvWrite = plan.planKvWrite;
    pub const planCausal = plan.planCausal;
    pub const planGatedDelta = plan.planGatedDelta;
    pub const planKeep = plan.planKeep;
    pub const planGdnReplay = plan.planGdnReplay;
    pub const pagesWrite = plan.pagesWrite;
    pub const pagesGather = plan.pagesGather;
    pub const planGather = plan.planGather;
    pub const gdnChunked = recurrence.gdnChunked;
    pub const tf_gdn_gate_prefill = recurrence.tf_gdn_gate_prefill;
    pub const tf_gnorm_silu = recurrence.tf_gnorm_silu;
    pub const tf_embed_rows = elementwise.tf_embed_rows;
    pub const tf_embed_dense = elementwise.tf_embed_dense;
    pub const tf_cast = elementwise.tf_cast;
    pub const tf_silu_mul = elementwise.tf_silu_mul;
    pub const tf_add = elementwise.tf_add;
    pub const tf_copy_cols = elementwise.tf_copy_cols;
    pub const tf_dense_rows = elementwise.tf_dense_rows;

    /// Launches on `s`; the counting stream runs nothing.
    pub fn go(l: *const Launcher, f: Function, grid: Dim3, block: Dim3, shared: u32, s: S, args: *Args) Error!void {
        if (s == util.counting) return;
        try launch.launch(f, .{ .grid = grid, .block = block, .shared = shared }, Stream{ .r = l.r, .handle = s }, args);
    }

    pub fn tf_affine(l: *const Launcher, x: C, words: C, scale: C, bias: C, scale_kind: c_int, out: P, m: c_int, n: c_int, k: c_int, bits: c_int, group: c_int, schedule: c_int, fp16: c_int, s: S, partial: F, splits: c_int, out_half: c_int) Error!void {
        try l.affine.run(l.r, .{ .x = ad(x), .words = ad(words), .scale = .{ .p = ad(scale), .kind = scale_kind }, .bias = .{ .p = ad(bias), .kind = scale_kind }, .out = ad(out), .m = m, .n = n, .k = k, .bits = bits, .group = group, .fp16 = fp16 }, schedule, s, ad(partial), splits, out_half != 0);
    }

    pub fn tf_affine_routed(l: *const Launcher, x: C, words: C, scale: C, bias: C, scale_kind: c_int, out: P, items: CI, count: c_int, members: CI, x_div: c_int, rows: c_int, n: c_int, k: c_int, bits: c_int, group: c_int, fp16: c_int, s: S) Error!void {
        try l.affine.routed(l.r, .{ .x = ad(x), .words = ad(words), .scale = .{ .p = ad(scale), .kind = scale_kind }, .bias = .{ .p = ad(bias), .kind = scale_kind }, .out = ad(out), .m = rows, .n = n, .k = k, .bits = bits, .group = group, .fp16 = fp16, .route = .{ .items = ad(items), .members = ad(members), .x_div = x_div } }, count, s);
    }

    /// The 256-thread, one-element-a-thread launch of an ops.hip kernel over `n` elements.
    pub fn flat(l: *const Launcher, f: Function, n: i64, s: S, a: *Args) Error!void {
        if (n == 0) return;
        try l.go(f, dim(cdiv(n, 256), 1, 1), dim(256, 1, 1), 0, s, a);
    }
};
