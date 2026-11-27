//! Inkling 2 Mini glue on Metal: the production shader and its real encoder dispatches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
pub const plan = @import("inkling2_glue_plan");

pub const hidden: u32 = 2304;
pub const experts: u32 = 128;
pub const intermediate: u32 = 1152;
pub const top: u32 = 8;
pub const Ref = @import("inkling2_metal_ref").Ref;
const unit = mtl.Size.of(32, 1, 1);

pub const Pipelines = struct {
    embed_norm: mtl.Pipeline,
    ar: mtl.Pipeline,
    cached_ar: mtl.Pipeline,
    block_inv_rms: mtl.Pipeline,
    attn_partial_ar: mtl.Pipeline,
    mlp_partial_ar: mtl.Pipeline,
    conv_static: mtl.Pipeline,
    conv_dynamic: mtl.Pipeline,
    kv_conv_static: mtl.Pipeline,
    kv_conv_dynamic: mtl.Pipeline,
    router: mtl.Pipeline,
    route_scores: mtl.Pipeline,
    route_select: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    logits: mtl.Pipeline,
    logits_r4: mtl.Pipeline,
    logits_r8: mtl.Pipeline,
    logits_r16: mtl.Pipeline,
    logits_r32: mtl.Pipeline,
    expert_pick: mtl.Pipeline,
    expert_combine: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.glue);
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [21]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{
            "tf_inkling2_embed_norm",
            "tf_inkling2_ar",
            "tf_inkling2_cached_ar",
            "tf_inkling2_block_inv_rms",
            "tf_inkling2_attn_partial_ar",
            "tf_inkling2_mlp_partial_ar",
            "tf_inkling2_conv_static",
            "tf_inkling2_conv_dynamic",
            "tf_inkling2_kv_conv_static",
            "tf_inkling2_kv_conv_dynamic",
            "tf_inkling2_router",
            "tf_inkling2_swiglu",
            "tf_inkling2_logits",
            "tf_inkling2_expert_pick",
            "tf_inkling2_expert_combine",
            "tf_inkling2_route_scores",
            "tf_inkling2_route_select",
            "tf_inkling2_coalesced_logits_r4",
            "tf_inkling2_coalesced_logits_r8",
            "tf_inkling2_coalesced_logits_r16",
            "tf_inkling2_coalesced_logits_r32",
        }, [_]usize{ 256, 256, 256, 256, 256, 256, 32, 256, 32, 256, 256, 32, 32, 32, 32, 32, 32, 128, 128, 128, 256 }, &pipes) |name, req_threads, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < req_threads) return error.GlueThreads;
        }
        return .{
            .embed_norm = pipes[0],
            .ar = pipes[1],
            .cached_ar = pipes[2],
            .block_inv_rms = pipes[3],
            .attn_partial_ar = pipes[4],
            .mlp_partial_ar = pipes[5],
            .conv_static = pipes[6],
            .conv_dynamic = pipes[7],
            .kv_conv_static = pipes[8],
            .kv_conv_dynamic = pipes[9],
            .router = pipes[10],
            .swiglu = pipes[11],
            .logits = pipes[12],
            .expert_pick = pipes[13],
            .expert_combine = pipes[14],
            .route_scores = pipes[15],
            .route_select = pipes[16],
            .logits_r4 = pipes[17],
            .logits_r8 = pipes[18],
            .logits_r16 = pipes[19],
            .logits_r32 = pipes[20],
        };
    }
    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
    }
};

const EmbedArgs = extern struct { rows: u32 };
pub fn embedNorm(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, table: Ref, weight: Ref, out: Ref, rows: u32) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.embed_norm);
    inline for (.{ ids, table, weight, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(EmbedArgs{ .rows = rows }, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const ARArgs = extern struct { sources: u32, queries: u32, rows: u32, blocks_stride: u32, partial_stride: u32, out_stride: u32 };
/// Row i of a batch has its blocks, partial and output `*_stride` elements past row 0's.
pub const ARRows = struct { rows: u32 = 1, blocks_stride: u32 = 0, partial_stride: u32 = 0, out_stride: u32 = 0 };
/// sources = completed BF16 blocks plus one unrounded F32 partial; output is one FP32 row per query and batch row.
pub fn ar(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, batch: ARRows) void {
    std.debug.assert(sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3 and batch.rows > 0);
    e.setPipeline(p.ar);
    inline for (.{ blocks, partial, queries, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ARArgs{ .sources = sources_, .queries = queries_, .rows = batch.rows, .blocks_stride = batch.blocks_stride, .partial_stride = batch.partial_stride, .out_stride = batch.out_stride }, 4);
    e.dispatchGroups(mtl.Size.of(queries_, batch.rows, 1), mtl.Size.of(256, 1, 1));
}

const BlockInvRmsArgs = extern struct { rows: u32, slot: u32 };
pub fn blockInvRms(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, inv_rms: Ref, rows: u32, slot: u32) void {
    std.debug.assert(rows > 0 and rows <= 256 and slot < 4);
    e.setPipeline(p.block_inv_rms);
    inline for (.{ blocks, inv_rms }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(BlockInvRmsArgs{ .rows = rows, .slot = slot }, 2);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const CachedARArgs = extern struct { sources: u32, queries: u32 };
/// AR with precomputed FP32 inv_rms for completed BF16 blocks[0..sources-2] and live reduction on partial.
pub fn cachedAr(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, block_inv_rms: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32) void {
    std.debug.assert(sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3);
    e.setPipeline(p.cached_ar);
    inline for (.{ blocks, partial, block_inv_rms, queries, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(CachedARArgs{ .sources = sources_, .queries = queries_ }, 5);
    e.dispatchGroups(mtl.Size.of(queries_, 1, 1), mtl.Size.of(256, 1, 1));
}

const AttnPartialArArgs = extern struct { rows: u32, sources: u32, boundary: u32 };
/// Fuses attention_partial, optional block_inv_rms, and single-query MLP cached_ar across all rows.
pub fn attnPartialAr(p: *const Pipelines, e: mtl.ComputeEncoder, p_: Ref, attn_conv: Ref, blocks: Ref, block_inv_rms: Ref, partial: Ref, query: Ref, out: Ref, rows: u32, sources_: u32, boundary: u32) void {
    std.debug.assert(rows > 0 and rows <= 256 and sources_ > 0 and sources_ <= 5 and boundary <= 4);
    e.setPipeline(p.attn_partial_ar);
    inline for (.{ p_, attn_conv, blocks, block_inv_rms, partial, query, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(AttnPartialArArgs{ .rows = rows, .sources = sources_, .boundary = boundary }, 7);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const MlpPartialArArgs = extern struct { rows: u32, sources: u32, queries: u32 };
/// Fuses mlp_partial, BF16->FP32 widen, and next-visit attn_ar (or output_query AR) across all rows.
pub fn mlpPartialAr(p: *const Pipelines, e: mtl.ComputeEncoder, partial_in: Ref, mlp_conv: Ref, blocks: Ref, block_inv_rms: Ref, queries: Ref, p_out: Ref, partial_out: Ref, ar_out: Ref, rows: u32, sources_: u32, queries_: u32) void {
    std.debug.assert(rows > 0 and rows <= 256 and sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3);
    e.setPipeline(p.mlp_partial_ar);
    inline for (.{ partial_in, mlp_conv, blocks, block_inv_rms, queries, p_out, partial_out, ar_out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(MlpPartialArArgs{ .rows = rows, .sources = sources_, .queries = queries_ }, 8);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const ConvArgs = extern struct { rows: u32, channels: u32 };
/// taps contains three old rows followed by `rows` current BF16 inputs.
pub fn convStatic(p: *const Pipelines, e: mtl.ComputeEncoder, taps: Ref, weight: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0);
    e.setPipeline(p.conv_static);
    inline for (.{ taps, weight, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ConvArgs{ .rows = rows, .channels = channels }, 3);
    e.dispatchThreads(mtl.Size.of(channels, rows, 1), unit);
}

/// Dynamic static_w is [lag][channel], the opposite orientation to static convolution taps.
pub fn convDynamic(p: *const Pipelines, e: mtl.ComputeEncoder, taps: Ref, input: Ref, static_w: Ref, projection: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0 and channels <= 1024);
    e.setPipeline(p.conv_dynamic);
    inline for (.{ taps, input, static_w, projection, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ConvArgs{ .rows = rows, .channels = channels }, 5);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Paired K+V static short convolution in a single Metal compute dispatch.
pub fn convStaticKv(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    k_taps: Ref,
    k_weight: Ref,
    k_out: Ref,
    v_taps: Ref,
    v_weight: Ref,
    v_out: Ref,
    rows: u32,
    channels: u32,
) void {
    std.debug.assert(rows > 0 and channels > 0);
    e.setPipeline(p.kv_conv_static);
    inline for (.{ k_taps, k_weight, k_out, v_taps, v_weight, v_out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ConvArgs{ .rows = rows, .channels = channels }, 6);
    e.dispatchThreads(mtl.Size.of(channels, rows, 2), unit);
}

/// Paired K+V dynamic short convolution in a single Metal compute dispatch.
pub fn convDynamicKv(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    k_taps: Ref,
    k_input: Ref,
    k_static_w: Ref,
    k_projection: Ref,
    k_out: Ref,
    v_taps: Ref,
    v_input: Ref,
    v_static_w: Ref,
    v_projection: Ref,
    v_out: Ref,
    rows: u32,
    channels: u32,
) void {
    std.debug.assert(rows > 0 and channels > 0 and channels <= 1024);
    e.setPipeline(p.kv_conv_dynamic);
    inline for (.{ k_taps, k_input, k_static_w, k_projection, k_out, v_taps, v_input, v_static_w, v_projection, v_out }, 0..) |r, i|
        e.setBuffer(r.buf, r.off, i);
    e.setValue(ConvArgs{ .rows = rows, .channels = channels }, 10);
    e.dispatchGroups(mtl.Size.of(rows, 2, 1), mtl.Size.of(256, 1, 1));
}

const RouteArgs = extern struct { rows: u32 };
/// Router IDs and weights are sorted by ascending expert index for the deterministic expert sum.
pub fn router(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, bias: Ref, ids: Ref, weights: Ref, rows: u32) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.router);
    inline for (.{ x, weight, bias, ids, weights }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 5);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// `router`'s bits in two launches, a thread an (expert, row) score then a thread a row; scores: [rows][128] F32.
pub fn routerSplit(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, bias: Ref, scores: Ref, ids: Ref, weights: Ref, rows: u32) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.route_scores);
    inline for (.{ x, weight, scores }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 3);
    e.dispatchGroups(mtl.Size.of(experts / 32, rows, 1), unit);
    e.setPipeline(p.route_select);
    inline for (.{ scores, bias, ids, weights }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 4);
    e.dispatchGroups(mtl.Size.of((rows + 31) / 32, 1, 1), unit);
}

const ActArgs = extern struct { rows: u32, width: u32 };
pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, gu: Ref, out: Ref, rows: u32, width: u32) void {
    std.debug.assert(rows > 0 and width > 0);
    e.setPipeline(p.swiglu);
    inline for (.{ gu, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ActArgs{ .rows = rows, .width = width }, 2);
    e.dispatchThreads(mtl.Size.of(width, rows, 1), unit);
}

const LogitArgs = extern struct { rows: u32, vocab: u32 };
pub const LogitVariant = enum { r4, r8, r16, r32 };
pub const LogitLaunch = struct { variant: LogitVariant, tile_r: u32, threads: u32 };
pub fn logitLaunch(rows: u32) LogitLaunch {
    std.debug.assert(rows > 0 and rows <= 256);
    if (rows <= 4) return .{ .variant = .r4, .tile_r = 4, .threads = 128 };
    if (rows <= 8) return .{ .variant = .r8, .tile_r = 8, .threads = 128 };
    if (rows <= 16) return .{ .variant = .r16, .tile_r = 16, .threads = 128 };
    return .{ .variant = .r32, .tile_r = 32, .threads = 256 };
}

pub fn logits(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) void {
    std.debug.assert(rows > 0 and rows <= 256 and vocab > 0 and vocab <= 200058);
    const launch = logitLaunch(rows);
    const pipe = switch (launch.variant) {
        .r4 => p.logits_r4,
        .r8 => p.logits_r8,
        .r16 => p.logits_r16,
        .r32 => p.logits_r32,
    };
    e.setPipeline(pipe);
    inline for (.{ h, unembed, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab }, 3);
    const groups = mtl.Size.of((vocab + 31) / 32, (rows + launch.tile_r - 1) / launch.tile_r, 1);
    e.dispatchGroups(groups, mtl.Size.of(launch.threads, 1, 1));
}

pub fn logitsBaseline(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) void {
    std.debug.assert(rows > 0 and vocab > 0 and vocab <= 200058);
    e.setPipeline(p.logits);
    inline for (.{ h, unembed, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab }, 3);
    e.dispatchThreads(mtl.Size.of(vocab, rows, 1), unit);
}

const ExpertArgs = extern struct { rows: u32, channels: u32 };
pub fn expertPick(p: *const Pipelines, e: mtl.ComputeEncoder, all: Ref, ids: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0);
    e.setPipeline(p.expert_pick);
    inline for (.{ all, ids, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ExpertArgs{ .rows = rows, .channels = channels }, 3);
    e.dispatchThreads(mtl.Size.of(8, channels, rows), unit);
}

pub fn expertCombine(p: *const Pipelines, e: mtl.ComputeEncoder, picked: Ref, weights: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0);
    e.setPipeline(p.expert_combine);
    inline for (.{ picked, weights, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ExpertArgs{ .rows = rows, .channels = channels }, 3);
    e.dispatchThreads(mtl.Size.of(channels, rows, 1), unit);
}
