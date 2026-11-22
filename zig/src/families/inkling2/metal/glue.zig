//! Inkling 2 Mini glue on Metal: the production shader and its real encoder dispatches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
pub const plan = @import("inkling2_glue_plan");

pub const hidden: u32 = 2304;
pub const experts: u32 = 128;
pub const intermediate: u32 = 1152;
pub const top: u32 = 8;
pub const max_rows: u32 = 512;
pub const Ref = @import("inkling2_metal_ref").Ref;
const unit = mtl.Size.of(32, 1, 1);

pub const Pipelines = struct {
    embed_norm: mtl.Pipeline,
    ar: mtl.Pipeline,
    ar_bf16: mtl.Pipeline,
    conv_static: mtl.Pipeline,
    conv_dynamic: mtl.Pipeline,
    conv_dynamic_segs: mtl.Pipeline,
    router: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    logits: mtl.Pipeline,
    argmax: mtl.Pipeline,
    expert_pick: mtl.Pipeline,
    expert_combine: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.glue);
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [12]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{
            "tf_inkling2_embed_norm",
            "tf_inkling2_ar",
            "tf_inkling2_ar_bf16",
            "tf_inkling2_conv_static",
            "tf_inkling2_conv_dynamic",
            "tf_inkling2_conv_dynamic_segs",
            "tf_inkling2_router",
            "tf_inkling2_swiglu",
            "tf_inkling2_logits",
            "tf_inkling2_reduce_argmax",
            "tf_inkling2_expert_pick",
            "tf_inkling2_expert_combine",
        }, &pipes) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return .{
            .embed_norm = pipes[0],
            .ar = pipes[1],
            .ar_bf16 = pipes[2],
            .conv_static = pipes[3],
            .conv_dynamic = pipes[4],
            .conv_dynamic_segs = pipes[5],
            .router = pipes[6],
            .swiglu = pipes[7],
            .logits = pipes[8],
            .argmax = pipes[9],
            .expert_pick = pipes[10],
            .expert_combine = pipes[11],
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

const ARArgs = extern struct { sources: u32, queries: u32, rows: u32 };
fn arDispatch(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, rows: u32) void {
    std.debug.assert(sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3 and rows > 0);
    e.setPipeline(pipe);
    inline for (.{ blocks, partial, queries, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ARArgs{ .sources = sources_, .queries = queries_, .rows = rows }, 4);
    e.dispatchGroups(mtl.Size.of(queries_, rows, 1), mtl.Size.of(256, 1, 1));
}

/// sources = completed BF16 blocks plus one unrounded F32 partial; output is one FP32 row per query.
pub fn ar(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32) void {
    arRows(p, e, blocks, partial, queries, out, sources_, queries_, 1);
}

/// As ar for `rows` rows in one launch: blocks [row][4][H], partial [row][H], output [query][row][H].
pub fn arRows(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, rows: u32) void {
    arDispatch(p.ar, e, blocks, partial, queries, out, sources_, queries_, rows);
}

pub fn arBf16(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32) void {
    arRowsBf16(p, e, blocks, partial, queries, out, sources_, queries_, 1);
}

pub fn arRowsBf16(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, rows: u32) void {
    arDispatch(p.ar_bf16, e, blocks, partial, queries, out, sources_, queries_, rows);
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
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), unit);
}

pub const InkConvDynSeg = extern struct { taps: u64, input: u64, out: u64, row0: u32, rows: u32 };
const ConvDynSegsArgs = extern struct { n_segs: u32, total_rows: u32, channels: u32 };

/// Dynamic convolution batched across segments with global row dispatch.
pub fn convDynamicSegs(p: *const Pipelines, e: mtl.ComputeEncoder, segs: Ref, static_w: Ref, projection: Ref, n_segs: u32, total_rows: u32, channels: u32) void {
    std.debug.assert(n_segs > 0 and total_rows > 0 and channels > 0 and channels <= 1024);
    e.setPipeline(p.conv_dynamic_segs);
    inline for (.{ segs, static_w, projection }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ConvDynSegsArgs{ .n_segs = n_segs, .total_rows = total_rows, .channels = channels }, 3);
    e.dispatchGroups(mtl.Size.of(total_rows, 1, 1), unit);
}

const RouteArgs = extern struct { rows: u32 };
/// Router IDs and weights are sorted by ascending expert index for the deterministic expert sum.
pub fn router(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, bias: Ref, ids: Ref, weights: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= max_rows);
    e.setPipeline(p.router);
    inline for (.{ x, weight, bias, ids, weights }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 5);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const ActArgs = extern struct { rows: u32, width: u32 };
pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, gu: Ref, out: Ref, rows: u32, width: u32) void {
    std.debug.assert(rows > 0 and width > 0);
    e.setPipeline(p.swiglu);
    inline for (.{ gu, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ActArgs{ .rows = rows, .width = width }, 2);
    e.dispatchThreads(mtl.Size.of(width, rows, 1), unit);
}

const LogitArgs = extern struct { rows: u32, vocab: u32, flags: u32, part_stride: u32 };
pub fn logits(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) void {
    std.debug.assert(rows > 0 and vocab > 0 and vocab <= 200058);
    e.setPipeline(p.logits);
    inline for (.{ h, unembed, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setBuffer(out.buf, out.off, 3);
    e.setBuffer(out.buf, out.off, 4);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab, .flags = 1, .part_stride = 0 }, 5);
    e.dispatchGroups(mtl.Size.of((vocab + 7) / 8, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
}

const ArgmaxArgs = extern struct { rows: u32, parts: u32 };
pub fn logitsArgmax(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    h: Ref,
    unembed: Ref,
    out: Ref,
    part_vals: Ref,
    part_ids: Ref,
    out_ids: Ref,
    rows: u32,
    vocab: u32,
    write_logits: bool,
) void {
    std.debug.assert(rows > 0 and vocab > 0 and vocab <= 200058);
    const parts: u32 = (vocab + 7) / 8;
    const flags: u32 = if (write_logits) 3 else 2;
    e.setPipeline(p.logits);
    inline for (.{ h, unembed, out, part_vals, part_ids }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab, .flags = flags, .part_stride = parts }, 5);
    e.dispatchGroups(mtl.Size.of(parts, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
    e.barrier();
    argmax(p, e, part_vals, part_ids, out_ids, rows, parts);
}

pub fn argmax(p: *const Pipelines, e: mtl.ComputeEncoder, part_vals: Ref, part_ids: Ref, out_ids: Ref, rows: u32, parts: u32) void {
    std.debug.assert(rows > 0 and parts > 0);
    e.setPipeline(p.argmax);
    inline for (.{ part_vals, part_ids, out_ids }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ArgmaxArgs{ .rows = rows, .parts = parts }, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
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
