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
    conv_static: mtl.Pipeline,
    conv_dynamic: mtl.Pipeline,
    router: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    logits: mtl.Pipeline,
    expert_pick: mtl.Pipeline,
    expert_combine: mtl.Pipeline,
    route_scores: mtl.Pipeline,
    /// The router's dots for up to 512 rows; launches in one encoder run in order, so they share it.
    route_dots: mtl.Buffer,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.glue);
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [10]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{
            "tf_inkling2_embed_norm",
            "tf_inkling2_ar",
            "tf_inkling2_conv_static",
            "tf_inkling2_conv_dynamic",
            "tf_inkling2_router",
            "tf_inkling2_swiglu",
            "tf_inkling2_logits",
            "tf_inkling2_expert_pick",
            "tf_inkling2_expert_combine",
            "tf_inkling2_route_scores",
        }, &pipes) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return .{
            .embed_norm = pipes[0],
            .ar = pipes[1],
            .conv_static = pipes[2],
            .conv_dynamic = pipes[3],
            .router = pipes[4],
            .swiglu = pipes[5],
            .logits = pipes[6],
            .expert_pick = pipes[7],
            .expert_combine = pipes[8],
            .route_scores = pipes[9],
            .route_dots = try device.buffer(@as(usize, max_rows) * experts * @sizeOf(f32), mtl.ResourceOptions.shared),
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
/// sources = completed BF16 blocks plus one unrounded F32 partial; output is one FP32 row per query.
pub fn ar(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32) void {
    arRows(p, e, blocks, partial, queries, out, sources_, queries_, 1);
}

/// As ar for `rows` rows in one launch: blocks [row][4][H], partial [row][H], output [query][row][H].
pub fn arRows(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, rows: u32) void {
    std.debug.assert(sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3 and rows > 0);
    e.setPipeline(p.ar);
    inline for (.{ blocks, partial, queries, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ARArgs{ .sources = sources_, .queries = queries_, .rows = rows }, 4);
    e.dispatchGroups(mtl.Size.of(queries_, rows, 1), mtl.Size.of(256, 1, 1));
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

const RouteArgs = extern struct { rows: u32 };
/// Router IDs and weights are sorted by ascending expert index for the deterministic expert sum.
pub fn router(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, bias: Ref, ids: Ref, weights: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= max_rows);
    e.setPipeline(p.route_scores);
    inline for (.{ x, weight }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setBuffer(p.route_dots, 0, 2);
    e.setValue(RouteArgs{ .rows = rows }, 3);
    e.dispatchGroups(mtl.Size.of(experts / 8, rows, 1), mtl.Size.of(256, 1, 1));
    e.barrier();
    e.setPipeline(p.router);
    e.setBuffer(p.route_dots, 0, 0);
    inline for (.{ bias, ids, weights }, 1..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Top-8 router selection reading precomputed FP32 dots from p.route_dots.
pub fn routerSelect(p: *const Pipelines, e: mtl.ComputeEncoder, bias: Ref, ids: Ref, weights: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= max_rows);
    e.setPipeline(p.router);
    e.setBuffer(p.route_dots, 0, 0);
    inline for (.{ bias, ids, weights }, 1..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 4);
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

const LogitArgs = extern struct { rows: u32, vocab: u32 };
pub fn logits(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) void {
    std.debug.assert(rows > 0 and vocab > 0 and vocab <= 200058);
    e.setPipeline(p.logits);
    inline for (.{ h, unembed, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab }, 3);
    e.dispatchGroups(mtl.Size.of((vocab + 7) / 8, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
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
