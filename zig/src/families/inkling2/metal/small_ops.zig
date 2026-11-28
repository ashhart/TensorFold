//! Inkling 2 Small (H=4608, E=384, 8 blocks/taps) Metal pipelines and encoders.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
pub const plan = @import("inkling2_small_ops_plan");

pub const Ref = @import("inkling2_metal_ref").Ref;

pub const Pipelines = struct {
    embed_norm: mtl.Pipeline,
    ar: mtl.Pipeline,
    conv_dynamic: mtl.Pipeline,
    router: mtl.Pipeline,
    logits: mtl.Pipeline,
    attention_partial: mtl.Pipeline,
    capture_tap: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.small_ops);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const kinds = std.enums.values(plan.Kind);
        var pipes: [kinds.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for (kinds, 0..) |k, i| {
            pipes[i] = try mtl.Pipeline.init(device, lib, k.symbol(), false);
            made += 1;
            if (pipes[i].maxThreads() < k.threads()) return error.SmallOpsThreads;
        }
        return .{
            .embed_norm = pipes[0],
            .ar = pipes[1],
            .conv_dynamic = pipes[2],
            .router = pipes[3],
            .logits = pipes[4],
            .attention_partial = pipes[5],
            .capture_tap = pipes[6],
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
pub const ARRows = struct { rows: u32 = 1, blocks_stride: u32 = 0, partial_stride: u32 = 0, out_stride: u32 = 0 };
pub fn ar(p: *const Pipelines, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, batch: ARRows) void {
    std.debug.assert(sources_ > 0 and sources_ <= plan.max_sources and queries_ > 0 and queries_ <= 3 and batch.rows > 0);
    e.setPipeline(p.ar);
    inline for (.{ blocks, partial, queries, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ARArgs{
        .sources = sources_,
        .queries = queries_,
        .rows = batch.rows,
        .blocks_stride = batch.blocks_stride,
        .partial_stride = batch.partial_stride,
        .out_stride = batch.out_stride,
    }, 4);
    e.dispatchGroups(mtl.Size.of(queries_, batch.rows, 1), mtl.Size.of(256, 1, 1));
}

const ConvArgs = extern struct { rows: u32, channels: u32 };
pub fn convDynamic(p: *const Pipelines, e: mtl.ComputeEncoder, taps: Ref, input: Ref, static_w: Ref, projection: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0 and channels % 128 == 0);
    e.setPipeline(p.conv_dynamic);
    inline for (.{ taps, input, static_w, projection, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(ConvArgs{ .rows = rows, .channels = channels }, 5);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const RouteArgs = extern struct { rows: u32 };
pub fn router(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, bias: Ref, ids: Ref, weights: Ref, rows: u32) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.router);
    inline for (.{ x, weight, bias, ids, weights }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RouteArgs{ .rows = rows }, 5);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const LogitArgs = extern struct { rows: u32, vocab: u32 };
pub fn logits(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) void {
    std.debug.assert(rows > 0 and vocab > 0);
    e.setPipeline(p.logits);
    inline for (.{ h, unembed, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab }, 3);
    e.dispatchThreads(mtl.Size.of(vocab, rows, 1), mtl.Size.of(32, 1, 1));
}

const RuntimeArgs = extern struct { count: u32, boundary: u32 };
pub fn attentionPartial(p: *const Pipelines, e: mtl.ComputeEncoder, prev: Ref, x: Ref, blocks: Ref, out: Ref, rows: usize, boundary: usize) void {
    const count: u32 = @intCast(rows * plan.hidden);
    e.setPipeline(p.attention_partial);
    inline for (.{ prev, x, blocks, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RuntimeArgs{ .count = count, .boundary = @intCast(boundary) }, 4);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn captureTap(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, out: Ref, rows: usize, channel: usize) void {
    std.debug.assert(channel < plan.tap_channels);
    const count: u32 = @intCast(rows * plan.hidden);
    e.setPipeline(p.capture_tap);
    inline for (.{ x, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RuntimeArgs{ .count = count, .boundary = @intCast(channel) }, 2);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}
