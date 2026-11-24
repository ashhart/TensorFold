//! Inkling 2 Mini DFlash 2 Metal kernel pipelines and typed launch wrappers.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
const w = @import("dflash2_weights.zig");

pub const Ref = @import("inkling2_metal_ref").Ref;
pub const hidden = w.hidden;
pub const query_width = w.query_width;
pub const kv_width = w.kv_width;
pub const q_heads = w.q_heads;
pub const kv_heads = w.kv_heads;
pub const head_dim = w.head_dim;
pub const intermediate = w.intermediate;
pub const rank = w.rank;
pub const conv_groups = w.conv_groups;
pub const block_size = w.block_size;
pub const top_k = w.top_k;
pub const max_batch: usize = 8;

pub const Ring = struct { k: Ref, v: Ref };

pub const EmbedArgs = extern struct { rows: u32, width: u32, use_norm: u32, eps: f32 };
pub const NormArgs = extern struct { rows: u32, width: u32, eps: f32, add: u32 };
pub const ConvArgs = extern struct { rows: u32, width: u32, block: u32, groups: u32, side: u32 };
pub const CtxSeg = extern struct { row0: u32, rows: u32, pos0: u32, pad: u32 = 0 };
pub const CtxArgs = extern struct { rows: u32, segs: u32, ring: u32, eps: f32, theta: f32 };
pub const BlockQkvArgs = extern struct { rows: u32, block: u32, eps: f32, theta: f32, pos0: [max_batch]u32 };
pub const AttendStream = extern struct { anchor: u32, from: u32, lo: u32, retained: u32, keys: u32, chunks: u32 };
pub const AttendArgs = extern struct { rows: u32, block: u32, window: u32, ring: u32, max_chunks: u32, scale: f32 };
pub const ActArgs = extern struct { count: u32 };
pub const TopkArgs = extern struct { rows: u32, vocab: u32, k: u32, stride: u32 };
pub const WalkArgs = extern struct { slots: u32, rank: u32, top: u32, vocab: u32, stride: u32, row0: u32, softcap: f32 };

pub const Kind = enum {
    embed,
    norm,
    norm16,
    conv,
    ctx_kv,
    block_qkv,
    attend,
    attend_merge,
    silu_mul,
    topk16,
    walk,

    pub fn symbol(k: Kind) [:0]const u8 {
        return switch (k) {
            .embed => "tf_ink2_df2_embed",
            .norm => "tf_ink2_df2_norm",
            .norm16 => "tf_ink2_df2_norm16",
            .conv => "tf_ink2_df2_conv",
            .ctx_kv => "tf_ink2_df2_ctx_kv",
            .block_qkv => "tf_ink2_df2_block_qkv",
            .attend => "tf_ink2_df2_attend",
            .attend_merge => "tf_ink2_df2_attend_merge",
            .silu_mul => "tf_ink2_df2_silu_mul",
            .topk16 => "tf_ink2_df2_topk16",
            .walk => "tf_ink2_df2_walk",
        };
    }

    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("metal inkling2 dflash2 {s} {s}", .{ @tagName(v), comptime v.symbol() }),
        };
    }
};

pub const Pipelines = struct {
    embed: mtl.Pipeline,
    norm: mtl.Pipeline,
    norm16: mtl.Pipeline,
    conv: mtl.Pipeline,
    ctx_kv: mtl.Pipeline,
    block_qkv: mtl.Pipeline,
    attend: mtl.Pipeline,
    attend_merge: mtl.Pipeline,
    silu_mul: mtl.Pipeline,
    topk16: mtl.Pipeline,
    walk: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.inkling2_dflash2);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        var out: Pipelines = undefined;
        var made: usize = 0;
        errdefer inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            if (made > i) @field(out, name).deinit();
        };
        inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            const kind: Kind = @fromBackingInt(@intCast(i));
            @field(out, name) = try mtl.Pipeline.init(device, lib, kind.symbol(), false);
            made += 1;
        }
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
        p.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

pub fn embed(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, table: Ref, weight: Ref, out: Ref, a: EmbedArgs) void {
    e.setPipeline(p.embed);
    bind(e, &.{ ids, table, weight });
    e.setValue(a, 3);
    e.setBuffer(out.buf, out.off, 4);
    e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn norm(p: *const Pipelines, e: mtl.ComputeEncoder, resid: Ref, add_x: Ref, weight: Ref, out: Ref, a: NormArgs) void {
    e.setPipeline(p.norm);
    bind(e, &.{ resid, add_x, weight });
    e.setValue(a, 3);
    e.setBuffer(out.buf, out.off, 4);
    e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn norm16(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, out: Ref, a: NormArgs) void {
    e.setPipeline(p.norm16);
    bind(e, &.{ x, weight });
    e.setValue(a, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn conv(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, dyn: Ref, base: Ref, out: Ref, a: ConvArgs) void {
    e.setPipeline(p.conv);
    bind(e, &.{ x, dyn, base });
    e.setValue(a, 3);
    e.setBuffer(out.buf, out.off, 4);
    e.dispatchThreads(mtl.Size.of(a.width, a.rows, 1), mtl.Size.of(256, 1, 1));
}

pub fn ctxKv(p: *const Pipelines, e: mtl.ComputeEncoder, k_raw: Ref, v_raw: Ref, k_norm: Ref, rings: []const Ring, a: CtxArgs, segs: []const CtxSeg) void {
    std.debug.assert(segs.len > 0 and segs.len <= max_batch and rings.len == segs.len and a.segs == segs.len);
    var seg_buf: [max_batch]CtxSeg = @splat(.{ .row0 = 0, .rows = 0, .pos0 = 0 });
    @memcpy(seg_buf[0..segs.len], segs);
    e.setPipeline(p.ctx_kv);
    bind(e, &.{ k_raw, v_raw, k_norm });
    e.setValue(a, 3);
    e.setValue(seg_buf, 4);
    for (0..max_batch) |s| {
        const ring = rings[if (s < rings.len) s else 0];
        e.setBuffer(ring.k.buf, ring.k.off, 5 + s);
        e.setBuffer(ring.v.buf, ring.v.off, 13 + s);
    }
    e.dispatchGroups(mtl.Size.of(w.kv_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
}

pub fn blockQkv(p: *const Pipelines, e: mtl.ComputeEncoder, q_raw: Ref, k_raw: Ref, v_raw: Ref, q_norm: Ref, k_norm: Ref, q: Ref, kb: Ref, vb: Ref, a: BlockQkvArgs) void {
    e.setPipeline(p.block_qkv);
    bind(e, &.{ q_raw, k_raw, v_raw, q_norm, k_norm });
    e.setValue(a, 5);
    e.setBuffer(q.buf, q.off, 6);
    e.setBuffer(kb.buf, kb.off, 7);
    e.setBuffer(vb.buf, vb.off, 8);
    e.dispatchGroups(mtl.Size.of(w.q_heads + w.kv_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
}

pub fn attendPart(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, rings: []const Ring, part: Ref, a: AttendArgs, streams: []const AttendStream) void {
    std.debug.assert(streams.len > 0 and streams.len <= max_batch and rings.len == streams.len);
    var stream_buf: [max_batch]AttendStream = @splat(.{ .anchor = 0, .from = 0, .lo = 0, .retained = 0, .keys = 0, .chunks = 0 });
    @memcpy(stream_buf[0..streams.len], streams);
    e.setPipeline(p.attend);
    bind(e, &.{ q, kb, vb });
    e.setValue(a, 3);
    e.setValue(stream_buf, 4);
    e.setBuffer(part.buf, part.off, 5);
    for (0..max_batch) |s| {
        const ring = rings[if (s < rings.len) s else 0];
        e.setBuffer(ring.k.buf, ring.k.off, 6 + s);
        e.setBuffer(ring.v.buf, ring.v.off, 14 + s);
    }
    e.dispatchGroups(mtl.Size.of(w.kv_heads, a.rows, a.max_chunks), mtl.Size.of(32, 1, 1));
}

pub fn attendMerge(p: *const Pipelines, e: mtl.ComputeEncoder, part: Ref, out: Ref, a: AttendArgs, streams: []const AttendStream) void {
    std.debug.assert(streams.len > 0 and streams.len <= max_batch);
    var stream_buf: [max_batch]AttendStream = @splat(.{ .anchor = 0, .from = 0, .lo = 0, .retained = 0, .keys = 0, .chunks = 0 });
    @memcpy(stream_buf[0..streams.len], streams);
    e.setPipeline(p.attend_merge);
    e.setBuffer(part.buf, part.off, 0);
    e.setValue(a, 1);
    e.setValue(stream_buf, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchGroups(mtl.Size.of(w.q_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
}

pub fn attend(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, rings: []const Ring, part: Ref, out: Ref, a: AttendArgs, streams: []const AttendStream) void {
    attendPart(p, e, q, kb, vb, rings, part, a, streams);
    attendMerge(p, e, part, out, a, streams);
}

pub fn siluMul(p: *const Pipelines, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, count: u32) void {
    e.setPipeline(p.silu_mul);
    bind(e, &.{ gate, up });
    e.setValue(ActArgs{ .count = count }, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn topk16(p: *const Pipelines, e: mtl.ComputeEncoder, logits: Ref, vals: Ref, ids: Ref, a: TopkArgs) void {
    e.setPipeline(p.topk16);
    e.setBuffer(logits.buf, logits.off, 0);
    e.setValue(a, 1);
    e.setBuffer(vals.buf, vals.off, 2);
    e.setBuffer(ids.buf, ids.off, 3);
    e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(512, 1, 1));
}

pub fn walk(p: *const Pipelines, e: mtl.ComputeEncoder, vals: Ref, ids: Ref, hid: Ref, pred: Ref, succ: Ref, seq: Ref, a: WalkArgs, streams: u32) void {
    std.debug.assert(streams > 0 and streams <= max_batch);
    e.setPipeline(p.walk);
    bind(e, &.{ vals, ids, hid, pred, succ });
    e.setValue(a, 5);
    e.setBuffer(seq.buf, seq.off, 6);
    e.dispatchGroups(mtl.Size.of(streams, 1, 1), mtl.Size.of(w.rank, 1, 1));
}

test "DFlash2 kernel argument ABI sizes match Metal structs" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(EmbedArgs));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(NormArgs));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(ConvArgs));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(CtxSeg));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(CtxArgs));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(BlockQkvArgs));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(AttendStream));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(AttendArgs));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(TopkArgs));
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(WalkArgs));
}
