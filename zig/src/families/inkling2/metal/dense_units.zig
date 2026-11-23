//! Inkling 2 Mini's BF16 dense projections on the M5 tensor units, uniform across all widths 1..512.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");

pub const Ref = @import("inkling2_metal_ref").Ref;

pub const single_names = [3][:0]const u8{ "tf_ink2_dense_units_16", "tf_ink2_dense_units_32", "tf_ink2_dense_units_64" };
pub const fused6_names = [3][:0]const u8{ "tf_ink2_dense6_units_16", "tf_ink2_dense6_units_32", "tf_ink2_dense6_units_64" };

pub const Band = enum(u2) { b16 = 0, b32 = 1, b64 = 2 };

pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
pub const Dense6Args = extern struct { rows: u32, k: u32, nq: u32, nk: u32, nv: u32, nr: u32, ns: u32, ng: u32 };
pub const LoopBlendArgs = extern struct { rows: u32, stride: u32 };
pub const RelTableArgs = extern struct { rows: u32 };

pub const Call = struct { rows: u32, n: u32, k: u32, ldy: u32 = 0 };
pub const Call6 = struct { rows: u32, k: u32, nq: u32, nk: u32, nv: u32, nr: u32, ns: u32, ng: u32 };

pub const Buffers6 = struct {
    xq: Ref,
    xk: Ref,
    xv: Ref,
    wq: Ref,
    wk: Ref,
    wv: Ref,
    wr: Ref,
    ws: Ref,
    wg: Ref,
    yq: Ref,
    yk: Ref,
    yv: Ref,
    yr: Ref,
    ys: Ref,
    yg: Ref,
};

pub const Pipelines = struct {
    single: [3]mtl.Pipeline,
    fused6: [3]mtl.Pipeline,
    route_scores_16: mtl.Pipeline,
    route_scores_64: mtl.Pipeline,
    loop_blend: mtl.Pipeline,
    rel_table: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, src.dense_units);
    }

    /// Compiles the dense_units kernels from `text`; contract controls supply an edited shader source.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        var made: usize = 0;
        errdefer {
            for (out.single[0..@min(made, 3)]) |pipe| pipe.deinit();
            if (made > 3) for (out.fused6[0..@min(made - 3, 3)]) |pipe| pipe.deinit();
            if (made > 6) out.route_scores_16.deinit();
            if (made > 7) out.route_scores_64.deinit();
            if (made > 8) out.loop_blend.deinit();
        }
        for (single_names, &out.single) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < 256) return error.DenseUnitsThreads;
        }
        for (fused6_names, &out.fused6) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < 256) return error.DenseUnitsThreads;
        }
        out.route_scores_16 = try mtl.Pipeline.init(device, lib, "tf_ink2_route_scores_units_16", false);
        made += 1;
        if (out.route_scores_16.maxThreads() < 256) return error.DenseUnitsThreads;
        out.route_scores_64 = try mtl.Pipeline.init(device, lib, "tf_ink2_route_scores_units_64", false);
        made += 1;
        if (out.route_scores_64.maxThreads() < 256) return error.DenseUnitsThreads;
        out.loop_blend = try mtl.Pipeline.init(device, lib, "tf_ink2_loop_blend_units", false);
        made += 1;
        out.rel_table = try mtl.Pipeline.init(device, lib, "tf_ink2_rel_table_units", false);
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        p.rel_table.deinit();
        p.loop_blend.deinit();
        p.route_scores_64.deinit();
        p.route_scores_16.deinit();
        for (p.fused6) |pipe| pipe.deinit();
        for (p.single) |pipe| pipe.deinit();
    }
};

pub fn bandFor(rows: u32) Band {
    return if (rows <= 16) .b16 else if (rows <= 32) .b32 else .b64;
}

pub fn bandRows(band: Band) u32 {
    return switch (band) {
        .b16 => 16,
        .b32 => 32,
        .b64 => 64,
    };
}

pub fn tiles(n: u32) u32 {
    return (n + 31) / 32;
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.n == 0 or c.k == 0 or c.k % 128 != 0) return error.BadDenseShape;
    const ldy = if (c.ldy == 0) c.n else c.ldy;
    if (ldy < c.n) return error.BadDenseShape;
}

pub fn check6(c: Call6) !void {
    if (c.rows == 0 or c.k == 0 or c.k % 128 != 0) return error.BadDenseShape;
    if (c.nq == 0 and c.nk == 0 and c.nv == 0 and c.nr == 0 and c.ns == 0 and c.ng == 0) return error.BadDenseShape;
}

/// Dispatches `Y = bf16(X * W^T)` choosing the band variant for `c.rows` with identical reduction order.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, c: Call) !void {
    return encodeBand(p, e, bandFor(c.rows), x, w, y, c);
}

/// Dispatches an explicit band variant (`16`, `32`, or `64` rows per threadgroup) for contract testing.
pub fn encodeBand(p: *const Pipelines, e: mtl.ComputeEncoder, band: Band, x: Ref, w: Ref, y: Ref, c: Call) !void {
    try check(c);
    const ldy = if (c.ldy == 0) c.n else c.ldy;
    e.setPipeline(p.single[@backingInt(band)]);
    for ([_]Ref{ x, w, y }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(DenseArgs{ .rows = c.rows, .n = c.n, .k = c.k, .ldy = ldy }, 3);
    const br = bandRows(band);
    e.dispatchGroups(mtl.Size.of(tiles(c.n), (c.rows + br - 1) / br, 1), mtl.Size.of(256, 1, 1));
}

/// Dispatches the fused 6-projection attention linear kernel choosing the band variant for `c.rows`.
pub fn encode6(p: *const Pipelines, e: mtl.ComputeEncoder, bufs: Buffers6, c: Call6) !void {
    return encode6Band(p, e, bandFor(c.rows), bufs, c);
}

/// Dispatches an explicit fused 6-projection band variant (`16`, `32`, or `64` rows per threadgroup).
pub fn encode6Band(p: *const Pipelines, e: mtl.ComputeEncoder, band: Band, bufs: Buffers6, c: Call6) !void {
    try check6(c);
    e.setPipeline(p.fused6[@backingInt(band)]);
    const all = [_]Ref{
        bufs.xq, bufs.xk, bufs.xv, bufs.wq, bufs.wk, bufs.wv, bufs.wr, bufs.ws, bufs.wg,
        bufs.yq, bufs.yk, bufs.yv, bufs.yr, bufs.ys, bufs.yg,
    };
    for (all, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(Dense6Args{ .rows = c.rows, .k = c.k, .nq = c.nq, .nk = c.nk, .nv = c.nv, .nr = c.nr, .ns = c.ns, .ng = c.ng }, 15);
    const total_tiles = tiles(c.nq) + tiles(c.nk) + tiles(c.nv) + tiles(c.nr) + tiles(c.ns) + tiles(c.ng);
    const br = bandRows(band);
    e.dispatchGroups(mtl.Size.of(total_tiles, (c.rows + br - 1) / br, 1), mtl.Size.of(256, 1, 1));
}

/// Dispatches width-invariant FP32 router dot products [rows, 2304] x [128, 2304]^T on M5 tensor units.
pub fn encodeRouteScores(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: u32) void {
    const wide = rows > 16;
    e.setPipeline(if (wide) p.route_scores_64 else p.route_scores_16);
    for ([_]Ref{ x, w, y }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(DenseArgs{ .rows = rows, .n = 128, .k = 2304, .ldy = 128 }, 3);
    const br: u32 = if (wide) 64 else 16;
    e.dispatchGroups(mtl.Size.of(4, (rows + br - 1) / br, 1), mtl.Size.of(256, 1, 1));
}

/// Dispatches M5 tensor-unit loop gate forget/gain projections and exact four-step FP32 blend.
pub fn encodeLoopBlend(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, zeta: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32, stride: u32) void {
    e.setPipeline(p.loop_blend);
    for ([_]Ref{ h, m, zeta, forget, gain, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(LoopBlendArgs{ .rows = rows, .stride = stride }, 6);
    e.dispatchGroups(mtl.Size.of(72, (rows + 15) / 16, 1), mtl.Size.of(32, 1, 1));
}

/// Dispatches M5 tensor-unit relative bias table [rows_heads, 16] x [16, 1024] -> BF16.
pub fn encodeRelTable(p: *const Pipelines, e: mtl.ComputeEncoder, rel: Ref, proj: Ref, tab: Ref, rows_heads: u32) void {
    e.setPipeline(p.rel_table);
    for ([_]Ref{ rel, proj, tab }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(RelTableArgs{ .rows = rows_heads }, 3);
    e.dispatchGroups(mtl.Size.of(4, (rows_heads + 15) / 16, 1), mtl.Size.of(32, 1, 1));
}

test "dense_units band selection, tile math, shape validation, and ABI sizes" {
    try std.testing.expectEqual(Band.b16, bandFor(1));
    try std.testing.expectEqual(Band.b16, bandFor(16));
    try std.testing.expectEqual(Band.b32, bandFor(17));
    try std.testing.expectEqual(Band.b32, bandFor(32));
    try std.testing.expectEqual(Band.b64, bandFor(33));
    try std.testing.expectEqual(Band.b64, bandFor(512));
    try std.testing.expectEqual(@as(u32, 16), bandRows(.b16));
    try std.testing.expectEqual(@as(u32, 32), bandRows(.b32));
    try std.testing.expectEqual(@as(u32, 64), bandRows(.b64));
    try std.testing.expectEqual(@as(u32, 1), tiles(24));
    try std.testing.expectEqual(@as(u32, 1), tiles(32));
    try std.testing.expectEqual(@as(u32, 2), tiles(33));
    try std.testing.expectEqual(@as(u32, 96), tiles(3072));
    try check(.{ .rows = 1, .n = 24, .k = 2304 });
    try check(.{ .rows = 512, .n = 3072, .k = 3072 });
    try std.testing.expectError(error.BadDenseShape, check(.{ .rows = 0, .n = 24, .k = 2304 }));
    try std.testing.expectError(error.BadDenseShape, check(.{ .rows = 1, .n = 0, .k = 2304 }));
    try std.testing.expectError(error.BadDenseShape, check(.{ .rows = 1, .n = 24, .k = 64 }));
    try std.testing.expectError(error.BadDenseShape, check(.{ .rows = 1, .n = 24, .k = 2304, .ldy = 16 }));
    try check6(.{ .rows = 1, .k = 2304, .nq = 3072, .nk = 1024, .nv = 1024, .nr = 384, .ns = 24, .ng = 3072 });
    try std.testing.expectError(error.BadDenseShape, check6(.{ .rows = 1, .k = 2304, .nq = 0, .nk = 0, .nv = 0, .nr = 0, .ns = 0, .ng = 0 }));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DenseArgs));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Dense6Args));
}
