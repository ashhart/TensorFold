//! Cache operators accept projected inputs; no source projection or hidden allocations live here.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const View = @import("view.zig").View;
pub const cache = @import("metal_kv_state.zig");
pub const Kind = enum(u32) { window_fp8 = 0, main_fp4 = 1, index_fp4 = 2 };
pub const Pack = extern struct {
    rows: u32,
    width: u32,
    stride: u32,
    packed_stride: u32,
    block: u32,
    kind: Kind,
    first: u32 = 0,
    ring: u32 = 0,
    inplace: u32 = 0,
    pub fn validate(p: Pack) !void {
        const block: u32 = if (p.kind == .main_fp4) 16 else 32;
        const bytes = if (p.kind == .window_fp8) p.width else p.width / 2;
        if (p.rows == 0 or p.width == 0 or p.width % block != 0 or p.block != block or
            p.stride < p.width or @as(u64, p.packed_stride) < @as(u64, bytes) + p.width / block or
            p.inplace > 1 or @as(u64, p.first) + p.rows > std.math.maxInt(u32) or
            @as(u64, p.rows -| 1) * p.stride + p.width > std.math.maxInt(u32) or
            @as(u64, if (p.ring != 0) p.ring else p.first +| p.rows) * p.packed_stride >
                std.math.maxInt(u32) or (p.ring != 0 and p.rows > p.ring))
            return error.InvalidCachePackShape;
    }
};
pub const Pool = extern struct {
    rows: u32,
    width: u32 = 512,
    ratio: u32,
    position: u32,
    stride: u32,
    input_f32: u32,
    eps: f32 = 1e-20,
    pub fn completed(p: Pool) u32 {
        return (p.position + p.rows) / p.ratio - p.position / p.ratio;
    }
    pub fn validate(p: Pool) !void {
        if (p.rows == 0 or p.width == 0 or p.width > 512 or (p.ratio != 1 and p.ratio != 2) or
            p.stride < p.width or p.input_f32 != @intFromBool(p.ratio > 1) or
            !std.math.isFinite(p.eps) or p.eps <= 0 or
            @as(u64, p.position) + p.rows > cache.max_context or
            @as(u64, p.rows -| 1) * p.stride + p.width > std.math.maxInt(u32))
            return error.InvalidCompressorShape;
    }
};
pub const Pipelines = struct {
    pack: mtl.Pipeline,
    pool: mtl.Pipeline,
    tail: mtl.Pipeline,
    copy: mtl.Pipeline,
    publish: mtl.Pipeline,
    unpack: mtl.Pipeline,
    pub fn load(device: mtl.Device) !Pipelines {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv, options);
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.pack = try mtl.Pipeline.init(device, lib, "dsv41_pack", false);
        errdefer p.pack.deinit();
        p.pool = try mtl.Pipeline.init(device, lib, "dsv41_pool", false);
        errdefer p.pool.deinit();
        p.tail = try mtl.Pipeline.init(device, lib, "dsv41_pool_tail", false);
        errdefer p.tail.deinit();
        p.copy = try mtl.Pipeline.init(device, lib, "dsv41_copy", false);
        errdefer p.copy.deinit();
        p.publish = try mtl.Pipeline.init(device, lib, "dsv41_publish_window", false);
        errdefer p.publish.deinit();
        p.unpack = try mtl.Pipeline.init(device, lib, "dsv41_unpack_rows", false);
        return p;
    }
    pub fn deinit(p: *Pipelines) void {
        p.pack.deinit();
        p.pool.deinit();
        p.tail.deinit();
        p.copy.deinit();
        p.publish.deinit();
        p.unpack.deinit();
    }
    pub fn encodePack(p: *const Pipelines, enc: mtl.ComputeEncoder, x: View, storage: View, args: Pack) !void {
        try args.validate();
        try x.requireRows(args.rows, args.stride, args.width, 2);
        try storage.requireRows(if (args.ring != 0) args.ring else args.first + args.rows, args.packed_stride, args.width / (if (args.kind == .window_fp8) @as(u32, 1) else 2) +
            args.width / args.block, 1);
        enc.setPipeline(p.pack);
        enc.setBuffer(x.buffer, x.offset, 0);
        enc.setBuffer(storage.buffer, storage.offset, 1);
        enc.setValue(args, 2);
        enc.dispatchGroups(mtl.Size.of(args.width / args.block, args.rows, 1), mtl.Size.of(32, 1, 1));
        enc.barrier();
    }
    pub fn encodeUnpack(p: *const Pipelines, enc: mtl.ComputeEncoder, storage: View, out: View, args: Pack) !void {
        try args.validate();
        try out.requireRows(args.rows, args.stride, args.width, 2);
        try storage.requireRows(if (args.ring != 0) args.ring else args.first + args.rows, args.packed_stride, args.width / (if (args.kind == .window_fp8) @as(u32, 1) else 2) +
            args.width / args.block, 1);
        enc.setPipeline(p.unpack);
        enc.setBuffer(storage.buffer, storage.offset, 0);
        enc.setBuffer(out.buffer, out.offset, 1);
        enc.setValue(args, 2);
        enc.dispatchThreads(mtl.Size.of(args.width, args.rows, 1), mtl.Size.of(32, 1, 1));
        enc.barrier();
    }
    /// ratio2 values/gates are FP32; ratio1 values and RMS weight are BF16. Latent output is BF16.
    pub fn encodePool(p: *const Pipelines, enc: mtl.ComputeEncoder, values: View, gates: View, partial: View, norm_weight: View, latent: View, args: Pool) !void {
        try args.validate();
        try values.requireRows(args.rows, args.stride, args.width, if (args.input_f32 != 0) 4 else 2);
        if (args.ratio > 1) {
            try gates.requireRows(args.rows, args.stride, args.width, 4);
            try partial.require(@as(usize, 2) * args.ratio * args.width * 4, 4);
        }
        if (args.completed() != 0) {
            try norm_weight.require(@as(usize, args.width) * 2, 2);
            try latent.requireRows(args.completed(), args.width, args.width, 2);
            enc.setPipeline(p.pool);
            for ([_]View{ values, gates, partial, norm_weight, latent }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 5);
            enc.dispatchGroups(mtl.Size.of(args.completed(), 1, 1), mtl.Size.of(128, 1, 1));
            enc.barrier();
        }
        if (args.ratio > 1) {
            enc.setPipeline(p.tail);
            for ([_]View{ values, gates, partial }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 3);
            enc.dispatchThreads(mtl.Size.of(args.width, 1, 1), mtl.Size.of(128, 1, 1));
            enc.barrier();
        }
    }
    pub fn encodeCopy(p: *const Pipelines, enc: mtl.ComputeEncoder, src: View, dst: View, bytes: u32) void {
        enc.setPipeline(p.copy);
        enc.setBuffer(src.buffer, src.offset, 0);
        enc.setBuffer(dst.buffer, dst.offset, 1);
        enc.setValue(bytes, 2);
        enc.dispatchThreads(mtl.Size.of(bytes, 1, 1), mtl.Size.of(256, 1, 1));
        enc.barrier();
    }
    /// Copy only the final window; each destination slot has one writer even for large prefill chunks.
    pub fn publishWindow(p: *const Pipelines, enc: mtl.ComputeEncoder, chunk: View, ring: View, rows: u32, position: u32) !void {
        if (rows == 0 or @as(u64, position) + rows > cache.max_context) return error.NoCacheRows;
        try chunk.requireRows(rows, 528, 528, 1);
        try ring.require(128 * 528, 1);
        enc.setPipeline(p.publish);
        enc.setBuffer(chunk.buffer, chunk.offset, 0);
        enc.setBuffer(ring.buffer, ring.offset, 1);
        enc.setValue([4]u32{ rows, position, 528, 528 }, 2);
        enc.dispatchThreads(mtl.Size.of(528, @min(rows, 128), 1), mtl.Size.of(32, 1, 1));
        enc.barrier();
    }
    pub fn snapshot(p: *const Pipelines, enc: mtl.ComputeEncoder, state: *cache.State) !void {
        try state.begin();
        p.encodeCopy(enc, .{ .buffer = state.live }, .{ .buffer = state.backup }, @intCast(state.plan.mutableBytes()));
    }
    pub fn rollback(p: *const Pipelines, enc: mtl.ComputeEncoder, state: *cache.State) !void {
        try state.rollback();
        p.encodeCopy(enc, .{ .buffer = state.backup }, .{ .buffer = state.live }, @intCast(state.plan.mutableBytes()));
    }
};
