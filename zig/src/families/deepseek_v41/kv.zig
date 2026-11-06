//! Cache operators accept projected inputs; no source projection or hidden allocations live here.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const View = @import("view.zig").View;
pub const cache = @import("kv_state.zig");
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
/// ROUND-486: the window KV's RMS norm + RoPE + window-fp8 pack in one dispatch, at the pack's
/// threadgroup geometry (16 a row) with the shipped rms reduction order recomputed per block.
pub const Glue = extern struct {
    rows: u32,
    width: u32,
    stride: u32,
    packed_stride: u32,
    block: u32,
    position: u32,
    rotary: u32,
    write_y: u32,
    eps: f32,
    comptime {
        @import("metal/index_mirror.zig").check(Glue, sources.deepseek_v41_kv_glue, "Dsv41KvGlue");
    }
    pub fn validate(p: Glue) !void {
        if (p.rows == 0 or p.width != 512 or p.block != 32 or p.stride < p.width or
            p.rotary == 0 or p.rotary % 2 != 0 or p.rotary > p.width or p.write_y > 1 or
            @as(u64, p.packed_stride) < @as(u64, p.width) + p.width / p.block or
            !std.math.isFinite(p.eps) or p.eps <= 0 or
            @as(u64, p.position) + p.rows > cache.max_context or
            @as(u64, p.rows - 1) * p.stride + p.width > std.math.maxInt(u32))
            return error.InvalidCacheGlueShape;
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
    /// ROUND-367: the capture's slots (zero disables it): the pool writes every row it consumes, so a
    /// rollback can re-cut the group that straddles the new position without another forward.
    capture_rows: u32 = 0,
    comptime {
        @import("metal/index_mirror.zig").check(Pool, sources.deepseek_v41_kv, "Dsv41Pool");
    }
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
    glue: mtl.Pipeline,
    pool: mtl.Pipeline,
    tail: mtl.Pipeline,
    copy: mtl.Pipeline,
    publish: mtl.Pipeline,
    ring_move: mtl.Pipeline,
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
        p.ring_move = try mtl.Pipeline.init(device, lib, "dsv41_ring_move", false);
        errdefer p.ring_move.deinit();
        p.unpack = try mtl.Pipeline.init(device, lib, "dsv41_unpack_rows", false);
        errdefer p.unpack.deinit();
        const glue_lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv_glue, options);
        defer glue_lib.deinit();
        p.glue = try mtl.Pipeline.init(device, glue_lib, "dsv41_kv_glue", false);
        return p;
    }
    pub fn deinit(p: *Pipelines) void {
        p.pack.deinit();
        p.pool.deinit();
        p.tail.deinit();
        p.copy.deinit();
        p.publish.deinit();
        p.ring_move.deinit();
        p.unpack.deinit();
        p.glue.deinit();
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
    /// ROUND-486: norm + rope + window-fp8 pack, one dispatch, bit-identical to the three shipped
    /// kernels (kv_test's receipt). Y is written only when args.write_y is set; the engine passes the
    /// packed buffer as an inert placeholder, so the raw rows are read-only and there is no race.
    pub fn encodeGlue(p: *const Pipelines, enc: mtl.ComputeEncoder, x: View, weight: View, factors: View, packed_view: View, y: View, args: Glue) !void {
        try args.validate();
        try x.requireRows(args.rows, args.stride, args.width, 2);
        try weight.require(512 * 2, 2);
        try factors.require(@as(usize, args.position + args.rows) * (args.rotary / 2) * 8, 4);
        try packed_view.requireRows(args.rows, args.packed_stride, args.width + args.width / args.block, 1);
        enc.setPipeline(p.glue);
        enc.setBuffer(x.buffer, x.offset, 0);
        enc.setBuffer(weight.buffer, weight.offset, 1);
        enc.setBuffer(factors.buffer, factors.offset, 2);
        enc.setBuffer(packed_view.buffer, packed_view.offset, 3);
        enc.setBuffer(y.buffer, y.offset, 4);
        enc.setValue(args, 5);
        enc.dispatchGroups(mtl.Size.of(args.width / args.block, args.rows, 1), mtl.Size.of(256, 1, 1));
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
    pub const RingMove = extern struct { layers: u32, move_rows: u32, max_rows: u32, window: u32, position: u32, stride_words: u32, first: u32, to_stash: u32 };
    comptime {
        @import("metal/index_mirror.zig").check(RingMove, sources.deepseek_v41_kv, "Dsv41RingMove");
    }
    /// ROUND-398: move `move_rows` of the window ring's slots to/from the stash. `first` is the absolute
    /// row offset within the stash's max_rows; the ring slot is `(position + first + rr) % window`.
    pub fn encodeRingMove(p: *const Pipelines, enc: mtl.ComputeEncoder, dst: View, src: View, args: RingMove) !void {
        if (args.layers == 0 or args.move_rows == 0 or args.stride_words == 0 or args.first + args.move_rows > args.max_rows) return;
        enc.setPipeline(p.ring_move);
        enc.setBuffer(dst.buffer, dst.offset, 0);
        enc.setBuffer(src.buffer, src.offset, 1);
        enc.setValue(args, 2);
        enc.dispatchThreads(.{ .width = @as(usize, args.layers) * args.move_rows * args.stride_words }, .{ .width = 256 });
        enc.barrier();
    }

    /// ratio2 values/gates are FP32; ratio1 values and RMS weight are BF16. Latent output is BF16.
    /// ROUND-367: `capture` (zero slots disables it) receives every row the pool consumes -- the row's
    /// own values and its gate -- so a rollback can re-cut the group that straddles the new position.
    pub fn encodePool(p: *const Pipelines, enc: mtl.ComputeEncoder, values: View, gates: View, partial: View, norm_weight: View, latent: View, capture: View, args: Pool) !void {
        try args.validate();
        try values.requireRows(args.rows, args.stride, args.width, if (args.input_f32 != 0) 4 else 2);
        if (args.ratio > 1) {
            try gates.requireRows(args.rows, args.stride, args.width, 4);
            try partial.require(@as(usize, 2) * args.ratio * args.width * 4, 4);
        }
        if (args.capture_rows != 0) try capture.require(@as(usize, args.capture_rows) * 2 * args.width * 4, 4);
        if (args.completed() != 0) {
            try norm_weight.require(@as(usize, args.width) * 2, 2);
            try latent.requireRows(args.completed(), args.width, args.width, 2);
            enc.setPipeline(p.pool);
            for ([_]View{ values, gates, partial, norm_weight, latent, capture }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 6);
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
    /// ROUND-367: the no-re-feed port's restore. The pool's open row comes back from the capture: the
    /// tail dispatch with the captured row's own values and gate, at the row's own position.
    pub fn restorePool(p: *const Pipelines, enc: mtl.ComputeEncoder, values: View, gates: View, partial: View, args: Pool) !void {
        try args.validate();
        if (args.ratio <= 1) return;
        try values.requireRows(1, args.stride, args.width, 4);
        try gates.requireRows(1, args.stride, args.width, 4);
        try partial.require(@as(usize, 2) * args.ratio * args.width * 4, 4);
        enc.setPipeline(p.tail);
        for ([_]View{ values, gates, partial }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
        enc.setValue(args, 3);
        enc.dispatchThreads(mtl.Size.of(args.width, 1, 1), mtl.Size.of(128, 1, 1));
        enc.barrier();
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
