//! Tiled GPU scores and bounded online heaps; the memory footprint does not grow with context length.
const std = @import("std");
const max_context = @import("kv_state.zig").max_context;
const mtl = @import("metal");
const sources = @import("kernel_sources");
const View = @import("view.zig").View;
pub const score_tile_width = 4096;
pub const Params = extern struct {
    rows: u32,
    heads: u32 = 32,
    width: u32 = 128,
    q_stride: u32,
    weight_stride: u32,
    position: u32,
    ratio: u32,
    keys: u32,
    key_stride: u32 = 68,
    tile_start: u32 = 0,
    tile_width: u32 = score_tile_width,
    topk: u32 = 512,
    candidate_topk: u32 = 2048,
    block_size: u32 = 8,
    use_candidates: u32 = 0,
    make_candidates: u32 = 0,
    scale: f32 = 0.015625,
    pub fn validate(p: Params) !void {
        if (p.rows == 0 or p.heads == 0 or p.heads > 32 or p.width == 0 or p.width > 128 or p.width % 32 != 0 or
            p.q_stride < p.heads * p.width or p.weight_stride < p.heads or
            (p.ratio != 1 and p.ratio != 2) or p.key_stride < p.width / 2 + p.width / 32 or
            p.topk != 512 or p.candidate_topk != 2048 or p.block_size != 8 or
            p.tile_width != score_tile_width or p.use_candidates > 1 or p.make_candidates > 1 or
            (p.use_candidates != 0 and p.make_candidates != 0) or
            @as(u64, p.position) + p.rows > max_context or p.keys > max_context or
            !std.math.isFinite(p.scale) or p.scale <= 0 or
            @as(u64, p.rows) * p.q_stride > std.math.maxInt(u32) or
            @as(u64, p.rows) * p.weight_stride > std.math.maxInt(u32) or
            @as(u64, p.keys) * p.key_stride > std.math.maxInt(u32))
            return error.InvalidIndexerShape;
    }
};
pub const Scratch = struct {
    buffer: mtl.Buffer,
    rows: u32,
    pub fn bytes(rows: u32) usize {
        return @as(usize, rows) * (score_tile_width * 4 + 512 * 8 + 2048 * 8 + 512 * 4 + 528 + 32 * 68);
    }
    pub fn init(device: mtl.Device, rows: u32) !Scratch {
        if (rows == 0 or rows > 4096) return error.InvalidIndexerScratch;
        return .{ .buffer = try device.buffer(bytes(rows), mtl.ResourceOptions.shared), .rows = rows };
    }
    pub fn deinit(s: *Scratch) void {
        s.buffer.deinit();
    }
    pub fn scores(s: Scratch) View {
        return .{ .buffer = s.buffer };
    }
    pub fn heap(s: Scratch) View {
        return s.scores().shifted(@as(usize, s.rows) * score_tile_width * 4);
    }
    pub fn candidates(s: Scratch) View {
        return s.heap().shifted(@as(usize, s.rows) * 512 * 8);
    }
    pub fn positions(s: Scratch) View {
        return s.candidates().shifted(@as(usize, s.rows) * 2048 * 8);
    }
    pub fn windowChunk(s: Scratch) View {
        return s.positions().shifted(@as(usize, s.rows) * 512 * 4);
    }
    pub fn queryPacked(s: Scratch) View {
        return s.windowChunk().shifted(@as(usize, s.rows) * 528);
    }
};
pub const Pipelines = struct {
    score: mtl.Pipeline,
    clear: mtl.Pipeline,
    merge: mtl.Pipeline,
    finish: mtl.Pipeline,
    pub fn load(device: mtl.Device) !Pipelines {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv ++ sources.deepseek_v41_index, options);
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.score = try mtl.Pipeline.init(device, lib, "dsv41_index_score", false);
        errdefer p.score.deinit();
        p.clear = try mtl.Pipeline.init(device, lib, "dsv41_index_clear", false);
        errdefer p.clear.deinit();
        p.merge = try mtl.Pipeline.init(device, lib, "dsv41_index_merge", false);
        errdefer p.merge.deinit();
        p.finish = try mtl.Pipeline.init(device, lib, "dsv41_index_finish", false);
        return p;
    }
    pub fn deinit(p: *Pipelines) void {
        p.score.deinit();
        p.clear.deinit();
        p.merge.deinit();
        p.finish.deinit();
    }
    /// Q has already undergone BF16 RoPE and in-place FP4/E8M0 quantization; weights are BF16 projection output.
    pub fn encode(p: *const Pipelines, enc: mtl.ComputeEncoder, q: View, weights: View, keys: View, scratch: Scratch, args: Params) !void {
        try args.validate();
        if (args.rows > scratch.rows) return error.IndexerScratchExceeded;
        try (View{ .buffer = scratch.buffer }).require(Scratch.bytes(scratch.rows), 4);
        if (args.keys != 0) {
            try q.requireRows(args.rows, args.q_stride, @as(usize, args.heads) * args.width, 2);
            try weights.requireRows(args.rows, args.weight_stride, args.heads, 2);
            try keys.requireRows(args.keys, args.key_stride, args.width / 2 + args.width / 32, 1);
        }
        enc.setPipeline(p.clear);
        for ([_]View{ scratch.heap(), scratch.candidates() }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
        enc.setValue(args, 2);
        enc.dispatchThreads(mtl.Size.of(@max(args.topk, args.candidate_topk), args.rows, 1), mtl.Size.of(128, 1, 1));
        enc.barrier();
        const count = if (args.use_candidates != 0) args.candidate_topk * args.block_size else args.keys;
        var at: u32 = 0;
        while (at < count) : (at += score_tile_width) {
            var tile = args;
            tile.tile_start = at;
            enc.setPipeline(p.score);
            for ([_]View{ q, weights, keys, scratch.candidates(), scratch.scores() }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(tile, 5);
            enc.dispatchGroups(mtl.Size.of(score_tile_width, args.rows, 1), mtl.Size.of(32, 1, 1));
            enc.barrier();
            enc.setPipeline(p.merge);
            for ([_]View{ scratch.scores(), scratch.heap(), scratch.candidates() }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(tile, 3);
            enc.dispatchThreads(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(1, 1, 1));
            enc.barrier();
        }
        enc.setPipeline(p.finish);
        for ([_]View{ scratch.heap(), scratch.candidates(), scratch.positions() }, 0..) |v, i|
            enc.setBuffer(v.buffer, v.offset, i);
        enc.setValue(args, 3);
        enc.dispatchThreads(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(1, 1, 1));
        enc.barrier();
    }
};
