//! Exact causal SWA plus shared compressed KV; fixed-buffer chunked prefill and decode use one path.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const View = @import("view.zig").View;
pub const index = @import("csa2_index.zig");
pub const kv = @import("kv.zig");
pub const rope = @import("rope.zig");
pub const Params = extern struct {
    rows: u32,
    heads: u32 = 64,
    width: u32 = 512,
    q_stride: u32,
    position: u32,
    chunk_rows: u32,
    window: u32 = 128,
    ring_stride: u32 = 528,
    chunk_stride: u32 = 528,
    main_stride: u32 = 288,
    main_block: u32 = 16,
    ratio: u32,
    compressed_rows: u32,
    topk: u32 = 512,
    index_stride: u32 = 512,
    scale: f32 = 0.04419417382415922,
    pub fn validate(p: Params) !void {
        if (p.rows == 0 or p.heads == 0 or p.width == 0 or p.width > 512 or p.width % 32 != 0 or
            @as(u64, p.q_stride) < @as(u64, p.heads) * p.width or
            p.chunk_rows < p.rows or p.window != 128 or
            p.ring_stride < p.width + p.width / 32 or p.chunk_stride < p.width + p.width / 32 or
            p.main_stride < p.width / 2 + p.width / 16 or p.main_block != 16 or
            (p.ratio != 0 and p.ratio != 1 and p.ratio != 2) or p.topk > 512 or p.index_stride < p.topk or
            @as(u64, p.position) + p.rows > kv.cache.max_context or
            p.compressed_rows > kv.cache.max_context or !std.math.isFinite(p.scale) or p.scale <= 0 or
            @as(u64, p.rows -| 1) * p.q_stride + @as(u64, p.heads) * p.width > std.math.maxInt(u32) or
            @as(u64, p.rows) * p.index_stride > std.math.maxInt(u32) or
            @as(u64, p.rows) * p.chunk_stride > std.math.maxInt(u32) or
            @as(u64, p.window) * p.ring_stride > std.math.maxInt(u32) or
            @as(u64, p.compressed_rows) * p.main_stride > std.math.maxInt(u32))
            return error.InvalidSparseAttentionShape;
    }
};
pub const Pipelines = struct {
    attention: mtl.Pipeline,
    attention_parallel: mtl.Pipeline,
    parallel_enabled: bool = false,
    pub fn load(device: mtl.Device) !Pipelines {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv ++ sources.deepseek_v41_csa2, options);
        defer lib.deinit();
        const attention = try mtl.Pipeline.init(device, lib, "dsv41_sparse_attention", false);
        errdefer attention.deinit();
        const parallel = try mtl.Pipeline.init(device, lib, "dsv41_sparse_attention_parallel", false);
        const mode = std.c.getenv("TF_DSV41_PARALLEL_ATTENTION");
        return .{ .attention = attention, .attention_parallel = parallel, .parallel_enabled = if (mode) |v| std.mem.eql(u8, std.mem.span(v), "1") else false };
    }
    pub fn deinit(p: *Pipelines) void {
        p.attention.deinit();
        p.attention_parallel.deinit();
    }
    /// BF16 Q/output [rows,heads,width], FP32 sinks, I32 indices; publish this chunk to the ring only after dispatch.
    pub fn encode(p: *const Pipelines, enc: mtl.ComputeEncoder, q: View, ring: View, chunk: View, compressed: View, positions: View, sink: View, out: View, args: Params) !void {
        try args.validate();
        try q.requireRows(args.rows, args.q_stride, @as(usize, args.heads) * args.width, 2);
        try out.requireRows(args.rows, args.q_stride, @as(usize, args.heads) * args.width, 2);
        try sink.require(@as(usize, args.heads) * 4, 4);
        if (args.position != 0) try ring.requireRows(@min(args.position, args.window), args.ring_stride, args.width + args.width / 32, 1);
        try chunk.requireRows(args.rows, args.chunk_stride, args.width + args.width / 32, 1);
        if (args.ratio != 0 and args.compressed_rows != 0 and args.topk != 0) {
            try compressed.requireRows(args.compressed_rows, args.main_stride, args.width / 2 + args.width / 16, 1);
            try positions.requireRows(args.rows, args.index_stride, args.topk, 4);
        }
        const groups: u32 = if (p.parallel_enabled) @min(8, args.width / 32) else 1;
        const pipeline = if (p.parallel_enabled) p.attention_parallel else p.attention;
        if (pipeline.maxThreads() < groups * 32) return error.InvalidSparseAttentionThreads;
        enc.setPipeline(pipeline);
        for ([_]View{ q, ring, chunk, compressed, positions, sink, out }, 0..) |v, i|
            enc.setBuffer(v.buffer, v.offset, i);
        enc.setValue(args, 7);
        enc.dispatchGroups(mtl.Size.of(args.heads, args.rows, 1), mtl.Size.of(32 * groups, 1, 1));
    }
};

test {
    _ = @import("kv_test.zig");
    _ = @import("csa2_parallel_test.zig");
}
