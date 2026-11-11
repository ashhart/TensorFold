//! Exact causal SWA plus shared compressed KV; fixed-buffer chunked prefill and decode use one path.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("core").frags;
const View = @import("view.zig").View;
pub const index = @import("csa2_index.zig");
pub const kv = @import("kv.zig");
pub const rope = @import("rope.zig");
/// One section of the dense KV pre-pass: a packed source, its stride, and where its rows land.
pub const DenseKv = extern struct { rows: u32, width: u32, stride: u32, first_row: u32, source: u32 };
comptime {
    @import("metal/index_mirror.zig").check(DenseKv, sources.deepseek_v41_csa2_nax, "DenseKv");
}

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
    splits: u32 = 1, // >1 only on a decode layer, where one threadgroup would scan every key alone
    /// The drafter's form -- every row sits at `position` and sees every chunk row.
    chunk_all: u32 = 0,
    comptime {
        @import("metal/index_mirror.zig").check(Params, sources.deepseek_v41_csa2, "Dsv41Attention");
    }
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
var merge_split_cache: ?u32 = null;

/// The merge's chunk split, default ON (four threadgroups). Zero restores the one-threadgroup merge.
fn mergeSplitOn() bool {
    if (merge_split_cache == null) {
        var v: u32 = 1;
        if (std.c.getenv("TF_DSV41_SPARSE_MERGE_SPLIT")) |raw| {
            v = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 1;
        }
        merge_split_cache = v;
    }
    return merge_split_cache.? != 0;
}

pub const Pipelines = struct {
    attention: mtl.Pipeline,
    nax: mtl.Pipeline, // the same attention over prompt chunks on the tensor units (csa2_nax.metal)
    tensor_units: bool,
    nax_min_rows: u32,

    dense_kv: mtl.Pipeline = undefined,
    dense_picked: mtl.Pipeline = undefined,
    merge: mtl.Pipeline = undefined,
    /// The merge with one accumulator chunk a threadgroup (four times the grid).
    merge_split: mtl.Pipeline = undefined,
    pub fn load(device: mtl.Device) !Pipelines {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const gpa = std.heap.page_allocator;
        const nax_text = try frags.source(device, gpa, sources.deepseek_v41_csa2_nax);
        defer gpa.free(nax_text);
        // TF_DSV41_ATTN_DENSE_VALUES compiles the sparse kernel's dense-table value path instead of its per-element.
        const dense_values: []const u8 = if (std.c.getenv("TF_DSV41_ATTN_DENSE_VALUES") != null) "#undef TF_SIMD_FRAGS\n" else "";
        const text = try std.mem.concat(gpa, u8, &.{ sources.deepseek_v41_kv, sources.deepseek_v41_csa2, dense_values, nax_text });
        defer gpa.free(text);
        const lib = try mtl.Library.fromSource(device, text, options);
        defer lib.deinit();
        var min_rows: u32 = 1;
        if (std.c.getenv("TF_DSV41_NAX_MIN_ROWS")) |v| min_rows = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1;
        const attention = try mtl.Pipeline.init(device, lib, "dsv41_sparse_attention", false);
        errdefer attention.deinit();
        const dense_kv = try mtl.Pipeline.init(device, lib, "dsv41_dense_kv", false);
        errdefer dense_kv.deinit();
        const dense_picked = try mtl.Pipeline.init(device, lib, "dsv41_dense_kv_picked", false);
        errdefer dense_picked.deinit();
        const merge = try mtl.Pipeline.init(device, lib, "dsv41_sparse_merge", false);
        const merge_split = try mtl.Pipeline.init(device, lib, "dsv41_sparse_merge_split", false);
        errdefer merge.deinit();
        const nax = try mtl.Pipeline.init(device, lib, "dsv41_sparse_nax", false);
        return .{
            .attention = attention,
            .nax = nax,
            .tensor_units = device.tensorUnits(),
            .dense_kv = dense_kv,
            .dense_picked = dense_picked,
            .merge = merge,
            .merge_split = merge_split,
            .nax_min_rows = min_rows,
        };
    }
    pub fn deinit(p: *Pipelines) void {
        p.attention.deinit();
        p.nax.deinit();
        p.dense_kv.deinit();
        p.dense_picked.deinit();
        p.merge.deinit();
        p.merge_split.deinit();
    }
    /// BF16 Q/output [rows,heads,width], FP32 sinks, I32 indices; publish this chunk to the ring only after dispatch.
    pub fn encode(p: *const Pipelines, enc: mtl.ComputeEncoder, q: View, ring: View, chunk: View, compressed: View, positions: View, sink: View, out: View, dense: View, partials: View, args: Params) !void {
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
        // Prompt chunks go through the tensor-unit kernel: its fragment layout wants width 512 and 16-head groups.
        const use_nax = p.tensor_units and args.width == 512 and args.heads % 16 == 0 and args.rows >= p.nax_min_rows;
        if (use_nax) {
            // Decode every visible key once: the window ring, this chunk, then the compressed picks.
            const sections = [_]struct { src: View, rows: u32, stride: u32, first: u32, source: u32 }{
                .{ .src = ring, .rows = @min(args.position, args.window), .stride = args.ring_stride, .first = 0, .source = 0 },
                .{ .src = chunk, .rows = args.rows, .stride = args.chunk_stride, .first = args.window, .source = 0 },
                .{ .src = compressed, .rows = args.compressed_rows, .stride = args.main_stride, .first = args.window + args.chunk_rows, .source = 1 },
            };
            enc.setPipeline(p.dense_kv);
            const repeats: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_DENSEKV") != null) 2 else 1;
            if (std.c.getenv("TF_DSV41_SKIP_DENSEKV") == null) {
                for (0..repeats) |_| {
                    // A one-row window reads only its picks, so the compressed section decodes those and no more.
                    const picked = args.rows == 1 and args.ratio != 0 and args.topk != 0 and args.compressed_rows > args.topk;
                    for (sections) |sec| {
                        if (sec.source == 1 and picked) {
                            enc.setPipeline(p.dense_picked);
                            enc.setBuffer(sec.src.buffer, sec.src.offset, 0);
                            enc.setBuffer(dense.buffer, dense.offset, 1);
                            enc.setValue(DenseKv{ .rows = @min(args.topk, args.compressed_rows), .width = args.width, .stride = sec.stride, .first_row = sec.first, .source = 1 }, 2);
                            enc.setBuffer(positions.buffer, positions.offset, 3);
                            enc.dispatchThreads(mtl.Size.of(@min(args.topk, args.compressed_rows), args.width / 4, 1), mtl.Size.of(32, 4, 1));
                            continue;
                        }
                        if (sec.rows == 0) continue;
                        enc.setBuffer(sec.src.buffer, sec.src.offset, 0);
                        enc.setBuffer(dense.buffer, dense.offset, 1);
                        enc.setValue(DenseKv{ .rows = sec.rows, .width = args.width, .stride = sec.stride, .first_row = sec.first, .source = sec.source }, 2);
                        enc.dispatchThreads(mtl.Size.of(sec.rows, args.width / 4, 1), mtl.Size.of(32, 4, 1));
                    }
                }
            }
            if (std.c.getenv("TF_DSV41_SKIP_SPARSE") != null) return;
            enc.setPipeline(p.nax);
            for ([_]View{ q, ring, chunk, compressed, positions, sink, out }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 7);
            enc.setBuffer(dense.buffer, dense.offset, 8);
            // A decode layer's scan takes a slice a threadgroup.
            var a2 = args;
            const splits: u32 = @min(args.splits, 32);
            a2.splits = splits;
            enc.setBuffer(partials.buffer, partials.offset, 9);
            enc.dispatchGroups(mtl.Size.of(args.heads / 16, args.rows, splits), mtl.Size.of(32, 4, 1));
            // TF_DSV41_DOUBLE_ATTN_SCANX -- the decode scan duplicated IN PLACE; it moves the reply, not same-output.
            if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_SCANX") != null) {
                enc.setPipeline(p.nax);
                for ([_]View{ q, ring, chunk, compressed, positions, sink, out }, 0..) |v, i|
                    enc.setBuffer(v.buffer, v.offset, i);
                enc.setValue(args, 7);
                enc.setBuffer(dense.buffer, dense.offset, 8);
                enc.setBuffer(partials.buffer, partials.offset, 9);
                enc.dispatchGroups(mtl.Size.of(args.heads / 16, args.rows, splits), mtl.Size.of(32, 4, 1));
            }
            // TF_DSV41_ATTN_SCAN_BARRIER -- the explicit barrier the csa2 NAX branch has no other place for.
            if (std.c.getenv("TF_DSV41_ATTN_SCAN_BARRIER") != null) enc.barrier();
            if (std.c.getenv("TF_DSV41_DOUBLE_SPARSE") != null) {
                const dr: u32 = args.rows;
                var a3 = args;
                a3.rows = dr;
                a3.position = args.position;
                a3.chunk_rows = dr;
                enc.setBuffer(out.buffer, out.offset + @as(usize, args.rows) * args.q_stride * 2, 6);
                enc.setValue(a3, 7);
                enc.dispatchGroups(mtl.Size.of(args.heads / 16, dr, splits), mtl.Size.of(32, 4, 1));
                enc.setBuffer(out.buffer, out.offset, 6);
            }
            if (splits > 1) {
                // TF_DSV41_SPARSE_MERGE_SPLIT -- the merge's own chunk split.
                const merge_split = mergeSplitOn();
                enc.setPipeline(if (merge_split) p.merge_split else p.merge);
                enc.setBuffer(partials.buffer, partials.offset, 0);
                enc.setBuffer(out.buffer, out.offset, 1);
                enc.setBuffer(sink.buffer, sink.offset, 2);
                enc.setValue(a2, 3);
                enc.dispatchGroups(mtl.Size.of(args.heads / 16, args.rows, if (merge_split) 4 else 1), mtl.Size.of(32, 4, 1));
                // TF_DSV41_DOUBLE_ATTN_MERGE -- the split merge's same-output duplicate.
                if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_MERGE") != null) {
                    // The duplicate must be the LANDED form (the split merge).
                    enc.setPipeline(if (merge_split) p.merge_split else p.merge);
                    enc.setBuffer(partials.buffer, partials.offset, 0);
                    enc.setBuffer(out.buffer, out.offset, 1);
                    enc.setBuffer(sink.buffer, sink.offset, 2);
                    enc.setValue(a2, 3);
                    enc.dispatchGroups(mtl.Size.of(args.heads / 16, args.rows, 1), mtl.Size.of(32, 4, 1));
                }
                // TF_DSV41_DOUBLE_ATTN_SCAN -- the decode scan duplicated AFTER the merge, where its write is unread.
                if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_SCAN") != null) {
                    enc.setPipeline(p.nax);
                    for ([_]View{ q, ring, chunk, compressed, positions, sink, out }, 0..) |v, i|
                        enc.setBuffer(v.buffer, v.offset, i);
                    enc.setValue(args, 7);
                    enc.setBuffer(dense.buffer, dense.offset, 8);
                    enc.setBuffer(partials.buffer, partials.offset, 9);
                    enc.dispatchGroups(mtl.Size.of(args.heads / 16, args.rows, splits), mtl.Size.of(32, 4, 1));
                }
            }
            return;
        }
        if (std.c.getenv("TF_DSV41_SKIP_SPARSE") != null) return;
        enc.setPipeline(p.attention);
        for ([_]View{ q, ring, chunk, compressed, positions, sink, out }, 0..) |v, i|
            enc.setBuffer(v.buffer, v.offset, i);
        enc.setValue(args, 7);
        enc.dispatchGroups(mtl.Size.of(args.heads, args.rows, 1), mtl.Size.of(32, 1, 1));
        // TF_DSV41_DOUBLE_ATTN_SCAN -- the decode scan's same-output duplicate.
        if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_SCAN") != null) {
            enc.setValue(args, 7);
            enc.dispatchGroups(mtl.Size.of(args.heads, args.rows, 1), mtl.Size.of(32, 1, 1));
        }
    }
};
