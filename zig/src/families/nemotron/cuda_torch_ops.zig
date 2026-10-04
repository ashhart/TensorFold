//! Launches of the torch-op replacements in zig/kernels/cuda/torch_ops, with their development launchers' geometry.

const std = @import("std");
const cuda = @import("cuda");

/// topk.cu's column tile, threads and workspace words a row (four 8-bit digit passes, then ordered compaction).
const topk_tile = 4096;
const topk_threads = 256;

fn tiles(columns: usize) usize {
    return (columns + topk_tile - 1) / topk_tile;
}

/// Scratch words `topk` needs for `rows` rows of `columns`.
pub fn topkScratchBytes(rows: usize, columns: usize) usize {
    return rows * (tiles(columns) * 260 + 3) * 4;
}

/// launch_shape.h's tf_launch_blocks: one block a `threads` elements, at most 65535.
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@min((count + threads - 1) / threads, 65535));
}

pub const Functions = struct {
    argmax: cuda.Function,
    hist: cuda.Function,
    digit: cuda.Function,
    count: cuda.Function,
    prefix: cuda.Function,
    compact: cuda.Function,
    to_f32: cuda.Function,
    lookup: cuda.Function,
    strided: cuda.Function,
    mamba_a: cuda.Function,

    /// The modules in build order: argmax, topk, pointwise, indexing, movement, nemotron_constants.
    pub fn resolve(m: []const cuda.Module) !Functions {
        return .{
            .argmax = try m[0].function("tf_argmax_rows_i32_kernel"),
            .hist = try m[1].function("tf_topk_f32_histogram_kernel"),
            .digit = try m[1].function("tf_topk_f32_choose_digit_kernel"),
            .count = try m[1].function("tf_topk_f32_count_kernel"),
            .prefix = try m[1].function("tf_topk_f32_prefix_kernel"),
            .compact = try m[1].function("tf_topk_f32_compact_kernel"),
            .to_f32 = try m[2].function("tf_bf16_to_f32_kernel"),
            .lookup = try m[3].function("tf_lookup_ids_kernel"),
            .strided = try m[4].function("tf_strided_copy_kernel"),
            .mamba_a = try m[5].function("tf_nemotron_mamba_a_f32_kernel"),
        };
    }
};

pub const Torch = struct {
    f: *const Functions,
    s: cuda.Stream,

    fn go(t: Torch, f: cuda.Function, grid: [2]usize, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]) }, .block = .{ .x = block } }, t.s, args);
    }

    /// torch.argmax(rows of bf16 logits) as int32: the first maximum, a NaN first.
    pub fn argmax(t: Torch, logits: u64, vocab: usize, ld: usize, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, vocab, ld }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(i32, 0)); // dtype 0: bf16
        try t.go(t.f.argmax, .{ rows, 1 }, 1024, &a);
    }

    /// `.float()` of `count` bf16 values (exact).
    pub fn toF32(t: Torch, in: u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try t.go(t.f.to_f32, .{ blocks(count, 256), 1 }, 256, &a);
    }

    /// torch.topk(rows of fp32, k, sorted=False): values above the k-th in column order, then ties in column order.
    pub fn topk(t: Torch, in: u64, columns: usize, rows: usize, k: usize, values: u64, indices: u64, scratch: u64) !void {
        const n = tiles(columns);
        const row_bytes: u64 = columns * 4;
        const hist = scratch;
        const threshold = hist + rows * n * 256 * 4;
        const remaining = threshold + rows * 4;
        const greater = remaining + rows * 4;
        const equal = greater + rows * n * 4;
        const greater_prefix = equal + rows * n * 4;
        const equal_prefix = greater_prefix + rows * n * 4;
        const greater_total = equal_prefix + rows * n * 4;
        var shift: u32 = 24;
        while (true) : (shift -= 8) {
            var h: cuda.Args = .{};
            for ([_]u64{ in, hist, threshold, columns, n, row_bytes, 4 }) |v| h.add(v);
            h.add(shift);
            try t.go(t.f.hist, .{ n, rows }, topk_threads, &h);
            var d: cuda.Args = .{};
            for ([_]u64{ hist, threshold, remaining, n }) |v| d.add(v);
            d.add(@as(u32, @intCast(k)));
            d.add(shift);
            try t.go(t.f.digit, .{ rows, 1 }, topk_threads, &d);
            if (shift == 0) break;
        }
        var c: cuda.Args = .{};
        for ([_]u64{ in, threshold, greater, equal, columns, n, row_bytes, 4 }) |v| c.add(v);
        try t.go(t.f.count, .{ n, rows }, topk_threads, &c);
        var p: cuda.Args = .{};
        for ([_]u64{ greater, equal, greater_prefix, equal_prefix, greater_total, n }) |v| p.add(v);
        try t.go(t.f.prefix, .{ rows, 1 }, 1, &p);
        var w: cuda.Args = .{};
        for ([_]u64{ in, values, indices, threshold, greater_prefix, equal_prefix, greater_total, columns, n, row_bytes, 4 }) |v| w.add(v);
        w.add(@as(u32, @intCast(k)));
        try t.go(t.f.compact, .{ n, rows }, topk_threads, &w);
    }

    /// id_map[local]: draft-head columns to token ids; an out-of-range column sets `invalid`.
    pub fn lookup(t: Torch, vocabulary: u64, size: usize, local: u64, global: u64, count: usize, invalid: u64) !void {
        var a: cuda.Args = .{};
        a.add(vocabulary);
        a.add(@as(u64, size));
        a.add(local);
        a.add(global);
        a.add(@as(u64, count));
        a.add(invalid);
        try t.go(t.f.lookup, .{ blocks(count, 256), 1 }, 256, &a);
    }

    /// `.contiguous()` of `rows` rows of `bytes` bytes, `src_ld` and `dst_ld` bytes apart.
    pub fn copyRows(t: Torch, src: u64, src_ld: usize, dst: u64, dst_ld: usize, bytes: usize, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        for ([_]u64{ count, 1, bytes, src_ld, bytes, dst_ld, bytes }) |v| a.add(v);
        try t.go(t.f.strided, .{ count, 1 }, 256, &a);
    }

    /// A = -exp(A_log) from fp32 A_log (startup only).
    pub fn mambaA(t: Torch, alog: u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(alog);
        a.add(out);
        a.add(@as(u64, count));
        try t.go(t.f.mamba_a, .{ blocks(count, 128), 1 }, 128, &a);
    }
};

test "topk scratch matches topk.cu's workspace" {
    try std.testing.expectEqual(@as(usize, 16 * (32 * 260 + 3) * 4), topkScratchBytes(16, 131072));
    try std.testing.expectEqual(@as(usize, (8 * 260 + 3) * 4), topkScratchBytes(1, 32768));
}
