//! GLM-5.3-Flash's DSA indexer on CUDA (the Python family's sparse): each row keeps its best 512 4-token pools.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

/// sparse.POOL, TOPK_POOLS, SELECT_LIST, SCORE_LOOP_FROM, SCORE_LOOP; geometry.MLA_SELECT_ROWS.
pub const pool = 4;
pub const topk_pools = 512;
pub const select_list = 1024;
pub const score_loop_from = 64;
pub const score_loop = 32;
pub const select_rows = 512;
/// A row's token list: the 512 pools' tokens, then the incomplete pool's up to 3.
pub const width = topk_pools * pool + pool - 1;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// sparse.index_update: the window's LayerNormed index keys and gates into ik and ig at pos, completed pools into pk.
pub fn indexUpdate(set: glue.Set, stream: cuda.Stream, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, pos: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ k_raw, @as(i64, @intCast(k_stride)), gate, ln_w, ln_b, ik, ig, pos, @as(f32, 1e-6) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.index_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
    var b: cuda.Args = .{};
    inline for (.{ ik, ig, ape, pk, pos, @as(i64, @intCast(rows)) }) |v| b.add(v);
    try cuda.launch.launch(set.glue.pool_keys, .{ .grid = .{ .x = @intCast(rows / 4 + 2) }, .block = .{ .x = 32 } }, stream, &b);
}

/// sparse.pool_bucket: visible pools for rows pos .. pos + rows - 1, a power of two (at least 1024), at most `np_max`.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    return @min(np_max, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
}

/// sparse.score_columns: the pool columns selection reads for those rows (the last row's complete pools, at least 512).
pub fn scoreColumns(pos: usize, rows: usize, np_max: usize) usize {
    return @min(np_max, @max(topk_pools, (pos + rows) / pool));
}

/// select_tokens' 1/sqrt(heads) head-weight scale (32 heads: 1 / 5.656854249492381, as the Python family writes it).
pub fn headScale(heads: usize) f32 {
    if (heads == 32) return @floatCast(1.0 / 5.656854249492381);
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(heads))));
}

/// The scratch of one selectTokens: scores fp32 [min(rows, 512)][np], and with np past select_list the list rows.
pub const Scratch = struct { scores: u64, ls: u64, li: u64, pools: u64, at: u64 };

/// sparse.select_tokens: each row's tokens (i32 [rows][width], ascending, -1 padded), count 0 under the dense limit.
pub fn selectTokens(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, np_max_cap: usize, s: Scratch, tokens: u64, counts: u64) !void {
    const np_max = poolBucket(host_pos, rows, np_max_cap);
    const b = @min(rows, select_rows);
    const loop: usize = if (rows >= score_loop_from) score_loop else 1;
    const add = try ops.function("tf_glm_add_i32");
    var a: usize = 0;
    while (a < rows) : (a += b) {
        const n = @min(b, rows - a);
        var at = pos;
        if (a > 0) {
            var args: cuda.Args = .{};
            args.add(pos);
            args.add(@as(c_int, @intCast(a)));
            args.add(s.at);
            try cuda.launch.launch(add, .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &args);
            at = s.at;
        }
        const nb = cdiv(scoreColumns(host_pos + a, n, np_max), 64);
        var sa: cuda.Args = .{};
        inline for (.{ qi + a * heads * d * 2, wts + a * w_stride * 2, @as(i64, @intCast(w_stride)), pk, s.scores, at, @as(i64, @intCast(np_max)), @as(i64, @intCast(nb)), @as(f32, @floatCast(std.math.pow(f64, @floatFromInt(d), -0.5))), headScale(heads), @as(i64, @intCast(heads)), @as(c_int, @intCast(loop)) }) |v| sa.add(v);
        if (d != 128 or heads > 32) return error.IndexShape;
        try cuda.launch.launch(set.glue.scores, .{ .grid = .{ .x = @intCast(n), .y = @intCast(cdiv(nb, loop)) }, .block = .{ .x = 128 } }, stream, &sa);
        var sel: cuda.Args = .{};
        inline for (.{ s.scores, s.pools + a * topk_pools * 8, @as(i64, @intCast(np_max)), at, @as(c_int, topk_pools) }) |v| sel.add(v);
        try cuda.launch.launch(set.glue.select_rows, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 128 } }, stream, &sel);
    }
    const assemble = try ops.function("tf_glm_dsa_tokens");
    var args: cuda.Args = .{};
    args.add(s.pools);
    args.add(pos);
    args.add(@as(c_int, topk_pools));
    args.add(tokens);
    args.add(counts);
    try cuda.launch.launch(assemble, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
}

test "sparse.py's sizing rules" {
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(100, 1, 1 << 20));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(5000, 1, 1 << 20)); // 1251 visible pools
    try std.testing.expectEqual(@as(usize, 512), scoreColumns(100, 1, 1024));
    try std.testing.expectEqual(@as(usize, 1250), scoreColumns(5000, 1, 2048));
    try std.testing.expectEqual(@as(usize, 2051), width);
    try std.testing.expectEqual(@as(f32, @floatCast(1.0 / 5.656854249492381)), headScale(32));
}
