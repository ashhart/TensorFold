//! GLM-5.3-Flash's BF16 matmuls on CUDA (attention, router, shared experts): the Python family's B16 path.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");

/// qmm.SPLIT_TARGET, B16_SPLIT_COLS, B16_BK, geometry.MLA_B16_SLICES and MLA_B16_ROWS_FROM.
pub const split_target = 192;
pub const split_cols = 64;
pub const bk = 64;
pub const max_slices = 8;
pub const rows_from = 128;

/// qmm.b16_split_k: K slices doubled while the 64-column tiles stay under 192 and each slice keeps 4 steps of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + split_cols - 1) / split_cols;
    const steps = k / bk;
    var sk: usize = 1;
    while (sk < max_slices and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Triton's one-row _bmm (aot.json M = 1, GB10): BLOCK_N, stages by (n, k), one stage -> two; else tf_glm_bmm_decode.
const DecodeTile = struct { n: usize, k: usize, bn: u32, st: u32 };
const decode_tiles = [_]DecodeTile{
    .{ .n = 4096, .k = 4096, .bn = 32, .st = 2 },  .{ .n = 4096, .k = 128, .bn = 32, .st = 2 },
    .{ .n = 4096, .k = 2816, .bn = 32, .st = 2 },  .{ .n = 4096, .k = 5632, .bn = 32, .st = 2 },
    .{ .n = 5376, .k = 1536, .bn = 32, .st = 2 },  .{ .n = 8192, .k = 1536, .bn = 32, .st = 2 },
    .{ .n = 12576, .k = 4096, .bn = 32, .st = 3 }, .{ .n = 4096, .k = 1024, .bn = 32, .st = 3 },
    .{ .n = 160, .k = 4096, .bn = 32, .st = 3 },   .{ .n = 1280, .k = 4096, .bn = 32, .st = 3 },
    .{ .n = 2816, .k = 128, .bn = 32, .st = 3 },   .{ .n = 2688, .k = 128, .bn = 32, .st = 3 },
    .{ .n = 4096, .k = 8192, .bn = 64, .st = 2 },  .{ .n = 4096, .k = 704, .bn = 64, .st = 2 },
    .{ .n = 4096, .k = 16384, .bn = 64, .st = 2 }, .{ .n = 4096, .k = 1536, .bn = 64, .st = 2 },
    .{ .n = 4096, .k = 2048, .bn = 64, .st = 2 },  .{ .n = 16384, .k = 1536, .bn = 64, .st = 2 },
    .{ .n = 2048, .k = 4096, .bn = 64, .st = 2 },  .{ .n = 1408, .k = 4096, .bn = 64, .st = 2 },
    .{ .n = 8192, .k = 128, .bn = 64, .st = 2 },   .{ .n = 24896, .k = 4096, .bn = 64, .st = 2 },
    .{ .n = 77440, .k = 4096, .bn = 64, .st = 2 }, .{ .n = 154880, .k = 4096, .bn = 64, .st = 2 },
};

fn decodeTile(n: usize, k: usize) ?DecodeTile {
    for (decode_tiles) |t| if (t.n == n and t.k == k) return t;
    return null;
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// The fp32 split-K partials a window of `m` rows needs (0 when it sums in registers or takes one slice).
pub fn partBytes(m: usize, n: usize, k: usize) usize {
    const sk = splitK(n, k);
    return if (m >= rows_from or sk == 1) 0 else sk * m * n * 4;
}

/// qmm.matmul for a B16 weight: out [m][n] bf16 (fp32 with `wide`) = x . w; 128+ rows sum in registers, else `part`.
pub fn matmul(set: glue.Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    if (k % bk != 0) return error.KNotMultipleOf64;
    if (set.mm) |mk| return mm.matmul(mk, stream, x, x_stride, .{ .w = w, .n = n, .k = k }, out, part, m, wide); // tiled weights
    const sk = splitK(n, k);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(w);
    a.add(out);
    if (m < rows_from) a.add(if (sk > 1) part else out);
    inline for (.{ @as(i64, @intCast(m)), @as(i64, @intCast(x_stride)), @as(i64, @intCast(n)), @as(i64, @intCast(k)), @as(c_int, @intCast(sk)), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    if (m >= rows_from and x % 16 == 0 and x_stride % 8 == 0 and n % 8 == 0 and out % 16 == 0) { // the cp.async tiles
        // One slice past 256 columns: 128 x 128 blocks, two an SM; K slices or narrow outputs: 128 x 64 blocks.
        const one = sk == 1 and n > 256;
        const f = if (one) set.glue.bmm_tile1 else set.glue.bmm_tile;
        const bn: usize = if (one) 128 else 64;
        const shared: u32 = if (one) glue.bmm_tile1_smem[@intFromBool(wide)] else glue.bmm_tile_smem;
        return cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(cdiv(m, 128)), .y = @intCast(cdiv(n, bn)) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
    }
    if (m >= rows_from) return cuda.launch.launch(set.glue.bmm_rows, .{ .grid = .{ .x = @intCast(cdiv(m, 128)), .y = @intCast(cdiv(n, 64)) }, .block = .{ .x = 128 } }, stream, &a);
    if (decodeTile(n, k)) |t| {
        if (x % 16 == 0 and x_stride % 8 == 0 and n % 8 == 0 and out % 16 == 0 and part % 16 == 0) {
            const f = switch (t.bn * 10 + t.st) {
                321 => set.glue.bmm_d32s1,
                322 => set.glue.bmm_d32s2,
                323 => set.glue.bmm_d32s3,
                642 => set.glue.bmm_d64s2,
                else => unreachable,
            };
            const shared: u32 = t.st * (16 + t.bn) * bk * 2; // Tile<16, bn, 64, st, 1, 4>'s stages (the output tile is smaller)
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(cdiv(m, 16)), .y = @intCast(cdiv(n, t.bn)), .z = @intCast(sk) }, .block = .{ .x = 128 }, .shared = shared }, stream, &a);
            if (sk > 1) try glue.sliceSum(set, stream, part, out, m * n, sk, wide);
            return;
        }
    }
    try cuda.launch.launch(set.glue.bmm_decode, .{ .grid = .{ .x = @intCast(cdiv(m, 16)), .y = @intCast(cdiv(n, 64)), .z = @intCast(sk) }, .block = .{ .x = 128 } }, stream, &a);
    if (sk > 1) try glue.sliceSum(set, stream, part, out, m * n, sk, wide);
}

/// qmm.group_sums: (m, k) bf16 rows (stride `x_stride`) -> (m, k / 64) fp32 sums of each 64 inputs.
pub fn groupSums(set: glue.Set, stream: cuda.Stream, x: u64, x_stride: usize, xs: u64, m: usize, k: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xs, @as(i64, @intCast(x_stride)), @as(i64, @intCast(k / 64)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.group_sums, .{ .grid = .{ .x = @intCast(m), .y = @intCast(cdiv(k / 64, 16)) }, .block = .{ .x = 128 } }, stream, &a);
}

test "K slices as qmm.py picks them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12576, 4096)); // 197 tiles: past the target at one slice
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 4096)); // 64 tiles: 2, then 4 (256 past the target)
    try std.testing.expectEqual(@as(usize, 8), splitK(160, 4096));
    try std.testing.expectEqual(@as(usize, 1), splitK(4096, 128)); // two steps: no slice keeps four
    try std.testing.expectEqual(@as(usize, 0), partBytes(200, 4096, 4096));
}
