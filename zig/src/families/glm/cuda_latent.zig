//! GLM-5.3-Flash's MLA on the 512-wide latent cache (DSA layers): the Python family's latent on BF16, row by row.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

/// latent.CHUNK, KT, HB, HB_WIDE, PROMPT_ROWS, PROMPT_RB.
pub const chunk = 512;
pub const key_tile = 32;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
pub const prompt_rows = 64;
pub const prompt_rb = 64;

/// latent.head_block: a rank's 32 heads a program in prompt chunks (each key read once), 16 in decode windows.
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) heads_tile_wide else heads_tile;
}

/// latent.chunks_for: 512-key chunks to cover `length` keys.
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// latent.absorb_q (BF16 kv_b): out [rows][heads][lw] bf16 = q [rows][heads][d] . wk [heads][d][lw], fp32 sums.
pub fn absorbQ(set: glue.Set, stream: cuda.Stream, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
    if (d != 256 or lw != 512) return error.LatentShape;
    try rowBlocks(set.glue.absorb_q, stream, q, wk, out, rows, heads, lw / 32);
}

/// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v (BF16 kv_b): out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw].
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (dv != 256 or lw != 512) return error.LatentShape;
    try rowBlocks(set.glue.expand_v, stream, ol, wv, out, rows, heads, dv / 16);
}

/// A latent cache holds BF16 rows, or E4M3 rows followed by one FP32 scale per row.
pub fn cacheBytes(capacity: usize, lw: usize, kv8: bool) !usize {
    if (kv8 and lw != 512) return error.LatentShape;
    const row = if (kv8) try std.math.add(usize, lw, 4) else try std.math.mul(usize, lw, 2);
    return std.math.mul(usize, capacity, row);
}

pub fn latentWriteFormat(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, scales: u64, pos: u64, rows: usize, lw: usize) !void {
    if (scales == 0) return latentWrite(set, stream, lat, lat_stride, cache, pos, rows, lw);
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, scales, pos }) |v| a.add(v);
    try cuda.launch.launch(set.glue.lat_write_fp8, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, stream, &a);
}

/// latent.latent_write: rows of `lat` (stride `lat_stride`) into slots pos .. pos + rows - 1 (pos an i32 on device).
pub fn latentWrite(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, pos: u64, rows: usize, lw: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, pos, @as(i64, @intCast(lw)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.lat_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The chunk partials of one attention call: po fp32 [nch][rows][heads][lw], pm and pl fp32 [nch][rows][heads].
pub const Partials = struct { po: u64, pm: u64, pl: u64 };

/// latent.attention: causal qa over the cache in `nch` 512-key chunks into out bf16; `dummy`: the merge's count i32.
pub fn attention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    return attentionFormat(set, stream, qa, cache, 0, pos, p, out, dummy, rows, heads, lw, nch, scale, hb);
}

pub fn attentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    try chunks(set, stream, qa, cache, scales, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (sparse false: keys through pos + r) and _sparse_chunks (row r's `counts` tokens of `tokens`).
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    const f = switch (hb) {
        heads_tile => if (scales == 0) set.glue.chunks_16 else set.glue.chunks_16_fp8,
        heads_tile_wide => if (scales == 0) set.glue.chunks_32 else set.glue.chunks_32_fp8,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (scales != 0) a.add(scales);
    inline for (.{ pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    return sparseAttentionFormat(set, stream, qa, cache, 0, tokens, counts, w, p, out, rows, heads, lw, scale);
}

pub fn sparseAttentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = chunksFor(w);
    try chunks(set, stream, qa, cache, scales, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, headBlock(rows), true);
    try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
}

fn merge(set: glue.Set, stream: cuda.Stream, p: Partials, out: u64, counts: u64, rows: usize, heads: usize, lw: usize, nch: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ p.po, p.pm, p.pl, out, counts, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &a);
}

test "latent.py's tiling rules" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(8));
    try std.testing.expectEqual(@as(usize, 32), headBlock(64));
    try std.testing.expectEqual(@as(usize, 3), chunksFor(1025));
    try std.testing.expectEqual(@as(usize, 1), chunksFor(512));
}
