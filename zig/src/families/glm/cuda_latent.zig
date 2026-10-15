//! GLM-5.3-Flash's MLA on the 512-wide latent cache (DSA layers), and Ling-3.1's with 64 roped columns after it.

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
    try rowBlocks(set.glue.absorb_q, stream, q, wk, out, rows, heads, lw / 32, .{});
}

/// Ling's absorb: q_nope the first d = 128 of each head's `q_stride` columns, out rows `out_stride` apart per head.
pub fn absorbQStrided(set: glue.Set, stream: cuda.Stream, q: u64, q_stride: usize, wk: u64, out: u64, out_stride: usize, rows: usize, heads: usize, d: usize, lw: usize) !void {
    if (d != 128 or lw != 512 or q_stride < d or out_stride < lw) return error.LatentShape;
    try rowBlocks(set.glue.absorb_q_128, stream, q, wk, out, rows, heads, lw / 32, .{ @as(i64, @intCast(q_stride)), @as(i64, @intCast(out_stride)) });
}

/// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize, extra: anytype) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) } ++ extra) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v (BF16 kv_b): out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw], dv 256|128.
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (lw != 512) return error.LatentShape;
    const f = switch (dv) {
        256 => set.glue.expand_v,
        128 => set.glue.expand_v_128,
        else => return error.LatentShape,
    };
    try rowBlocks(f, stream, ol, wv, out, rows, heads, dv / 16, .{});
}

/// Ling's 64 rope columns of `heads` heads a row: in at x + r x_row + h x_head, out likewise (bf16 elements).
pub const RopeRows = struct { x: u64, x_row: usize, x_head: usize, out: u64, o_row: usize, o_head: usize };

/// The rope (GPT-J pairs) at positions pos .. pos + rows - 1 (pos an i32 on device), inv_freq 32 fp32 on device.
pub fn rope(set: glue.Set, stream: cuda.Stream, v: RopeRows, rows: usize, heads: usize, pos: u64, inv_freq: u64) !void {
    var a: cuda.Args = .{};
    inline for (.{ v.x, @as(i64, @intCast(v.x_row)), @as(i64, @intCast(v.x_head)), v.out, @as(i64, @intCast(v.o_row)), @as(i64, @intCast(v.o_head)), pos, inv_freq }) |x| a.add(x);
    try cuda.launch.launch(set.glue.rope, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = rope_width / 2 } }, stream, &a);
}

/// Ling's qk_rope_head_dim: the roped columns after the latent in q and in the cache.
pub const rope_width = 64;

/// The rope's inv_freq: theta^(-2i / 64) for i < 32, from float64 rounded to fp32.
pub fn ropeFreqs(theta: f64) [rope_width / 2]f32 {
    var f: [rope_width / 2]f32 = undefined;
    for (&f, 0..) |*x, i| x.* = @floatCast(1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / rope_width));
    return f;
}

/// Ling's head-wise gate: o [rows][heads][dv] bf16 *= sigmoid(g fp32 [rows][g_stride], column h), in place.
pub fn headGate(set: glue.Set, stream: cuda.Stream, o: u64, g: u64, g_stride: usize, rows: usize, heads: usize, dv: usize) !void {
    if (dv == 0 or dv > 256) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ o, @as(i64, @intCast(heads)), g, @as(i64, @intCast(g_stride)) }) |x| a.add(x);
    try cuda.launch.launch(set.glue.head_gate, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = @intCast(dv) } }, stream, &a);
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
    try chunks(set, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, 0, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// Ling's latent.attention: qa and cache rows lw + rope_width wide (roped columns last), 16-head blocks; out lw wide.
pub fn attentionRoped(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32) !void {
    try chunks(set, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, rope_width, nch, 0, scale, heads_tile, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (sparse false: keys through pos + r) and _sparse_chunks (row r's `counts` tokens of `tokens`).
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, rw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512 or (rw != 0 and rw != rope_width)) return error.LatentShape;
    if (rw != 0 and hb != heads_tile) return error.HeadBlock;
    const f = if (rw != 0) set.glue.chunks_r16 else switch (hb) {
        heads_tile => set.glue.chunks_16,
        heads_tile_wide => set.glue.chunks_32,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    inline for (.{ qa, cache, pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    const shared = if (rw != 0) glue.chunksSmemRoped(heads_tile) else if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = chunksFor(w);
    try chunks(set, stream, qa, cache, 0, tokens, counts, p, rows, heads, lw, 0, nch, w, scale, headBlock(rows), true);
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
