//! GLM-5.3-Flash on CUDA: latent attention (glm/latent.cu) and DSA indexer (glm/index.cu) launches, via kern.zig.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");
const K = kern.K;
const cdiv = kern.cdiv;
const nextPow2 = kern.nextPow2;
const Rows = kern.Rows;
const HL = kern.HL;
const DV = kern.DV;
const KV_LORA = kern.KV_LORA;
const SCALE = kern.SCALE;
const SPARSE_CHUNKS = kern.SPARSE_CHUNKS;
const POOL = kern.POOL;

// -- latent.cu -------------------------------------------------------------------------------------------------
pub fn absorbQ(k: *K, q: u64, wk: u64, qa: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(q);
    a.add(wk);
    a.add(qa);
    a.add(@as(i32, @intCast(R)));
    try k.go(try k.ext(.latent, "glm_absorb_q"), .{ .x = @intCast(HL), .y = @intCast(@divExact(KV_LORA, 64)), .z = @intCast(cdiv(R, 16)) }, 64, 0, &a);
}

pub fn expandV(k: *K, ol: u64, wv: u64, out: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(ol);
    a.add(wv);
    a.add(out);
    a.add(@as(i32, @intCast(R)));
    try k.go(try k.ext(.latent, "glm_expand_v"), .{ .x = @intCast(HL), .y = @intCast(@divExact(DV, 8)), .z = @intCast(cdiv(R, 16)) }, 256, 0, &a);
}

/// each row's latent into its stream's cache ``ci`` at its position.
pub fn latentWrite(k: *K, lat: u64, lat_stride: i64, t: Rows, ci: usize, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(lat);
    a.add(lat_stride);
    a.add(t.lcAt(ci));
    a.add(t.posAt());
    try k.go(try k.ext(.latent, "glm_lat_write"), .{ .x = @intCast(R) }, 64, 0, &a);
}

const ATT_SMEM: u32 = 32 * 1024 + 2 * 16 * 1024 + 2 * 32 * 16 * 4 + 32 * 16 * 2 + 3 * 32 * 4;

/// Dense causal attention of qa [R, H, 512] (rows below the dense limit) over ``nch`` 512-key chunks -> out.
pub fn denseAttention(k: *K, qa: u64, t: Rows, ci: usize, po: u64, pm: u64, pl: u64, out: u64, R: i64, nch: i64) !void {
    const f = try k.ext(.latent, "glm_attn_dense");
    try f.allowDynamicShared(ATT_SMEM);
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(t.lcAt(ci));
    a.add(t.posAt());
    a.add(po);
    a.add(pm);
    a.add(pl);
    a.add(@as(i32, @intCast(R)));
    a.add(SCALE);
    try k.go(f, .{ .x = @intCast(R), .y = @intCast(nch) }, 256, ATT_SMEM, &a);
    try merge(k, po, pm, pl, out, t.posAt(), 0, R, nch, false);
}

/// Attention of the rows with counts > 0 over their selected tokens, written into out (others untouched).
pub fn sparseAttention(k: *K, qa: u64, t: Rows, ci: usize, tokens: u64, counts: u64, po: u64, pm: u64, pl: u64, out: u64, R: i64) !void {
    const f = try k.ext(.latent, "glm_attn_sparse");
    try f.allowDynamicShared(ATT_SMEM);
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(t.lcAt(ci));
    a.add(tokens);
    a.add(counts);
    a.add(po);
    a.add(pm);
    a.add(pl);
    a.add(@as(i32, @intCast(R)));
    a.add(SCALE);
    try k.go(f, .{ .x = @intCast(R), .y = @intCast(SPARSE_CHUNKS) }, 256, ATT_SMEM, &a);
    try merge(k, po, pm, pl, out, 0, counts, R, SPARSE_CHUNKS, true);
}

fn merge(k: *K, po: u64, pm: u64, pl: u64, out: u64, pos: u64, counts: u64, R: i64, nch: i64, sparse: bool) !void {
    var a: cuda.Args = .{};
    a.add(po);
    a.add(pm);
    a.add(pl);
    a.add(out);
    a.add(pos);
    a.add(counts);
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(nch)));
    a.add(@as(i32, @intFromBool(sparse)));
    try k.go(try k.ext(.latent, "glm_attn_merge"), .{ .x = @intCast(R), .y = @intCast(HL) }, 128, 0, &a);
}

// -- index.cu: the DSA indexer ---------------------------------------------------------------------------------

/// The rows' index keys (LayerNorm) and gates into their streams' rings for cache ``ci``, then the pools they complete.
pub fn indexUpdate(k: *K, kr: u64, k_stride: i64, gate: u64, ln_w: u64, ln_b: u64, ape: u64, t: Rows, ci: usize, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(kr);
    a.add(k_stride);
    a.add(gate);
    a.add(ln_w);
    a.add(ln_b);
    a.add(t.ikAt(ci));
    a.add(t.igAt(ci));
    a.add(t.posAt());
    a.add(@as(f32, 1e-6));
    try k.go(try k.ext(.index, "glm_index_write"), .{ .x = @intCast(R) }, 128, 0, &a);
    var b: cuda.Args = .{};
    b.add(t.ikAt(ci));
    b.add(t.igAt(ci));
    b.add(ape);
    b.add(t.pkAt(ci));
    b.add(t.posAt());
    try k.go(try k.ext(.index, "glm_pool_keys"), .{ .x = @intCast(R) }, 128, 0, &b);
}

/// Pools to score up to position ``last``: the visible ones up to a power of two, at least 1024, at most ``np_max``.
pub fn poolBucket(last: i64, np_max: i64) i64 {
    const visible = @divFloor(last + 1, POOL) + 1;
    return @min(np_max, @max(1024, @as(i64, @intCast(nextPow2(@intCast(visible))))));
}

/// Each of n rows' scores over np pools -> scores [n, np] fp32.
pub fn scores(k: *K, qi: u64, w: u64, w_stride: i64, t: Rows, ci: usize, out: u64, n: i64, np: i64) !void {
    var a: cuda.Args = .{};
    a.add(qi);
    a.add(w);
    a.add(w_stride);
    a.add(t.pkAt(ci));
    a.add(out);
    a.add(t.posAt());
    a.add(@as(i32, @intCast(np)));
    a.add(@as(f32, 0.08838834764831845));
    a.add(@as(f32, 1.0 / 5.656854249492381));
    try k.go(try k.ext(.index, "glm_index_scores"), .{ .x = @intCast(n), .y = @intCast(cdiv(np, 64)) }, 128, 0, &a);
}

/// Each row's 512 best visible pools, ascending (int64).
pub fn selectRows(k: *K, sc: u64, out: u64, np: i64, t: Rows, n: i64) !void {
    var a: cuda.Args = .{};
    a.add(sc);
    a.add(out);
    a.add(@as(i32, @intCast(np)));
    a.add(t.posAt());
    try k.go(try k.ext(.index, "glm_index_select"), .{ .x = @intCast(n) }, 1024, 0, &a);
}

/// Each row's attended tokens and their count (0: a dense row).
pub fn sparseTokens(k: *K, pools: u64, t: Rows, rows: i64, tokens: u64, counts: u64) !void {
    var a: cuda.Args = .{};
    a.add(pools);
    a.add(t.posAt());
    a.add(tokens);
    a.add(counts);
    try k.go(try k.ext(.index, "glm_index_tokens"), .{ .x = @intCast(rows) }, 256, 0, &a);
}
