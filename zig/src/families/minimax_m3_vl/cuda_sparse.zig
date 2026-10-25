//! MiniMax M3.1's MSA sparse-indexer on CUDA (the exllamav3-fork m3_sparse_attn contract, by hand):
//! launches minimax_m3.cu's four kernels. The numerics' grading reference is this family's CPU
//! mirror `sparse.zig` (fp32 scoring, right-open causal, +inf boosts post-reduction, ties at the
//! smaller id, left-packed, -1 right-pad, key-only side plane); device parity receipts are Stage 3.

const std = @import("std");
const cuda = @import("cuda");

/// The contract numbers (asserted in config.zig from the pack): these are the fork's.
pub const d = 128; // index_head_dim == qkv head_dim
pub const block = 128; // index_block_size: keys per scored block; a 128-run never straddles page 256
pub const topk = 16; // index_topk_blocks, the fixed selection width
pub const page = 256; // the paged fp16 cache's in-tree PAGE_SIZE
pub const max_nb = 1024; // the topk kernel's mark bitmap bound: NB must stay <= this

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// A content length's block count (slot anchor): NB = ceil(Sk / 128), at least one — a served row
/// always scores its own key (Sk >= 1), so an empty context still carries the one block.
pub fn nbOf(keys: usize) usize {
    const n = cdiv(keys, block);
    return if (n == 0) 1 else n;
}

/// One split's half-open slice of the fixed-width selection: per = ceil(k / nsplit), bounded by k.
/// The nsplit split count is bounded by topk (the wrapper contract).
pub fn splitBounds(k: usize, nsplit: usize, split: usize) [2]usize {
    const per = cdiv(k, nsplit);
    const lo = split * per;
    const hi = @min(lo + per, k);
    return .{ lo, hi };
}

/// The decode scratch's stride: per (batch, q head, split) row, D accumulators + m + l.
pub const scratch_elts = d + 2;

/// The family's four kernels from the minimax_m3 image.
pub const Kernels = struct {
    scores: cuda.Function,
    topk: cuda.Function,
    decode: cuda.Function,
    combine: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        k.scores = try m.function("tf_m3_scores");
        k.topk = try m.function("tf_m3_topk");
        k.decode = try m.function("tf_m3_decode");
        k.combine = try m.function("tf_m3_decode_combine");
        return k;
    }
};

/// m3_indexer_scores: the whole score table in one launch — a program per (query row, key block,
/// batch); slot-anchored blocks, fp32 scoring, boosts override post-reduction, no Sq x Sk exists.
pub fn scores(
    ks: Kernels,
    stream: cuda.Stream,
    q: u64, // (B, H, Sq, D) fp16 index queries
    k: u64, // (B, Sk, D) fp16 index keys (side-plane rows or fresh K)
    qpos: u64, // (B, Sq) i64
    kpos: u64, // (B, Sk) i64, -1 = padded
    out: u64, // (B, H, Sq, NB) fp32
    sq: usize,
    nb: usize,
    sk: usize,
    heads: usize, // 4
    batches: usize,
    local: u32, // index_local_blocks (1)
    init: u32, // sparse_init_block (0)
) !void {
    var a: cuda.Args = .{};
    inline for (.{ q, k, qpos, kpos, out, @as(i64, @intCast(sq)), @as(i64, @intCast(nb)), @as(i64, @intCast(sk)), @as(i64, @intCast(heads)), local, init }) |v| a.add(v);
    try cuda.launch.launch(ks.scores, .{ .grid = .{ .x = @intCast(sq), .y = @intCast(nb), .z = @intCast(batches) }, .block = .{ .x = 128 } }, stream, &a);
}

/// m3_topk16: k picks per (batch, head, row), descending, ties at the smaller id, -1 right-pad.
pub fn topk16(ks: Kernels, stream: cuda.Stream, s: u64, out: u64, nb: usize, rows: usize) !void { // m3_topk16: k picks per (batch, head, row)
    if (nb > max_nb) return error.ScoreColumns; // the mark bitmap's bound: chunk the prefill there
    var a: cuda.Args = .{};
    inline for (.{ s, out, @as(i64, @intCast(nb)), @as(i64, @intCast(rows)), @as(c_int, topk) }) |v| a.add(v);
    try cuda.launch.launch(ks.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// m3_sparse_decode_attn's split phase: one program per (batch, q head, split) into the fp32 scratch.
pub fn decode(
    ks: Kernels,
    stream: cuda.Stream,
    q: u64, // (B, Hq, D) fp16
    kt: u64, // (P, PAGE, Hkv, D) fp16
    vt: u64, // (P, PAGE, Hkv, D)
    table: u64, // (B, npp) i32 logical page -> physical
    sel: u64, // (B, Hkv, topk) i32
    seqlens: u64, // (B,) i64
    part: u64, // (B, Hq, nsplit, scratch_elts) fp32
    q_heads: usize,
    kv_heads: usize,
    npp: usize,
    scale: f32,
    batches: usize,
    nsplit: u32,
) !void {
    var a: cuda.Args = .{};
    // 13 kernel parameters: the seven pointers, Hq/Hkv/npp (i64), the fp32 scale, the fixed
    // selection width k = topk (16; the sel lists are (B, Hkv, k)), and the split count.
    inline for (.{ q, kt, vt, table, sel, seqlens, part, @as(i64, @intCast(q_heads)), @as(i64, @intCast(kv_heads)), @as(i64, @intCast(npp)), scale, @as(c_int, topk), nsplit }) |v| a.add(v);
    try cuda.launch.launch(ks.decode, .{ .grid = .{ .x = @intCast(batches), .y = @intCast(q_heads), .z = nsplit }, .block = .{ .x = d } }, stream, &a);
}

/// _m3_decode_combine_kernel: the splits' fp32 (acc, m, l) merge into the fp16 row; zero-valid rows
/// come out all-zero. nsplit stays <= 8 (the combine's shared arrays; 4 is the served envelope).
pub fn combine(ks: Kernels, stream: cuda.Stream, part: u64, out: u64, q_heads: usize, batches: usize, nsplit: u32) !void {
    if (nsplit > 8) return error.TooManySplits;
    var a: cuda.Args = .{};
    inline for (.{ part, out, @as(i64, @intCast(q_heads)), nsplit }) |v| a.add(v);
    try cuda.launch.launch(ks.combine, .{ .grid = .{ .x = @intCast(batches), .y = @intCast(q_heads) }, .block = .{ .x = d } }, stream, &a);
}

/// A decode call's scratch bytes: (B, Hq, nsplit, D + 2) fp32.
pub fn scratchBytes(q_heads: usize, batches: usize, nsplit: usize) usize {
    return q_heads * batches * nsplit * scratch_elts * 4;
}

test "the contract numbers are the fork's" {
    try std.testing.expectEqual(@as(usize, 128), block);
    try std.testing.expectEqual(@as(usize, 128), d);
    try std.testing.expectEqual(@as(usize, 16), topk);
    try std.testing.expectEqual(@as(usize, 2) * d, page); // 256 = the fp16 page; a 128-key run never straddles it
    try std.testing.expectEqual(@as(usize, d + 2), scratch_elts);
}

test "NB is the ceil of the content's key blocks" {
    try std.testing.expectEqual(@as(usize, 1), nbOf(0));
    try std.testing.expectEqual(@as(usize, 1), nbOf(128));
    try std.testing.expectEqual(@as(usize, 2), nbOf(129));
    try std.testing.expectEqual(@as(usize, 1024), nbOf(128 * 1024));
}

test "split bounds partition the fixed width without gaps or overlap" {
    for ([_]usize{ 1, 3, 4, 8 }) |ns| for (0..ns) |s| {
        const b = splitBounds(topk, ns, s);
        try std.testing.expect(b[0] < topk or (b[0] == topk and b[1] == topk)); // never past k
        try std.testing.expect(b[1] <= topk);
        try std.testing.expect(b[1] - b[0] <= topk / ns + 1);
        if (s + 1 < ns) { // adjacent splits touch: per is a constant stride
            const nxt = splitBounds(topk, ns, s + 1);
            try std.testing.expectEqual(b[1], nxt[0]);
        }
    };
    try std.testing.expectEqual([2]usize{ 0, 4 }, splitBounds(topk, 4, 0));
    try std.testing.expectEqual([2]usize{ 12, 16 }, splitBounds(topk, 4, 3));
}

test "the scratch sizing formula" {
    try std.testing.expectEqual(@as(usize, 64 * 8192 * 4 * 130 * 4), scratchBytes(64, 8192, 4));
}

// The NB <= 1024 mark-bitmap bound, enforced at the wrapper before any launch: 1024 passes
// (the wide prefill chunk), 1025 refuses by name. The kernels and stream are never reached.
test "the topk refuses a score table wider than the mark bitmap's 1024 blocks" {
    const ks: Kernels = undefined;
    var s: cuda.Stream = undefined;
    _ = &s;
    try std.testing.expectEqual(@as(usize, 1024), max_nb);
    try std.testing.expectError(error.ScoreColumns, topk16(ks, s, 0, 0, max_nb + 1, 1));
}

// The nsplit <= 8 combine bound (its shared merge arrays): 8 passes the wrapper, 9 refuses
// before any launch. nsplit is the decode's split phase count; the served envelope is 4.
test "the split merge refuses more splits than its shared arrays hold" {
    const ks: Kernels = undefined;
    var s: cuda.Stream = undefined;
    _ = &s;
    try std.testing.expectError(error.TooManySplits, combine(ks, s, 0, 0, 1, 1, 9));
}