//! The MSA indexer's selection contracts as a CPU reference (the CUDA kernels' grade): block-max-pool
//! scores over the index heads, the local/init boost override, and the fixed-width top-k selection.
//! Semantics exactly the fork's receipts (INTERFACE-M3-KERNELS.md §1): scoring fp32, keys masked -inf
//! iff k_pos < 0 or k_pos > q_pos (right-open causal on content positions), blocks anchored to key slots.

const std = @import("std");

/// Whole key blocks the score table carries: the slot anchor's ceil(Sk / block).
pub fn nb(sk: usize, block: usize) usize {
    return (sk + block - 1) / block;
}

/// (B, H_idx, Sq, dim) index queries' fp32 block scores: -> [h, sq, NB], amax over each block's keys.
/// `dim` is the indexer head's width (the pack: 128); fp32 accumulation in index order, dot by dot.
pub fn scores(
    gpa: std.mem.Allocator,
    q: []const f32, // [h, sq, dim], row-major
    k: []const f32, // [sk, dim]
    q_pos: []const i64, // [sq] the queries' content positions
    k_pos: []const i64, // [sk]; -1 = padded or post causal (k_pos > q_pos)
    h: usize,
    sq: usize,
    sk: usize,
    dim: usize,
    block: usize,
) ![]f32 {
    const nbv = nb(sk, block);
    const out = try gpa.alloc(f32, h * sq * nbv);
    errdefer gpa.free(out);
    const neg_inf = -std.math.inf(f32);
    for (0..h) |hi| for (0..sq) |ri| {
        const qp = q_pos[ri];
        for (0..nbv) |b| {
            var best = neg_inf;
            const lo = b * block;
            const hi_end = @min(sk, lo + block);
            for (lo..hi_end) |ki| {
                if (k_pos[ki] < 0 or k_pos[ki] > qp) continue; // the mask: -inf, not skipped-from-pooling? no: masked keys sit at -inf
                var dot: f32 = 0;
                for (0..dim) |di| {
                    dot += q[(hi * sq + ri) * dim + di] * k[ki * dim + di];
                }
                if (dot > best) best = dot; // score type "max": the block's amax
            }
            // the last block's tail keys beyond sk count as masked (their mask word stays -inf)
            out[(hi * sq + ri) * nbv + b] = best;
        }
    };
    return out;
}

/// The fixed-width topk over block scores: boosts OVERRIDE pooled scores (HF scatter(+inf) semantics),
/// valid ids in descending-score order, ties at the smaller block id first, -1 right-pad, left-packed.
/// -> [h, sq, width] int32, always full width. The boosts: the row's own block and the `local - 1`
/// before it (max(0, q_block - l)), then the leading [0, init) blocks when init > 0.
pub fn topk(
    gpa: std.mem.Allocator,
    s: []const f32, // [h, sq, NB], scores()'s table
    h: usize,
    sq: usize,
    nbv: usize,
    width: usize,
    block: usize,
    local: usize,
    init: usize,
    q_pos: []const i64,
) ![]i32 {
    const out = try gpa.alloc(i32, h * sq * width);
    errdefer gpa.free(out);
    const neg_inf = -std.math.inf(f32);
    const boosted = try gpa.alloc(f32, nbv);
    defer gpa.free(boosted);
    const used = try gpa.alloc(bool, nbv);
    defer gpa.free(used);
    for (0..h) |hi| for (0..sq) |ri| {
        const row = (hi * sq + ri);
        const qp = q_pos[ri];
        const q_block = @divTrunc(qp, @as(i64, @intCast(block)));
        @memcpy(boosted, s[row * nbv ..][0..nbv]);
        @memset(used, false);
        // the overrides land after the full reduction: +inf beats every pooled score
        for (0..local) |l| {
            const b = q_block - @as(i64, @intCast(l));
            if (b >= 0 and b < nbv) boosted[@intCast(b)] = std.math.inf(f32);
        }
        for (0..init) |bi| boosted[bi] = std.math.inf(f32);
        for (0..width) |slot| {
            var best: usize = nbv;
            var best_score = neg_inf;
            for (0..nbv) |b| {
                if (used[b]) continue;
                if (boosted[b] > best_score) { // ties stay at the smaller id: strict >
                    best_score = boosted[b];
                    best = b;
                }
            }
            if (best == nbv or best_score == neg_inf) { // the row ran out of valid blocks: -1 right-pad
                out[row * width + slot] = -1;
                continue;
            }
            used[best] = true;
            out[row * width + slot] = @intCast(best); // valid ids left-packed, descending score
        }
    };
    return out;
}

// --- the CPU reference's own receipt tests ---

test "block scores pool by amax and mask the causal keys at -inf" {
    const gpa = std.testing.allocator;
    const dim = 2;
    // one head, four query rows = two keys per block... use block 2, sk 4: two blocks of 2 keys
    const q = [_]f32{ 1, 0, 0, 1, 2, 0, 0, 2 }; // rows r0..r3
    const k = [_]f32{ 1, 0, 0, 1, 3, 0, 0, 3 };
    const q_pos = [_]i64{ 0, 1, 2, 3 };
    const k_pos = [_]i64{ 0, 1, 2, 3 };
    const s = try scores(gpa, &q, &k, &q_pos, &k_pos, 1, 4, 4, dim, 2);
    defer gpa.free(s);
    try std.testing.expectEqual(@as(usize, 2), nb(4, 2));
    // block 0 of row r0: dots [1, 0] (key 1 masked: its k_pos 1 > q_pos 0) → amax 1
    try std.testing.expectEqual(@as(f32, 1), s[0]);
    try std.testing.expectEqual(@as(f32, negInf()), s[1]); // block 1 fully masked for r0
    // r1: block 0 dots [1, 1] → 1; key 2 has k_pos 2 > 1 → block 1 masked
    try std.testing.expectEqual(@as(f32, 1), s[1 * 2 + 0]);
    // r2: block 0 [1, 1], block 1 [6, 0] → 6
    try std.testing.expectEqual(@as(f32, 6), s[2 * 2 + 1]);
    // r3: block 1 [6, 6] → 6
    try std.testing.expectEqual(@as(f32, 6), s[3 * 2 + 1]);
}

fn negInf() f32 {
    return -std.math.inf(f32);
}

test "the top-16 selection: boost override, ties at the smaller id, left-packed, -1 right-pad" {
    const gpa = std.testing.allocator;
    const s = [_]f32{ 3, 1, 4, 1, 5, -std.math.inf(f32), -std.math.inf(f32), 2 };
    const q_pos = [_]i64{ 7 }; // row r0 sits in block 3 (7 / 2 = 3): its own block boosted over 4's block 2
    const sel = try topk(gpa, &s, 1, 1, 8, 3, 2, 1, 0, &q_pos);
    defer gpa.free(sel);
    // boosts: block 3 (the row's own) to +inf; then descending: 3 (+inf), 4 (5), 2 (4); id 5,6 are -inf → skipped
    try std.testing.expectEqualSlices(i32, &.{ 3, 4, 2 }, sel[0..3]);
    const ties = try topk(gpa, &[_]f32{ 4, 4, 4, 6, 9, 9, 2, 2 }, 1, 1, 8, 8, 2, 0, 0, &q_pos);
    defer gpa.free(ties);
    // no boosts (local 0), descending with every tie at the smaller id, full width, none padded
    try std.testing.expectEqualSlices(i32, &.{ 4, 5, 3, 0, 1, 2, 6, 7 }, ties[0..8]);
    const short = try topk(gpa, &[_]f32{ 1, 2, -std.math.inf(f32), -std.math.inf(f32) }, 1, 1, 4, 3, 2, 0, 0, &q_pos);
    defer gpa.free(short);
    try std.testing.expectEqualSlices(i32, &.{ 1, 0, -1 }, short[0..3]); // fewer valid blocks: -1 right-pad
}

test "the local boost override wins over a higher pooled score, and multiple boosts tie at the smaller id" {
    const gpa = std.testing.allocator;
    const s = [_]f32{ 100, 1, 1, 1, 50, 50, 50, 50 };
    const q_pos = [_]i64{ 3 }; // the row's own block is 1
    const sel = try topk(gpa, &s, 1, 1, 8, 4, 2, 1, 0, &q_pos);
    defer gpa.free(sel);
    try std.testing.expectEqualSlices(i32, &.{ 1, 0, 4, 5 }, sel[0..4]); // +inf first, then the 100, then the 50s
    const boosted = try topk(gpa, &s, 1, 1, 8, 3, 2, 4, 0, &q_pos); // the row's block and three before it (two in range)
    defer gpa.free(boosted);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 4 }, boosted[0..3]); // boosted ties: smaller ids first, then the 50s
}