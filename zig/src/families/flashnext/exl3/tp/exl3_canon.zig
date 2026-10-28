//! The canonical expert reduction at any TP degree: 64 leaves of 8 ids, an fma chain each, a fixed binary tree.
const std = @import("std");

pub const leaves: usize = 64;
pub const leaf_log: u8 = 6;
pub const leaf_ids: usize = 8;
pub const slots: usize = 11;
/// Most subtrees one contiguous leaf run of the tree can need (two a level).
pub const max_blocks: usize = 12;
/// Most pieces for up to 8 ranks (a run of 8 leaves is at most 4 blocks; 64 covers any split of the 64 leaves).
pub const max_pieces: usize = 64;

/// The leaf of an expert id: routed ids by eights, the shared expert (512) in the last leaf.
pub fn leafOf(id: usize) usize {
    return if (id >= leaves * leaf_ids) leaves - 1 else id / leaf_ids;
}

/// A complete subtree: leaves [start, start + 2^log).
pub const Block = struct { start: u8, log: u8 };

/// A subtree that rank `rank` computes as its `idx`th partial.
pub const Piece = struct { start: u8, log: u8, rank: u8, idx: u8 };

pub const Run = struct { lo: usize, hi: usize };

/// The contiguous leaves rank `rank` of `world` owns.
pub fn run(world: usize, rank: usize) Run {
    return .{ .lo = rank * leaves / world, .hi = (rank + 1) * leaves / world };
}

/// The maximal complete subtrees of the run [lo, hi), left to right.
pub fn blocks(out: *[max_blocks]Block, lo: usize, hi: usize) usize {
    var n: usize = 0;
    var at = lo;
    while (at < hi) {
        var log: u8 = leaf_log;
        while ((at & ((@as(usize, 1) << @intCast(log)) - 1)) != 0 or at + (@as(usize, 1) << @intCast(log)) > hi) log -= 1;
        out[n] = .{ .start = @intCast(at), .log = log };
        n += 1;
        at += @as(usize, 1) << @intCast(log);
    }
    return n;
}

/// The most partials any rank of `world` computes (1 for a power of two): the all-gather's fixed width.
pub fn maxPartials(world: usize) usize {
    var most: usize = 0;
    var b: [max_blocks]Block = undefined;
    for (0..world) |r| {
        const w = run(world, r);
        most = @max(most, blocks(&b, w.lo, w.hi));
    }
    return most;
}

/// Every rank's pieces in leaf order (the order the finish merges them); returns the count.
pub fn pieces(out: *[max_pieces]Piece, world: usize) usize {
    var n: usize = 0;
    var b: [max_blocks]Block = undefined;
    for (0..world) |r| {
        const w = run(world, r);
        const m = blocks(&b, w.lo, w.hi);
        for (b[0..m], 0..) |blk, i| {
            out[n] = .{ .start = blk.start, .log = blk.log, .rank = @intCast(r), .idx = @intCast(i) };
            n += 1;
        }
    }
    return n;
}

/// One column of a row: the value of the subtree [start, start + 2^log) (the reference of fn_exl3_leaf_tree).
pub fn subtree(y: *const [slots]f32, w: *const [slots]f32, picks: *const [slots]u16, start: usize, log: u8) f32 {
    var leaf: [slots]u8 = undefined;
    var val: [slots]f32 = undefined;
    var n: usize = 0;
    const end = start + (@as(usize, 1) << @intCast(log));
    for (0..slots) |k| {
        const l = leafOf(picks[k]);
        if (l < start or l >= end) continue;
        var j: usize = 0;
        while (j < n and leaf[j] != l) j += 1;
        if (j == n) {
            leaf[n] = @intCast(l);
            val[n] = 0.0;
            n += 1;
        }
        val[j] = @mulAdd(f32, y[k], w[k], val[j]);
    }
    // ascending leaf order, then level by level: adjacent siblings add, a lone node is its own parent (x + 0 = x)
    var i: usize = 1;
    while (i < n) : (i += 1) {
        var j = i;
        while (j > 0 and leaf[j - 1] > leaf[j]) : (j -= 1) {
            std.mem.swap(u8, &leaf[j - 1], &leaf[j]);
            std.mem.swap(f32, &val[j - 1], &val[j]);
        }
    }
    var pos: [slots]u8 = undefined;
    for (0..n) |q| pos[q] = leaf[q] - @as(u8, @intCast(start));
    for (0..log) |_| {
        var m: usize = 0;
        var q: usize = 0;
        while (q < n) {
            const parent = pos[q] >> 1;
            if (q + 1 < n and parent == pos[q + 1] >> 1) {
                val[m] = val[q] + val[q + 1];
                q += 2;
            } else {
                val[m] = val[q];
                q += 1;
            }
            pos[m] = parent;
            m += 1;
        }
        n = m;
    }
    return if (n == 0) 0.0 else val[0];
}

/// The finish: pieces in leaf order merged left to right, two equal adjacent blocks with an aligned left start adding.
pub fn finish(ps: []const Piece, values: []const f32) f32 {
    var s_start: [max_pieces]u8 = undefined;
    var s_log: [max_pieces]u8 = undefined;
    var s_val: [max_pieces]f32 = undefined;
    var top: usize = 0;
    for (ps, values) |p, v| {
        s_start[top] = p.start;
        s_log[top] = p.log;
        s_val[top] = v;
        top += 1;
        while (top >= 2 and s_log[top - 1] == s_log[top - 2] and s_start[top - 2] % (@as(usize, 2) << @intCast(s_log[top - 1])) == 0) {
            s_val[top - 2] = s_val[top - 2] + s_val[top - 1];
            s_log[top - 2] += 1;
            top -= 1;
        }
    }
    return s_val[0];
}

// ---- tests -------------------------------------------------------------------------------------------------------

const Row = struct { y: [slots]f32, w: [slots]f32, p: [slots]u16 };

fn randomRow(rng: std.Random) Row {
    var r: Row = undefined;
    for (0..slots) |k| {
        r.y[k] = rng.floatNorm(f32) * 0.05;
        r.w[k] = rng.floatNorm(f32) * 0.2;
        if (rng.uintLessThan(u8, 23) == 0) r.y[k] = -0.0;
        if (rng.uintLessThan(u8, 29) == 0) r.w[k] = -0.0;
        // clustered picks make several slots share a leaf; the last slot is the shared expert
        r.p[k] = if (k == slots - 1) 512 else if (rng.boolean()) rng.uintLessThan(u16, 512) else 256 + rng.uintLessThan(u16, 24);
    }
    return r;
}

/// The definition with no shortcut: all 64 leaf sums (empty = +0), then the full tree level by level.
fn naive(r: Row) f32 {
    var level: [leaves]f32 = @splat(0.0);
    for (0..slots) |k| {
        const l = leafOf(r.p[k]);
        level[l] = @mulAdd(f32, r.y[k], r.w[k], level[l]);
    }
    var n: usize = leaves;
    while (n > 1) : (n /= 2) for (0..n / 2) |i| {
        level[i] = level[2 * i] + level[2 * i + 1];
    };
    return level[0];
}

/// One virtual rank a piece: every piece of `world` computed by its own subtree, then merged.
fn emulate(world: usize, r: Row) f32 {
    var ps: [max_pieces]Piece = undefined;
    const n = pieces(&ps, world);
    var vals: [max_pieces]f32 = undefined;
    for (ps[0..n], vals[0..n]) |p, *v| v.* = subtree(&r.y, &r.w, &r.p, p.start, p.log);
    return finish(ps[0..n], vals[0..n]);
}

fn bits(f: f32) u32 {
    return @bitCast(f);
}

test "exl3: canon pieces cover the leaves in order for every degree to 8" {
    for (1..9) |world| {
        var ps: [max_pieces]Piece = undefined;
        const n = pieces(&ps, world);
        var at: usize = 0;
        for (ps[0..n]) |p| {
            try std.testing.expectEqual(at, p.start);
            try std.testing.expect(p.start % (@as(usize, 1) << @intCast(p.log)) == 0);
            at += @as(usize, 1) << @intCast(p.log);
        }
        try std.testing.expectEqual(leaves, at);
        if (std.math.isPowerOfTwo(world)) {
            try std.testing.expectEqual(world, n);
            try std.testing.expectEqual(@as(usize, 1), maxPartials(world));
        } else try std.testing.expect(maxPartials(world) > 1);
    }
}

test "exl3: canon every degree 1 to 8 is bit-equal to the whole tree and to the unshortcut definition" {
    var prng = std.Random.DefaultPrng.init(0x7ee);
    const rng = prng.random();
    for (0..4000) |_| {
        const row = randomRow(rng);
        const whole = subtree(&row.y, &row.w, &row.p, 0, leaf_log);
        try std.testing.expectEqual(bits(naive(row)), bits(whole));
        for (1..9) |world| try std.testing.expectEqual(bits(whole), bits(emulate(world, row)));
    }
}

test "exl3: canon mutations of the order, a slot, a leaf or a level change bits" {
    var prng = std.Random.DefaultPrng.init(0x7ef);
    const rng = prng.random();
    var rev_slots: usize = 0;
    var dropped: usize = 0;
    var flat: usize = 0;
    var wrong_leaf: usize = 0;
    for (0..4000) |_| {
        const row = randomRow(rng);
        const want = bits(subtree(&row.y, &row.w, &row.p, 0, leaf_log));
        var rev = row;
        for (0..slots) |k| {
            rev.y[k] = row.y[slots - 1 - k];
            rev.w[k] = row.w[slots - 1 - k];
            rev.p[k] = row.p[slots - 1 - k];
        }
        rev_slots += @intFromBool(bits(subtree(&rev.y, &rev.w, &rev.p, 0, leaf_log)) != want);
        var drop = row;
        drop.w[rng.uintLessThan(usize, slots)] = 0.0;
        dropped += @intFromBool(bits(subtree(&drop.y, &drop.w, &drop.p, 0, leaf_log)) != want);
        // one flat chain over the 11 slots is not the tree
        var chain: f32 = 0.0;
        for (0..slots) |k| chain = @mulAdd(f32, row.y[k], row.w[k], chain);
        flat += @intFromBool(bits(chain) != want);
        // a slot filed under the neighbouring leaf
        var moved = row;
        moved.p[0] = (row.p[0] + 8) % 512;
        wrong_leaf += @intFromBool(bits(subtree(&moved.y, &moved.w, &moved.p, 0, leaf_log)) != want);
    }
    try std.testing.expect(rev_slots > 0 and dropped > 0 and flat > 0 and wrong_leaf > 0);
}
