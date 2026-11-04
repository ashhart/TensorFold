//! The canonical expert reduction at any degree: 64 leaves, fmaf chains, a balanced tree, gathered block partials.

const std = @import("std");

pub const leaves: usize = 64;
pub const ids_per_leaf: usize = 8;
pub const shared_id: usize = 512;

/// The leaf an expert id belongs to: eight routed ids each, the shared expert with the last leaf.
pub fn leafOf(id: usize) usize {
    return if (id >= shared_id) leaves - 1 else id / ids_per_leaf;
}

/// A complete subtree of the leaf tree: `size` leaves (a power of two) from the multiple of `size` at `start`.
pub const Block = struct { start: usize, size: usize };

/// The contiguous leaf run [lo, hi) of `rank` of `world` ranks.
pub fn run(world: usize, rank: usize) [2]usize {
    return .{ leaves * rank / world, leaves * (rank + 1) / world };
}

/// The maximal complete subtrees covering the leaf run [lo, hi), in leaf order.
pub fn blocks(lo: usize, hi: usize, out: []Block) usize {
    var n: usize = 0;
    var at = lo;
    while (at < hi) {
        var size: usize = 1;
        while (at % (size * 2) == 0 and at + size * 2 <= hi) size *= 2;
        out[n] = .{ .start = at, .size = size };
        n += 1;
        at += size;
    }
    return n;
}

/// The most blocks a run of leaves needs (two per level).
pub const max_blocks: usize = 12;

/// The partials every rank sends at `world` ranks: the largest block count of any rank (1 for a power of two).
pub fn partials(world: usize) usize {
    var most: usize = 1;
    for (0..world) |r| {
        const lr = run(world, r);
        var bl: [max_blocks]Block = undefined;
        most = @max(most, blocks(lr[0], lr[1], &bl));
    }
    return most;
}

/// Sum of the leaf values `v[0..n]` (n a power of two) in the tree's order: siblings added left plus right.
pub fn treeSum(v: []f32) f32 {
    var n = v.len;
    while (n > 1) : (n /= 2) for (0..n / 2) |i| {
        v[i] = v[2 * i] + v[2 * i + 1];
    };
    return v[0];
}

/// The branch from all ranks' block partials in leaf order: equal aligned neighbours merge until 64 leaves are left.
pub fn finish(starts: []const Block, vals: []f32) f32 {
    var blk: [leaves]Block = undefined;
    var n = starts.len;
    @memcpy(blk[0..n], starts);
    while (n > 1) {
        var i: usize = 0;
        var m: usize = 0;
        while (i < n) {
            if (i + 1 < n and blk[i].size == blk[i + 1].size and blk[i].start % (blk[i].size * 2) == 0) {
                vals[m] = vals[i] + vals[i + 1];
                blk[m] = .{ .start = blk[i].start, .size = blk[i].size * 2 };
                i += 2;
            } else {
                vals[m] = vals[i];
                blk[m] = blk[i];
                i += 1;
            }
            m += 1;
        }
        n = m;
    }
    return vals[0];
}

/// Leaf sums of one output element: leaf[j] = fmaf chain of y[k] * w[k] over slots in leaf j, in slot order, from +0.
pub fn leafSums(picks: []const i32, y: []const f32, w: []const f32, out: *[leaves]f32) void {
    @memset(out, 0);
    for (picks, 0..) |p, k| {
        if (p < 0) continue;
        const j = leafOf(@intCast(p));
        out[j] = @mulAdd(f32, y[k], w[k], out[j]);
    }
}

/// The whole tree over the 64 leaf sums on one rank.
pub fn single(leaf: *const [leaves]f32) f32 {
    var v = leaf.*;
    return treeSum(&v);
}

/// The same branch from `world` virtual ranks: each sums its blocks, the partials are gathered, the tree is finished.
pub fn gathered(world: usize, leaf: *const [leaves]f32) f32 {
    var starts: [leaves]Block = undefined;
    var vals: [leaves]f32 = undefined;
    var n: usize = 0;
    for (0..world) |r| {
        const lr = run(world, r);
        var bl: [max_blocks]Block = undefined;
        const nb = blocks(lr[0], lr[1], &bl);
        for (bl[0..nb]) |b| {
            var part: [leaves]f32 = undefined;
            @memcpy(part[0..b.size], leaf[b.start..][0..b.size]);
            starts[n] = b;
            vals[n] = treeSum(part[0..b.size]);
            n += 1;
        }
    }
    return finish(starts[0..n], vals[0..n]);
}

test "exl3: leaves ids map eight to a leaf and the shared expert joins the last" {
    try std.testing.expectEqual(@as(usize, 0), leafOf(0));
    try std.testing.expectEqual(@as(usize, 0), leafOf(7));
    try std.testing.expectEqual(@as(usize, 1), leafOf(8));
    try std.testing.expectEqual(@as(usize, 63), leafOf(511));
    try std.testing.expectEqual(@as(usize, 63), leafOf(512));
}

test "exl3: leaves the runs of any degree cover the 64 leaves once, in maximal complete subtrees" {
    var world: usize = 1;
    while (world <= 8) : (world += 1) {
        var next: usize = 0;
        for (0..world) |r| {
            const lr = run(world, r);
            try std.testing.expectEqual(next, lr[0]);
            next = lr[1];
            var bl: [max_blocks]Block = undefined;
            const nb = blocks(lr[0], lr[1], &bl);
            var at = lr[0];
            for (bl[0..nb]) |b| {
                try std.testing.expectEqual(at, b.start);
                try std.testing.expect(std.math.isPowerOfTwo(b.size) and b.start % b.size == 0);
                at += b.size;
            }
            try std.testing.expectEqual(lr[1], at);
        }
        try std.testing.expectEqual(leaves, next);
    }
}

test "exl3: leaves the gathered tree equals the single-rank tree bit for bit at every degree" {
    var prng = std.Random.DefaultPrng.init(0x1ea5);
    const rng = prng.random();
    for (0..2000) |_| {
        var leaf: [leaves]f32 = undefined;
        for (&leaf) |*v| v.* = if (rng.uintLessThan(u8, 5) == 0) 0 else (rng.float(f32) - 0.5) * @exp(rng.float(f32) * 20 - 10);
        const want = single(&leaf);
        var world: usize = 1;
        while (world <= 8) : (world += 1) {
            const got = gathered(world, &leaf);
            try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
        }
    }
}

test "exl3: leaves a row's leaf sums chain the picks in slot order from +0" {
    const picks = [_]i32{ 3, 7, 300, 5, -1, 512, 9, 20, 21, 22, 23 };
    var y: [11]f32 = undefined;
    var w: [11]f32 = undefined;
    for (&y, &w, 0..) |*a, *b, i| {
        a.* = @as(f32, @floatFromInt(i)) * 0.37 + 0.1;
        b.* = 1.0 / (@as(f32, @floatFromInt(i)) + 1.5);
    }
    var out: [leaves]f32 = undefined;
    leafSums(&picks, &y, &w, &out);
    var l0: f32 = 0;
    for ([_]usize{ 0, 1, 3 }) |k| l0 = @mulAdd(f32, y[k], w[k], l0);
    try std.testing.expectEqual(@as(u32, @bitCast(l0)), @as(u32, @bitCast(out[0])));
    var l63: f32 = 0;
    l63 = @mulAdd(f32, y[5], w[5], l63);
    try std.testing.expectEqual(@as(u32, @bitCast(l63)), @as(u32, @bitCast(out[63])));
    try std.testing.expectEqual(@as(f32, 0), out[4]);
}

test "exl3: leaves a power-of-two degree sends one partial per rank" {
    for ([_]usize{ 1, 2, 4, 8 }) |world| try std.testing.expectEqual(@as(usize, 1), partials(world));
    try std.testing.expect(partials(3) > 1 and partials(3) <= max_blocks);
}
