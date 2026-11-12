//! Exact integer products check canonical rounding error, sampled no-worse accuracy and TP independence.
const std = @import("std");
const canon = @import("exl3_canon.zig");

const slots = canon.slots;
/// Every f32 product and sum below is a multiple of 2^-170 and under 2^30 in these tests: i256 holds them exactly.
const lsb: i32 = -170;

const Parts = struct { m: u64, e: i32, neg: bool };

fn parts(x: f32) Parts {
    const b: u32 = @bitCast(x);
    const ex: i32 = @intCast((b >> 23) & 0xff);
    const frac: u64 = b & 0x7fffff;
    std.debug.assert(ex != 0xff);
    const m: u64 = if (ex == 0) frac else frac | 0x800000;
    // drop the mantissa's trailing zeros: the lowest set bit is what has to fit above 2^lsb
    const z: u6 = if (m == 0) 0 else @intCast(@ctz(m));
    return .{ .m = m >> z, .e = (if (ex == 0) 1 else ex) - 150 + z, .neg = b >> 31 != 0 };
}

fn place(m: u64, e: i32, neg: bool) i256 {
    if (m == 0) return 0;
    std.debug.assert(e >= lsb and e - lsb < 200);
    const v: i256 = @as(i256, m) << @intCast(e - lsb);
    return if (neg) -v else v;
}

fn exact(x: f32) i256 {
    const p = parts(x);
    return place(p.m, p.e, p.neg);
}

/// y * w with no rounding (24 bit by 24 bit mantissas).
fn product(y: f32, w: f32) i256 {
    const a = parts(y);
    const b = parts(w);
    return place(a.m * b.m, a.e + b.e, a.neg != b.neg);
}

/// What one row's order produced against the exact sum of its products.
const Miss = struct {
    /// |result - sum|, in 2^-170.
    abs: u256,
    /// the sum of |y w|, in 2^-170.
    mag: u256,
    /// |the exact sum|, in 2^-170.
    sum: u256,
};

fn miss(r: canon.Row, got: f32) Miss {
    var sum: i256 = 0;
    var mag: u256 = 0;
    for (0..slots) |k| {
        const t = product(r.y[k], r.w[k]);
        sum += t;
        mag += @abs(t);
    }
    return .{ .abs = @abs(exact(got) - sum), .mag = mag, .sum = @abs(sum) };
}

/// |err| <= n u / (1 - n u) mag with u = 2^-24, in integers: |err| (2^24 - n) <= n mag.
fn withinGamma(m: Miss, comptime n: u256) bool {
    return (m.abs << 24) - m.abs * n <= m.mag * n;
}

/// Root bound: 2^-24 |sum| + 18 * 2^-53 sum(|y w|) + half a minimum f32 subnormal.
fn withinRoot(m: Miss) bool {
    return m.abs << 53 <= (m.sum << 29) + 18 * m.mag + (@as(u256, 1) << (53 - 150 - lsb));
}

/// The leading 64 bits of v as a float (the rest is under 2^-63 of it; the error figures need no more).
fn approx(v: u256) f64 {
    var sh: u8 = 0;
    while (v >> sh >> 64 != 0) sh += 1;
    return std.math.ldexp(@as(f64, @floatFromInt(@as(u64, @truncate(v >> sh)))), @as(i32, sh));
}

/// The error in units of u times the magnitude of the terms (the number of roundings that mattered).
fn units(m: Miss) f64 {
    if (m.mag == 0) return 0;
    return approx(m.abs) * 0x1p24 / approx(m.mag);
}

// ---- the orders --------------------------------------------------------------------------------------------------

/// The canonical root: the f64 tree rounded once to f32.
fn tree(r: canon.Row) f32 {
    return canon.toF32(canon.subtree(&r.y, &r.w, &r.p, 0, canon.leaf_log));
}

/// The order before the tree: one fma chain over the slots, in slot order, from +0.
fn chain(r: canon.Row) f32 {
    var acc: f32 = 0.0;
    for (0..slots) |k| acc = @mulAdd(f32, r.y[k], r.w[k], acc);
    return acc;
}

/// The slot chain in f64: the leaf of an all-one-leaf row.
fn chain64(r: canon.Row) f64 {
    var acc: f64 = 0.0;
    for (0..slots) |k| acc = @mulAdd(f64, r.y[k], r.w[k], acc);
    return acc;
}

/// A rank-split sum: each rank chains the slots whose leaf it owns; partials add in rank order (moves with the degree).
fn rankSplit(r: canon.Row, world: usize) f32 {
    var acc: f32 = 0.0;
    for (0..world) |rank| {
        const own = canon.run(world, rank);
        var part: f32 = 0.0;
        for (0..slots) |k| {
            const l = canon.leafOf(r.p[k]);
            if (l >= own.lo and l < own.hi) part = @mulAdd(f32, r.y[k], r.w[k], part);
        }
        acc = if (rank == 0) part else acc + part;
    }
    return acc;
}

// ---- data --------------------------------------------------------------------------------------------------------

const Kind = enum {
    /// Ten distinct routed experts, softmax weights (sum 1), a sigmoid-gated shared expert, outputs of column scales.
    router,
    /// As `router` with output magnitudes over 2^-12 to 2^12: sums that cancel and absorb.
    wide,
    /// Picks on both sides of the leaf 31 / 32 boundary and in the shared leaf (the rank split of two).
    boundary,
    /// All 11 slots in one leaf (a chain of 11, the tree adds only zeros).
    one_leaf,
};

fn row(rng: std.Random, kind: Kind) canon.Row {
    var r: canon.Row = undefined;
    var logits: [slots - 1]f32 = undefined;
    var top: f32 = -std.math.inf(f32);
    for (&logits) |*l| {
        l.* = rng.floatNorm(f32);
        top = @max(top, l.*);
    }
    var norm: f32 = 0.0;
    for (&logits) |*l| {
        l.* = @exp(l.* - top);
        norm += l.*;
    }
    const leaf = rng.uintLessThan(u16, canon.leaves - 1);
    for (0..slots) |k| {
        const scale: f32 = switch (kind) {
            .wide => std.math.exp2(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -12, 12)))),
            else => std.math.exp2(-rng.float(f32) * 4.0) * 0.1,
        };
        r.y[k] = rng.floatNorm(f32) * scale;
        r.w[k] = if (k == slots - 1) 1.0 / (1.0 + @exp(-rng.floatNorm(f32))) else logits[k] / norm;
        r.p[k] = switch (kind) {
            .router, .wide => if (k == slots - 1) 512 else distinct(rng, r.p[0..k]),
            .boundary => if (k == slots - 1) 512 else 248 + rng.uintLessThan(u16, 16),
            .one_leaf => @as(u16, @intCast(leaf * canon.leaf_ids)) + rng.uintLessThan(u16, canon.leaf_ids),
        };
    }
    return r;
}

fn distinct(rng: std.Random, taken: []const u16) u16 {
    while (true) {
        const id = rng.uintLessThan(u16, 512);
        if (std.mem.indexOfScalar(u16, taken, id) == null) return id;
    }
}

const Stat = struct {
    worst: f64 = 0.0,
    sum: f64 = 0.0,
    n: usize = 0,

    fn add(s: *Stat, m: Miss) !void {
        // n = 11 roundings at most on any term, for the tree as for the chain and the rank splits
        try std.testing.expect(withinGamma(m, slots));
        const u = units(m);
        s.worst = @max(s.worst, u);
        s.sum += u;
        s.n += 1;
    }

    fn mean(s: Stat) f64 {
        return s.sum / @as(f64, @floatFromInt(s.n));
    }
};

fn bits(f: f32) u32 {
    return @bitCast(f);
}

fn bits64(f: f64) u64 {
    return @bitCast(f);
}

const kinds = [_]Kind{ .router, .wide, .boundary, .one_leaf };

test "exl3: canon fp64 oracle sees what fp32 loses and nothing where fp32 is exact" {
    var r: canon.Row = .{ .y = @splat(0.0), .w = @splat(1.0), .p = @splat(0) };
    // 2^24 + 1 is not an fp32 number: the chain rounds it to 2^24 (a miss of exactly 1)
    r.y[0] = 0x1p24;
    r.y[1] = 1.0;
    const m = miss(r, chain(r));
    try std.testing.expectEqual(@as(u256, 1) << -lsb, m.abs);
    try std.testing.expectEqual((@as(u256, 0x1p24 + 1)) << -lsb, m.mag);
    try std.testing.expect(withinGamma(m, 1));
    try std.testing.expect(!withinGamma(.{ .abs = 2 * m.abs, .mag = m.mag, .sum = m.sum }, 1));
    // integers whose sums stay under 2^24 are exact in every order
    var prng = std.Random.DefaultPrng.init(0xf640);
    const rng = prng.random();
    for (0..2000) |_| {
        var q = row(rng, .router);
        for (0..slots) |k| {
            q.y[k] = @floatFromInt(rng.intRangeAtMost(i32, -4096, 4096));
            q.w[k] = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
        }
        for ([_]f32{ tree(q), chain(q), rankSplit(q, 2), rankSplit(q, 8), canon.toF32(canon.emulate(5, q)) }) |got| {
            try std.testing.expectEqual(@as(u256, 0), miss(q, got).abs);
        }
    }
}

test "exl3: canon fp64 root obeys its rounding bound and loses no sampled row to the slot chain or rank splits" {
    var prng = std.Random.DefaultPrng.init(0xf64c);
    const rng = prng.random();
    for (kinds) |kind| {
        var t: Stat = .{};
        var c: Stat = .{};
        var s2: Stat = .{};
        var s8: Stat = .{};
        for (0..20000) |_| {
            const r = row(rng, kind);
            const root = miss(r, tree(r));
            try t.add(root);
            try std.testing.expect(withinRoot(root));
            for ([_]f32{ chain(r), rankSplit(r, 2), rankSplit(r, 8) }) |other| try std.testing.expect(root.abs <= miss(r, other).abs);
            try c.add(miss(r, chain(r)));
            try s2.add(miss(r, rankSplit(r, 2)));
            try s8.add(miss(r, rankSplit(r, 8)));
            // a leaf's chain is the slot chain: with every slot in one leaf the tree adds only zeros
            if (kind == .one_leaf) try std.testing.expectEqual(bits64(chain64(r)), bits64(canon.subtree(&r.y, &r.w, &r.p, 0, canon.leaf_log)));
        }
        // the worst output of the tree over these rows is at or under each other order's worst
        try std.testing.expect(t.worst <= c.worst and t.worst <= s2.worst and t.worst <= s8.worst);
        // and its mean is within a quarter of the best other order's
        try std.testing.expect(t.mean() <= 1.25 * @min(c.mean(), @min(s2.mean(), s8.mean())));
    }
}

test "exl3: canon fp64 tree bits are the same at TP 1 to 8 and as the halves of leaves 0-31 and 32-63" {
    var prng = std.Random.DefaultPrng.init(0xf64d);
    const rng = prng.random();
    var split_differs: usize = 0;
    for (kinds) |kind| for (0..20000) |_| {
        const r = row(rng, kind);
        const whole = canon.subtree(&r.y, &r.w, &r.p, 0, canon.leaf_log);
        for (1..9) |world| {
            // f64 partials gathered at any degree merge to the whole tree's f64 bits, hence the same f32 root
            try std.testing.expectEqual(bits64(whole), bits64(canon.emulate(world, r)));
            try std.testing.expectEqual(bits(tree(r)), bits(canon.toF32(canon.emulate(world, r))));
        }
        const low = canon.subtree(&r.y, &r.w, &r.p, 0, canon.leaf_log - 1);
        const high = canon.subtree(&r.y, &r.w, &r.p, canon.leaves / 2, canon.leaf_log - 1);
        try std.testing.expectEqual(bits64(whole), bits64(low + high));
        // the rank-split sums are not that: their bits move with the degree, so the check above has teeth
        const moved = bits(rankSplit(r, 2)) != bits(rankSplit(r, 1)) or bits(rankSplit(r, 8)) != bits(rankSplit(r, 3));
        split_differs += @intFromBool(moved);
    };
    try std.testing.expect(split_differs > 0);
}

test "exl3: canon fp64 root is exact where f32 cancellation loses (tree 0, slot chain 1, independent sum 2) at every degree" {
    // Distinct leaves give 1.25e8 + 1 - 1.25e8 + 1: old f32 tree 0, slot chain 1, exact sum 2.
    var r: canon.Row = .{
        .y = .{ 1e9, 8, -1e9, 8, 0, 0, 0, 0, 0, 0, 0 },
        .w = .{ 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0, 0 },
        .p = .{ 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 512 },
    };
    try std.testing.expectEqual(@as(f32, 1), chain(r));
    try std.testing.expectEqual(@as(f32, 2), tree(r));
    try std.testing.expectEqual(@as(u256, 0), miss(r, tree(r)).abs);
    try std.testing.expect(miss(r, tree(r)).abs <= miss(r, chain(r)).abs);
    for (1..9) |world| try std.testing.expectEqual(@as(f64, 2), canon.emulate(world, r));
    // the cancelled pair apart on the two TP halves is the same sum
    r.p[2] = 256;
    for (1..9) |world| try std.testing.expectEqual(@as(f64, 2), canon.emulate(world, r));
}
