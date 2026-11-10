//! Qwen3.8-27B decode attention's slice merge: its partials, the float64 merge and the exact-sum stress fixture.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

pub const q_heads = 24;
pub const head_dim = 256;
pub const width = q_heads * head_dim;
pub const split = 32;
pub const max_keys = 131073;

/// shape.k is the root's visible key count, including itself; shape.rows is the tree width.
pub const served = [_]c.Shape{
    .{ .rows = 1, .n = width, .k = 33 },
    .{ .rows = 16, .n = width, .k = 33 },
    .{ .rows = 1, .n = width, .k = 34 },
    .{ .rows = 16, .n = width, .k = 34 },
    .{ .rows = 1, .n = width, .k = 8193 },
    .{ .rows = 2, .n = width, .k = 8193 },
    .{ .rows = 4, .n = width, .k = 8193 },
    .{ .rows = 8, .n = width, .k = 8193 },
    .{ .rows = 12, .n = width, .k = 8193 },
    .{ .rows = 16, .n = width, .k = 8193 },
    .{ .rows = 1, .n = width, .k = 32768 },
    .{ .rows = 16, .n = width, .k = 32768 },
    .{ .rows = 1, .n = width, .k = 32769 },
    .{ .rows = 16, .n = width, .k = 32769 },
    .{ .rows = 1, .n = width, .k = 32770 },
    .{ .rows = 16, .n = width, .k = 32770 },
    .{ .rows = 1, .n = width, .k = max_keys },
};

/// One add a slice for the denominator and for each output, then the exp, the product and the divide.
pub fn depth(s: c.Shape) usize {
    return 2 + (s.k - 1 + s.rows + split - 1) / split;
}

pub fn spec() c.Spec {
    return .{
        .name = "xpu attn_merge (16 subgroups a head)",
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .out = .bf16,
        .replaces = "the attn_merge of 8a7838a65",
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = split } },
        .served = &served,
    };
}

pub fn valid(s: c.Shape) bool {
    return s.rows > 0 and s.rows <= 16 and s.n == width and s.k > 0 and s.k <= max_keys;
}

/// What the slices past a row's last hold: a kernel that weighs them cannot match the bytes of one that does not.
const stale_m: f32 = 100;
const stale_l: f32 = 1e6;

/// The partials of one decode step (max, sum and weighted values a slice) and the tree table of its rows.
pub const Fixture = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    pm: []f32,
    pl: []f32,
    po: []f32,
    tree: []i32,
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, s: c.Shape) !Fixture {
        if (!valid(s)) return error.InvalidShape;
        const parts = s.rows * q_heads * ((s.k - 1 + s.rows + split - 1) / split);
        const pm = try a.alloc(f32, parts);
        errdefer a.free(pm);
        const pl = try a.alloc(f32, parts);
        errdefer a.free(pl);
        const po = try a.alloc(f32, parts * head_dim);
        errdefer a.free(po);
        const tree = try a.alloc(i32, s.rows * 17);
        @memset(tree, 0);
        for (0..s.rows) |t| tree[t * 17] = @intCast(std.math.log2_int(usize, t + 1));
        return .{ .a = a, .shape = s, .pm = pm, .pl = pl, .po = po, .tree = tree };
    }

    pub fn deinit(f: Fixture) void {
        f.a.free(f.tree);
        f.a.free(f.po);
        f.a.free(f.pl);
        f.a.free(f.pm);
    }

    pub fn slices(f: Fixture) usize {
        return (f.shape.k - 1 + f.shape.rows + split - 1) / split;
    }

    /// The slices of row t that hold keys: the merge reads these and no others.
    pub fn used(f: Fixture, t: usize) usize {
        return (f.shape.k - 1 + @as(usize, @intCast(f.tree[t * 17] & 255))) / split + 1;
    }

    fn part(f: Fixture, t: usize, h: usize, s: usize) usize {
        return (t * q_heads + h) * f.slices() + s;
    }

    pub fn fill(f: *Fixture, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        f.stress = mode == .stress;
        @memset(f.pm, stale_m);
        @memset(f.pl, stale_l);
        @memset(f.po, stale_l);
        for (0..f.shape.rows) |t| {
            const u = f.used(t);
            for (0..q_heads) |h| for (0..u) |s| {
                const at = f.part(t, h, s);
                if (mode == .stress) {
                    f.pm[at] = 0;
                    f.pl[at] = 1;
                    @memset(f.po[at * head_dim ..][0..head_dim], @floatCast(stressTerm(f.shape.k, u, s)));
                    continue;
                }
                f.pm[at] = @floatCast(rng.float(f64) * 12 - 6);
                f.pl[at] = @floatCast(1 + rng.float(f64) * 39);
                for (f.po[at * head_dim ..][0..head_dim]) |*x| x.* = @floatCast((rng.float(f64) * 2 - 1) * f.pl[at]);
            };
        }
        if (mode != .nonfinite) return;
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        const at = [_]struct { usize, usize }{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 } };
        for (at, 0..) |p, i| if (p[0] < f.shape.rows) {
            const x = f.part(p[0], p[1], 0);
            switch (i) {
                0 => f.pm[x] = nan,
                1 => f.po[x * head_dim + 3] = inf,
                2 => f.po[x * head_dim + 5] = -inf,
                else => f.pl[x] = nan,
            }
        };
    }

    /// Slice s of u, exact in fp32 in any order: the first half adds 256v + e, the second takes back all but 2048.
    fn stressTerm(k: usize, u: usize, s: usize) f64 {
        const half = u / 2;
        const j = if (s < half) s else s - half;
        const h = std.hash.Wyhash.hash(k, std.mem.asBytes(&j));
        const x: f64 = @floatFromInt((1 + h % 32) * 256 + (h >> 8) % 251);
        if (s < half) return x;
        if (s < 2 * half) return @as(f64, if (j < 8) 256 else 0) - x;
        return 0;
    }

    pub fn reference(f: Fixture, want: []f64, mag: []f64) !void {
        if (f.stress) {
            try f.uniform(.none, .sequential, want, mag);
            return;
        }
        for (0..f.shape.rows) |t| for (0..q_heads) |h| {
            const u = f.used(t);
            var top: f64 = -std.math.inf(f64);
            for (0..u) |s| {
                const x: f64 = f.pm[f.part(t, h, s)];
                if (std.math.isNan(x) or std.math.isNan(top)) top = std.math.nan(f64) else top = @max(top, x);
            }
            var den: f64 = 0;
            const at = (t * q_heads + h) * head_dim;
            @memset(want[at..][0..head_dim], 0);
            @memset(mag[at..][0..head_dim], 0);
            for (0..u) |s| {
                const p = f.part(t, h, s);
                const w = @exp(@as(f64, f.pm[p]) - top);
                den += @as(f64, f.pl[p]) * w;
                for (0..head_dim) |d| {
                    const term = @as(f64, f.po[p * head_dim + d]) * w;
                    want[at + d] += term;
                    mag[at + d] += @abs(term);
                }
            }
            for (0..head_dim) |d| {
                want[at + d] /= den;
                mag[at + d] /= @abs(den);
            }
        };
    }

    /// Stress partials have unit weights and one value for every dimension, so the merge is a sum over a count.
    fn uniform(f: Fixture, acc: c.Accum, order: rows.Order, out: []f64, mag: ?[]f64) !void {
        const terms = try f.a.alloc(f64, f.slices());
        defer f.a.free(terms);
        const ones = try f.a.alloc(f64, terms.len);
        defer f.a.free(ones);
        @memset(ones, 1);
        for (0..f.shape.rows) |t| {
            const u = f.used(t);
            var magnitude: f64 = 0;
            for (terms[0..u], 0..) |*v, s| {
                v.* = stressTerm(f.shape.k, u, s);
                magnitude += @abs(v.*);
            }
            const value = rows.sumIn(acc, order, terms[0..u]) / rows.sumIn(acc, order, ones[0..u]);
            @memset(out[t * width ..][0..width], value);
            if (mag) |m| @memset(m[t * width ..][0..width], magnitude / @as(f64, @floatFromInt(u)));
        }
    }

    pub fn simulate(f: Fixture, acc: c.Accum, order: rows.Order, out: []f64) !void {
        if (!f.stress) return error.ExpectedStressFixture;
        try f.uniform(acc, order, out, null);
    }
};

test "XPU merge: only the slices a row holds count, and each row's depth sets how many" {
    var f = try Fixture.init(std.testing.allocator, .{ .rows = 8, .n = width, .k = 33 });
    defer f.deinit();
    try std.testing.expectEqual(@as(usize, 2), f.slices());
    try std.testing.expectEqual(@as(usize, 2), f.used(0));
    try std.testing.expectEqual(@as(usize, 2), f.used(7));
    var g = try Fixture.init(std.testing.allocator, .{ .rows = 4, .n = width, .k = 32 });
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 1), g.used(0));
    try std.testing.expectEqual(@as(usize, 2), g.used(1));
    try std.testing.expectEqual(@as(usize, 2), g.used(3));
}

test "XPU merge: two slices weigh by their maxima, sums and values" {
    var f = try Fixture.init(std.testing.allocator, .{ .rows = 1, .n = width, .k = 33 });
    defer f.deinit();
    f.fill(.random, 1);
    f.pm[0] = 0;
    f.pm[1] = @log(@as(f32, 3));
    f.pl[0] = 1;
    f.pl[1] = 2;
    @memset(f.po[0..head_dim], 4);
    @memset(f.po[head_dim .. 2 * head_dim], 2);
    var want: [width]f64 = undefined;
    var mag: [width]f64 = undefined;
    try f.reference(&want, &mag);
    try std.testing.expectApproxEqAbs((4.0 + 3 * 2) / (1.0 + 3 * 2), want[0], 1e-6);
}

test "XPU merge: nonfinite partials leave finite, NaN and infinite classes" {
    var f = try Fixture.init(std.testing.allocator, .{ .rows = 5, .n = width, .k = 33 });
    defer f.deinit();
    f.fill(.nonfinite, 0x6b63);
    const want = try f.a.alloc(f64, 5 * width);
    defer f.a.free(want);
    const mag = try f.a.alloc(f64, want.len);
    defer f.a.free(mag);
    try f.reference(want, mag);
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[0]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[(1 * q_heads + 1) * head_dim + 3]));
    try std.testing.expectEqual(c.Class.neg_inf, c.classOf(want[(2 * q_heads + 2) * head_dim + 5]));
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[(3 * q_heads + 3) * head_dim]));
    try std.testing.expectEqual(c.Class.finite, c.classOf(want[(1 * q_heads + 1) * head_dim]));
}

test "XPU merge: gate 4 distinguishes both reduced accumulators in every order" {
    for ([_]usize{ 32769, 32770, max_keys }) |n| {
        var f = try Fixture.init(std.testing.allocator, .{ .rows = 1, .n = width, .k = n });
        defer f.deinit();
        f.fill(.stress, std.hash.Wyhash.hash(0x6b63, spec().name));
        var want: [width]f64 = undefined;
        var mag: [width]f64 = undefined;
        var got: [width]f64 = undefined;
        try f.reference(&want, &mag);
        try std.testing.expectEqual(2048.0 / @as(f64, @floatFromInt(f.used(0))), want[0]);
        for (std.enums.values(rows.Order)) |order| {
            try f.simulate(.f32, order, &got);
            try std.testing.expectEqualSlices(f64, &want, &got);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try f.simulate(acc, order, &got);
                try std.testing.expect(!std.math.isFinite(got[0]) or
                    @abs(got[0] - want[0]) >= 100 * c.boundExact(spec(), want[0]));
            }
        }
    }
}
