//! Qwen3.8-27B decode attention: independent float64 softmax over prefix keys and the tree's visible ancestors.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

pub const q_heads = 24;
pub const kv_heads = 4;
pub const head_dim = 256;
pub const width = q_heads * head_dim;
pub const kv_width = kv_heads * head_dim;
pub const scale: f64 = 1.0 / 16.0;
pub const split = 32;

/// shape.k is the root's visible key count, including itself; shape.rows is the tree width.
pub const served = [_]c.Shape{
    .{ .rows = 1, .n = width, .k = 33 },
    .{ .rows = 16, .n = width, .k = 33 },
    .{ .rows = 1, .n = width, .k = 34 },
    .{ .rows = 16, .n = width, .k = 34 },
    .{ .rows = 1, .n = width, .k = 8193 },
    .{ .rows = 16, .n = width, .k = 8193 },
    .{ .rows = 2, .n = width, .k = 8193 },
    .{ .rows = 4, .n = width, .k = 8193 },
    .{ .rows = 8, .n = width, .k = 8193 },
    .{ .rows = 12, .n = width, .k = 8193 },
    .{ .rows = 1, .n = width, .k = 32768 },
    .{ .rows = 1, .n = width, .k = 32769 },
    .{ .rows = 16, .n = width, .k = 32768 },
    .{ .rows = 16, .n = width, .k = 32769 },
    .{ .rows = 1, .n = width, .k = 32770 },
    .{ .rows = 16, .n = width, .k = 32770 },
};

/// Sixteen lane products, four subgroup additions, online slice updates, then the ordered slice merge.
pub fn depth(s: c.Shape) usize {
    return 16 + 4 + 2 * split + 2 * ((s.k + s.rows - 1 + split - 1) / split);
}

pub fn spec() c.Spec {
    return .{
        .name = "xpu attn_decode + attn_merge (bf16 cache, tree)",
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .out = .bf16,
        .replaces = "upstream 1a5644892 attn_decode + attn_merge",
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = split } },
        .served = &served,
    };
}

pub fn valid(s: c.Shape) bool {
    return s.rows > 0 and s.rows <= 16 and s.n == width and s.k > 0 and s.k <= 32770;
}

fn bf(v: f64) u16 {
    return c.toBf16(@floatCast(v));
}

/// BF16 inputs and a branching binary tree, independent of the device's sliced online softmax.
pub const Fixture = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    q: []u16,
    k: []u16,
    v: []u16,
    tree: []i32,
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, s: c.Shape) !Fixture {
        if (!valid(s)) return error.InvalidShape;
        const q = try a.alloc(u16, s.rows * width);
        errdefer a.free(q);
        const k = try a.alloc(u16, (s.k - 1 + s.rows) * kv_width);
        errdefer a.free(k);
        const v = try a.alloc(u16, k.len);
        errdefer a.free(v);
        const tree = try a.alloc(i32, s.rows * 17);
        @memset(tree, 0);
        for (0..s.rows) |t| {
            var at = t;
            var path: [16]i32 = undefined;
            var n: usize = 0;
            while (true) {
                path[n] = @intCast(at);
                n += 1;
                if (at == 0) break;
                at = (at - 1) / 2;
            }
            tree[t * 17] = @intCast(n - 1);
            for (0..n) |d| tree[t * 17 + 1 + d] = path[n - 1 - d];
        }
        return .{ .a = a, .shape = s, .q = q, .k = k, .v = v, .tree = tree };
    }

    pub fn deinit(f: Fixture) void {
        f.a.free(f.tree);
        f.a.free(f.v);
        f.a.free(f.k);
        f.a.free(f.q);
    }

    pub fn keys(f: Fixture, t: usize) usize {
        return f.shape.k + @as(usize, @intCast(f.tree[t * 17] & 255));
    }

    pub fn keyRow(f: Fixture, t: usize, j: usize) usize {
        const p0 = f.shape.k - 1;
        return if (j < p0) j else p0 + @as(usize, @intCast(f.tree[t * 17 + 1 + j - p0]));
    }

    pub fn slices(f: Fixture) usize {
        return (f.shape.k - 1 + f.shape.rows + split - 1) / split;
    }

    pub fn fill(f: *Fixture, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        f.stress = mode == .stress;
        if (mode == .stress) {
            @memset(f.q, 0);
            @memset(f.k, 0);
            @memset(f.v, 0);
            const half = f.shape.k / 2;
            for (0..half) |j| {
                const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                @memset(f.v[j * kv_width ..][0..kv_width], bf(v * 8));
                @memset(f.v[(half + j) * kv_width ..][0..kv_width], bf((@as(f64, if (j < 8) 1 else 0) - v) * 8));
            }
            return;
        }
        for (f.q) |*x| x.* = bf(rng.float(f64) - 0.5);
        for (f.k) |*x| x.* = bf(rng.float(f64) - 0.5);
        for (f.v) |*x| x.* = bf(rng.float(f64) * 2 - 1);
        if (mode == .nonfinite) {
            const values = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (values, 0..) |v, r| if (r < f.shape.rows) {
                f.q[r * width + r * head_dim] = bf(v);
            };
            if (f.shape.rows > 3) {
                f.q[3 * width + 3 * head_dim] = bf(std.math.inf(f64));
                f.q[3 * width + 3 * head_dim + 1] = bf(-std.math.inf(f64));
            }
            f.v[head_dim] = bf(std.math.inf(f64));
            f.v[2 * head_dim] = bf(-std.math.inf(f64));
            f.v[3 * head_dim] = bf(std.math.nan(f64));
        } else {
            // The root's last key makes a lost one-key slice visible after the bf16 store.
            @memset(f.v[(f.shape.k - 1) * kv_width ..][0..kv_width], bf(32));
        }
    }

    fn probabilities(f: Fixture, t: usize, h: usize, p: []f64) void {
        const hk = h / (q_heads / kv_heads);
        var top: f64 = -std.math.inf(f64);
        for (p, 0..) |*s, j| {
            const kr = f.keyRow(t, j);
            var dot: f64 = 0;
            for (0..head_dim) |d| dot += @as(f64, c.fromBf16(f.q[(t * q_heads + h) * head_dim + d])) *
                c.fromBf16(f.k[(kr * kv_heads + hk) * head_dim + d]);
            s.* = dot * scale;
            if (std.math.isNan(s.*) or std.math.isNan(top)) top = std.math.nan(f64) else top = @max(top, s.*);
        }
        var den: f64 = 0;
        for (p) |*s| {
            s.* = @exp(s.* - top);
            den += s.*;
        }
        for (p) |*s| s.* /= den;
    }

    pub fn reference(f: Fixture, want: []f64, mag: []f64) !void {
        if (f.stress) {
            try f.uniform(.none, .sequential, want, mag);
            return;
        }
        const p = try f.a.alloc(f64, f.shape.k + f.shape.rows);
        defer f.a.free(p);
        for (0..f.shape.rows) |t| for (0..q_heads) |h| {
            const n = f.keys(t);
            f.probabilities(t, h, p[0..n]);
            const hk = h / (q_heads / kv_heads);
            const at = (t * q_heads + h) * head_dim;
            @memset(want[at..][0..head_dim], 0);
            @memset(mag[at..][0..head_dim], 0);
            for (p[0..n], 0..) |prob, j| {
                const kr = f.keyRow(t, j);
                for (0..head_dim) |d| {
                    const term = prob * c.fromBf16(f.v[(kr * kv_heads + hk) * head_dim + d]);
                    want[at + d] += term;
                    mag[at + d] += @abs(term);
                }
            }
        };
    }

    /// Uniform stress attention has exact unit probabilities before normalization and identical value columns.
    fn uniform(f: Fixture, acc: c.Accum, order: rows.Order, out: []f64, mag: ?[]f64) !void {
        const terms = try f.a.alloc(f64, f.shape.k + f.shape.rows);
        defer f.a.free(terms);
        const ones = try f.a.alloc(f64, terms.len);
        defer f.a.free(ones);
        @memset(ones, 1);
        for (0..f.shape.rows) |t| {
            const n = f.keys(t);
            var magnitude: f64 = 0;
            for (terms[0..n], 0..) |*v, j| {
                v.* = c.fromBf16(f.v[f.keyRow(t, j) * kv_width]);
                magnitude += @abs(v.*);
            }
            const den = rows.sumIn(acc, order, ones[0..n]);
            const value = rows.sumIn(acc, order, terms[0..n]) / den;
            @memset(out[t * width ..][0..width], value);
            if (mag) |m| @memset(m[t * width ..][0..width], magnitude / @as(f64, @floatFromInt(n)));
        }
    }

    pub fn simulate(f: Fixture, acc: c.Accum, order: rows.Order, out: []f64) !void {
        if (!f.stress) return error.ExpectedStressFixture;
        try f.uniform(acc, order, out, null);
    }
};

test "XPU attention: tree ancestors exclude siblings, and GQA maps six query heads to each KV head" {
    var f = try Fixture.init(std.testing.allocator, .{ .rows = 4, .n = width, .k = 1 });
    defer f.deinit();
    @memset(f.q, 0);
    @memset(f.k, 0);
    for (0..4) |j| for (0..kv_heads) |h| {
        @memset(f.v[(j * kv_heads + h) * head_dim ..][0..head_dim], bf(@floatFromInt(j * 4 + h)));
    };
    const want = try f.a.alloc(f64, 4 * width);
    defer f.a.free(want);
    const mag = try f.a.alloc(f64, want.len);
    defer f.a.free(mag);
    try f.reference(want, mag);
    try std.testing.expectEqual(@as(usize, 1), f.keyRow(3, 1));
    try std.testing.expectEqual(@as(usize, 3), f.keyRow(3, 2));
    for (0..q_heads) |h| {
        const hk: f64 = @floatFromInt(h / 6);
        try std.testing.expectApproxEqAbs(hk, want[h * head_dim], 1e-12);
        try std.testing.expectApproxEqAbs(4 + hk, want[(2 * q_heads + h) * head_dim], 1e-12);
        try std.testing.expectApproxEqAbs(16.0 / 3.0 + hk, want[(3 * q_heads + h) * head_dim], 1e-12);
    }
}

test "XPU attention: scaled dot products give the independent softmax of prefix and root" {
    var f = try Fixture.init(std.testing.allocator, .{ .rows = 1, .n = width, .k = 2 });
    defer f.deinit();
    @memset(f.q, 0);
    @memset(f.k, 0);
    @memset(f.v, 0);
    f.q[0] = bf(16);
    f.k[kv_width] = bf(1);
    @memset(f.v[kv_width..][0..head_dim], bf(5));
    var want: [width]f64 = undefined;
    var mag: [width]f64 = undefined;
    try f.reference(&want, &mag);
    try std.testing.expectApproxEqAbs(5 * @exp(@as(f64, 1)) / (1 + @exp(@as(f64, 1))), want[0], 1e-12);
    try std.testing.expectEqual(@as(f64, 2.5), want[head_dim]);
    try std.testing.expectEqual(@as(f64, 0), want[6 * head_dim]);
}

test "XPU attention: nonfinite inputs leave both finite and nonfinite reference classes" {
    var f = try Fixture.init(std.testing.allocator, .{ .rows = 5, .n = width, .k = 33 });
    defer f.deinit();
    f.fill(.nonfinite, 0x6b63);
    const want = try f.a.alloc(f64, 5 * width);
    defer f.a.free(want);
    const mag = try f.a.alloc(f64, want.len);
    defer f.a.free(mag);
    try f.reference(want, mag);
    var nonfinite: usize = 0;
    for (want) |v| if (!std.math.isFinite(v)) {
        nonfinite += 1;
    };
    try std.testing.expect(nonfinite > 0 and nonfinite < want.len);
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[0]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[6 * head_dim]));
    try std.testing.expectEqual(c.Class.neg_inf, c.classOf(want[12 * head_dim]));
}

test "XPU attention: losing the one-key tail changes every bf16 output in the control's first shape" {
    var f = try Fixture.init(std.testing.allocator, served[0]);
    defer f.deinit();
    f.fill(.random, std.hash.Wyhash.hash(0x6b63, "control: xpu attention merge losing its last slice"));
    var want: [width]f64 = undefined;
    var mag: [width]f64 = undefined;
    var dropped: [width]f64 = undefined;
    try f.reference(&want, &mag);
    f.shape.k -= 1;
    try f.reference(&dropped, &mag);
    for (want, dropped) |w, d| try std.testing.expect(bf(w) != bf(d));
}

test "XPU attention: gate 4 distinguishes both reduced accumulators in every order" {
    for ([_]usize{ 32769, 32770 }) |n| {
        var f = try Fixture.init(std.testing.allocator, .{ .rows = 1, .n = width, .k = n });
        defer f.deinit();
        f.fill(.stress, std.hash.Wyhash.hash(0x6b63, spec().name));
        var want: [width]f64 = undefined;
        var mag: [width]f64 = undefined;
        var got: [width]f64 = undefined;
        try f.reference(&want, &mag);
        try std.testing.expectEqual(64.0 / @as(f64, @floatFromInt(n)), want[0]);
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
