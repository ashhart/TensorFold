//! Control vectors: the operator's parameters, its float64 reference, the bounds the gates check and its launch.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");
const bf16 = @import("../bf16_round.zig");

/// The hidden width this family's residual stream has.
pub const hidden = 5120;

/// The threads a row's dot is spread over, as the kernel launches it.
pub const threads = 256;

/// How a direction is applied: its component leaves the stream, or it is added as a bias.
pub const Mode = enum(u32) { project = 0, add = 1 };

/// The operator's parameters: the mode, the strength, the rows and the hidden width.
pub const Params = struct {
    mode: Mode = .project,
    alpha: f32 = 1.0,
    rows: usize = 1,
    hidden: usize = hidden,
};

/// One call's inputs: the residual rows in bf16 and the direction in fp32, with the float64 reference beside them.
pub const Steer = struct {
    a: std.mem.Allocator,
    p: Params,
    x: []u16,
    dir: []f32,

    /// A call's rows and a direction as wide as they are.
    pub fn init(a: std.mem.Allocator, p: Params) !Steer {
        if (p.hidden == 0 or p.rows == 0) return error.BadDirectionWidth;
        const x = try a.alloc(u16, p.rows * p.hidden);
        errdefer a.free(x);
        return .{ .a = a, .p = p, .x = x, .dir = try a.alloc(f32, p.hidden) };
    }

    /// A call whose direction is the caller's width: a width that does not match the rows is refused.
    pub fn withDirection(a: std.mem.Allocator, p: Params, dir: []const f32) !Steer {
        var s = try init(a, p);
        errdefer s.deinit();
        a.free(s.dir);
        s.dir = try a.dupe(f32, dir);
        return s;
    }

    pub fn deinit(s: Steer) void {
        s.a.free(s.x);
        s.a.free(s.dir);
    }

    /// The outputs a call writes: the rows the hidden width.
    pub fn outputs(p: Params) usize {
        return p.rows * p.hidden;
    }

    /// The inputs a gate asks for: ordinary rows with a unit direction, NaN and infinities, or sums that test fp32.
    pub fn fill(s: *Steer, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var norm: f64 = 0;
        for (s.dir) |*d| {
            d.* = rng.float(f32) * 2 - 1;
            norm += @as(f64, d.*) * @as(f64, d.*);
        }
        const scale: f32 = if (norm > 0) @floatCast(1 / @sqrt(norm)) else 1;
        for (s.dir) |*d| d.* *= scale;
        for (s.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        switch (mode) {
            .random => {},
            .nonfinite => s.plant(),
            .stress => s.stress(rng),
        }
    }

    /// A NaN and both infinities, where float64 and the kernel must agree on the class.
    fn plant(s: *Steer) void {
        const k = s.p.hidden;
        if (k < 4) return;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
        const put = [_]u16{ nan, inf, ninf, inf, ninf };
        for (at, put) |p, v| if (p[0] < s.p.rows) {
            s.x[p[0] * k + p[1]] = v;
        };
    }

    /// Gate 4's rows: integer terms, the second half mirroring the first (8 nudged by one): big partials, small result.
    fn stress(s: *Steer, rng: std.Random) void {
        const k = s.p.hidden;
        const h = k / 2;
        for (0..s.p.rows) |row| {
            for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                const nudge: f32 = if (i < 8) 1 else 0;
                s.x[row * k + i] = c.toBf16(v);
                s.x[row * k + h + i] = c.toBf16(v + nudge);
            }
            if (k % 2 == 1) s.x[row * k + k - 1] = 0;
        }
        for (0..h) |i| {
            const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            s.dir[i] = w;
            s.dir[h + i] = -w;
        }
        if (k % 2 == 1) s.dir[k - 1] = 0;
    }

    /// A row's dot and the magnitude of its terms, in float64.
    fn dot(s: Steer, r: usize) struct { sum: f64, mag: f64 } {
        const k = s.p.hidden;
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..k) |i| {
            const t = c.fromBf16(s.x[r * k + i]) * @as(f64, s.dir[i]);
            sum += t;
            mag += @abs(t);
        }
        return .{ .sum = sum, .mag = mag };
    }

    /// The float64 outputs and, per output, the magnitude the sums' error carries into it.
    pub fn reference(s: Steer, want: []f64, mag: []f64) !void {
        if (s.dir.len != s.p.hidden) return error.BadDirectionWidth;
        const k = s.p.hidden;
        for (0..s.p.rows) |r| {
            const d = s.dot(r);
            for (0..k) |i| {
                const x = c.fromBf16(s.x[r * k + i]);
                const g: f64 = s.dir[i];
                want[r * k + i] = switch (s.p.mode) {
                    .project => x - @as(f64, s.p.alpha) * d.sum * g,
                    .add => x + @as(f64, s.p.alpha) * g,
                };
                mag[r * k + i] = switch (s.p.mode) {
                    .project => if (std.math.isFinite(d.sum)) @abs(@as(f64, s.p.alpha) * g) * d.mag else 0,
                    .add => 0,
                };
            }
        }
    }

    /// The outputs if the dot's sums ran in `acc` in `order`, the rest of the arithmetic exact.
    pub fn simulate(s: Steer, acc: c.Accum, order: rows.Order, out: []f64) !void {
        if (s.dir.len != s.p.hidden) return error.BadDirectionWidth;
        const k = s.p.hidden;
        const t = try s.a.alloc(f64, k);
        defer s.a.free(t);
        for (0..s.p.rows) |r| {
            for (t, 0..) |*o, i| o.* = c.fromBf16(s.x[r * k + i]) * @as(f64, s.dir[i]);
            const sum = rows.sumIn(acc, order, t);
            for (0..k) |i| {
                const x = c.fromBf16(s.x[r * k + i]);
                const g: f64 = s.dir[i];
                out[r * k + i] = switch (s.p.mode) {
                    .project => x - @as(f64, s.p.alpha) * sum * g,
                    .add => x + @as(f64, s.p.alpha) * g,
                };
            }
        }
    }

    /// The kernel's own arithmetic: each thread sums its stride in fp32, the simdgroups tree, then the groups add up.
    pub fn kernelOut(s: Steer, out: []f64) void {
        const k = s.p.hidden;
        var lanes: [32]f32 = undefined;
        var groups: [threads / 32]f32 = undefined;
        for (0..s.p.rows) |r| {
            @memset(&groups, 0);
            for (0..groups.len) |g| {
                for (0..32) |l| {
                    const t = g * 32 + l;
                    var acc: f32 = 0;
                    var i = t;
                    while (i < k) : (i += threads) acc += c.fromBf16(s.x[r * k + i]) * s.dir[i];
                    lanes[l] = acc;
                }
                var stride: usize = 16;
                while (stride > 0) : (stride /= 2) {
                    for (0..stride) |i| lanes[i] = lanes[i] + lanes[i + stride];
                }
                groups[g] = lanes[0];
            }
            var sum: f32 = 0;
            for (groups) |g| sum += g;
            for (0..k) |i| {
                const x = c.fromBf16(s.x[r * k + i]);
                const g = s.dir[i];
                const v: f32 = switch (s.p.mode) {
                    .project => x - (s.p.alpha * sum) * g,
                    .add => x + s.p.alpha * g,
                };
                out[r * k + i] = bf16.round32(v);
            }
        }
    }

    /// The allowance per output: the sums' gamma over the terms, the fp32 operations after them, then the rounding.
    pub fn bounds(s: Steer, want: []const f64, mag: []const f64, allow: []f64) void {
        const u = 0x1p-24;
        const k = s.p.hidden;
        const gamma = c.gamma(@floatFromInt(depthSums(.{ .rows = s.p.rows, .n = k, .k = k })), u);
        for (0..s.p.rows) |r| {
            const d = s.dot(r);
            const scale = @abs(@as(f64, s.p.alpha));
            for (0..k) |i| {
                if (!std.math.isFinite(d.sum)) {
                    allow[r * k + i] = 0; // the nonfinite gate compares classes, not distances
                    continue;
                }
                const w = want[r * k + i];
                const prod = scale * @abs(@as(f64, s.dir[i])) * @abs(d.sum);
                const work = gamma * mag[r * k + i] + u * (scale * @abs(d.sum) + prod + @abs(w));
                allow[r * k + i] = work + 0x1p-8 * (@abs(w) + work) + 0x1p-133;
            }
        }
    }
};

/// The roundings a row's dot passes through: the threads' chains, the simd tree, then the groups' partials.
pub fn depthSums(shape: c.Shape) usize {
    return (shape.k + threads - 1) / threads + 13;
}

/// The add mode has no sums.
pub fn depthNone(_: c.Shape) usize {
    return 0;
}

/// The kernel's registration: the served shapes, the accumulator and the sums' bound.
pub fn spec(p: Params) c.Spec {
    const sums = p.mode == .project;
    return .{
        .name = if (sums) "qwen27 steer project" else "qwen27 steer add",
        .accum = if (sums) .f32 else .none,
        .depth = if (sums) depthSums else depthNone,
        .ops = if (sums) 3 else 2,
        .out = .bf16,
        .in_place = true,
        .tiles = .{
            .{ .size = 8, .step = 8, .min = 8 },
            .{ .size = hidden, .step = hidden, .min = hidden },
            .{ .size = hidden, .step = hidden, .min = hidden },
        },
        .served = &.{
            .{ .rows = 1, .n = hidden, .k = hidden },
            .{ .rows = 8, .n = hidden, .k = hidden },
            .{ .rows = 64, .n = hidden, .k = hidden },
        },
        .sums = .{
            .why = "5,120 fp32 terms a row: the partials stay under 2^24, so fp32 is exact; fp16 overflows, bf16 rounds away.",
        },
    };
}

test "the kernel's own arithmetic lands inside the bound the spec declares" {
    const a = std.testing.allocator;
    for ([_]Mode{ .project, .add }) |mode| {
        var s = try Steer.init(a, .{ .mode = mode, .alpha = 0.75, .rows = 8 });
        defer s.deinit();
        s.fill(.random, 11);
        const n = Steer.outputs(s.p);
        const want = try a.alloc(f64, n);
        defer a.free(want);
        const mag = try a.alloc(f64, n);
        defer a.free(mag);
        const got = try a.alloc(f64, n);
        defer a.free(got);
        const allow = try a.alloc(f64, n);
        defer a.free(allow);
        try s.reference(want, mag);
        s.kernelOut(got);
        s.bounds(want, mag, allow);
        for (got, want, allow) |g, w, l| try std.testing.expect(@abs(g - w) <= l);
    }
}

test "a direction that is not a unit vector: the reference follows the formula" {
    const a = std.testing.allocator;
    var s = try Steer.withDirection(a, .{ .mode = .project, .alpha = 1, .rows = 1, .hidden = 2 }, &.{ 3, 4 });
    defer s.deinit();
    s.x[0] = c.toBf16(1);
    s.x[1] = c.toBf16(0);
    var want: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    try s.reference(&want, &mag);
    try std.testing.expectEqual(@as(f64, -8), want[0]); // 1 - 3 * 3
    try std.testing.expectEqual(@as(f64, -12), want[1]); // 0 - 3 * 4
    var bias = try Steer.withDirection(a, .{ .mode = .add, .alpha = 1, .rows = 1, .hidden = 2 }, &.{ 3, 4 });
    defer bias.deinit();
    bias.x[0] = c.toBf16(1);
    bias.x[1] = c.toBf16(0);
    var added: [2]f64 = undefined;
    var added_mag: [2]f64 = undefined;
    try bias.reference(&added, &added_mag);
    try std.testing.expectEqual(@as(f64, 4), added[0]);
    try std.testing.expectEqual(@as(f64, 4), added[1]);
}

test "a NaN in a row makes the whole row NaN, as float64 does" {
    const a = std.testing.allocator;
    var s = try Steer.init(a, .{ .mode = .project, .alpha = 1, .rows = 5, .hidden = 64 });
    defer s.deinit();
    s.fill(.nonfinite, 3);
    const n = Steer.outputs(s.p);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const got = try a.alloc(f64, n);
    defer a.free(got);
    try s.reference(want, mag);
    s.kernelOut(got);
    for (want[0..64]) |w| try std.testing.expect(std.math.isNan(w));
    for (got[0..64]) |g| try std.testing.expect(std.math.isNan(g));
    for (got, want) |g, w| try std.testing.expectEqual(c.classOf(w), c.classOf(g));
}

test "a zero-length or mismatched direction is refused" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.BadDirectionWidth, Steer.init(a, .{ .rows = 1, .hidden = 0 }));
    var s = try Steer.withDirection(a, .{ .rows = 1, .hidden = 4 }, &.{ 1, 2, 3 });
    defer s.deinit();
    var want: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    try std.testing.expectError(error.BadDirectionWidth, s.reference(&want, &mag));
}

test "the fp32-sum rule: fp32 sums are exact on the stress rows, half precision misses by far" {
    const a = std.testing.allocator;
    var s = try Steer.init(a, .{ .mode = .project, .alpha = 1, .rows = 2, .hidden = 4096 });
    defer s.deinit();
    s.fill(.stress, 5);
    const n = Steer.outputs(s.p);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const got = try a.alloc(f64, n);
    defer a.free(got);
    try s.reference(want, mag);
    for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
        try s.simulate(.f32, order, got);
        try std.testing.expectEqualSlices(f64, want, got);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try s.simulate(acc, order, got);
            var most: f64 = 0; // the worst output's miss in bf16 ulps of its result
            for (got, want) |g, w| {
                const ulp = 0x1p-8 * @abs(w) + 0x1p-133;
                most = @max(most, if (std.math.isFinite(g)) @abs(g - w) / ulp else std.math.inf(f64));
            }
            try std.testing.expect(most >= 100);
        }
    }
}
