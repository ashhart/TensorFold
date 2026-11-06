//! DeepSeek V4.1's small matvecs for the contract: inputs, the kernel's fp32 order as oracle, float64 reference.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const rows = contract.rows;

/// Longest chain of roundings: a lane's FMAs, the butterfly's five adds and the output's own.
pub fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5 + 1;
}

pub const Out = struct { oracle: f32, exact: f64, mag: f64 };

/// Host inputs: x [rows, k] bf16 and w [n, k] as bf16 bits or fp32 (`wide`), nothing else a matvec needs.
pub const Mv = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    wide: bool,
    x: []u16,
    wb: []u16,
    wf: []f32,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, wide: bool) !Mv {
        const x = try a.alloc(u16, shape.rows * shape.k);
        errdefer a.free(x);
        const wb = try a.alloc(u16, if (wide) 0 else shape.n * shape.k);
        errdefer a.free(wb);
        return .{ .a = a, .shape = shape, .wide = wide, .x = x, .wb = wb, .wf = try a.alloc(f32, if (wide) shape.n * shape.k else 0) };
    }

    pub fn deinit(f: Mv) void {
        f.a.free(f.x);
        f.a.free(f.wb);
        f.a.free(f.wf);
    }

    pub fn outputs(f: Mv) usize {
        return f.shape.rows * f.shape.n;
    }

    pub fn wBytes(f: Mv) []const u8 {
        return if (f.wide) std.mem.sliceAsBytes(f.wf) else std.mem.sliceAsBytes(f.wb);
    }

    pub fn w(f: Mv, j: usize, i: usize) f32 {
        const at = j * f.shape.k + i;
        return if (f.wide) f.wf[at] else c.fromBf16(f.wb[at]);
    }

    fn xv(f: Mv, r: usize, i: usize) f32 {
        return c.fromBf16(f.x[r * f.shape.k + i]);
    }

    fn put(f: Mv, at: usize, v: f32) void {
        if (f.wide) f.wf[at] = v else f.wb[at] = c.toBf16(v);
    }

    /// `snap` keeps fp32 weights on bf16 values, so the two weight types hold the same numbers.
    pub fn fill(f: Mv, mode: rows.Fill, seed: u64, snap: bool) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const total = f.shape.n * f.shape.k;
        for (0..total) |i| {
            const v = (rng.float(f32) - 0.5) * 2;
            if (f.wide and snap) f.wf[i] = c.fromBf16(c.toBf16(v)) else f.put(i, v);
        }
        for (f.x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2);
        switch (mode) {
            .random => {},
            .nonfinite => {
                const k = f.shape.k;
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const val = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, val) |p, v| if (p[0] < f.shape.rows) {
                    f.x[p[0] * k + p[1]] = c.toBf16(v);
                };
            },
            .stress => f.stress(rng),
        }
    }

    /// Gate 4's inputs: integer weights 1..8, K's second half mirroring the first negated: exact fp32, tiny result.
    fn stress(f: Mv, rng: std.Random) void {
        const k = f.shape.k;
        const h = k / 2;
        const xmax: u32 = @min(255, ((1 << 24) - 1) / (8 * @as(u32, @intCast(k))));
        for (0..f.shape.n) |j| for (0..h) |i| {
            const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 8));
            f.put(j * k + i, q);
            f.put(j * k + h + i, -q);
        };
        for (0..f.shape.rows) |r| for (0..h) |i| {
            const v = 1 + rng.uintLessThan(u32, xmax);
            f.x[r * k + i] = c.toBf16(@floatFromInt(v));
            f.x[r * k + h + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
        };
        if (k % 2 == 1) unreachable;
    }

    /// Output (r, j): 32 lane chains, then the butterfly; also the float64 sum and its magnitude.
    pub fn one(f: Mv, r: usize, j: usize) Out {
        var lane: [32]f32 = @splat(0);
        var exact: f64 = 0;
        var mag: f64 = 0;
        for (0..f.shape.k) |i| {
            const wv = f.w(j, i);
            const xi = f.xv(r, i);
            lane[i % 32] = @mulAdd(f32, xi, wv, lane[i % 32]);
            exact += @as(f64, xi) * wv;
            mag += @abs(@as(f64, xi) * wv);
        }
        var m: usize = 1;
        while (m < 32) : (m *= 2) {
            var next: [32]f32 = undefined;
            for (&next, 0..) |*v, l| v.* = lane[l] + lane[l ^ m];
            lane = next;
        }
        return .{ .oracle = lane[0], .exact = exact, .mag = mag };
    }

    /// The kernel's order's outputs as `want` (bf16-rounded if asked); it must sit inside the fp32 bound of float64.
    pub fn reference(f: Mv, want: []f64, mag: []f64, bf16: bool) !void {
        const d: f64 = @floatFromInt(depth(f.shape));
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const o = f.one(r, j);
            const at = r * f.shape.n + j;
            if (std.math.isFinite(o.exact) and @abs(@as(f64, o.oracle) - o.exact) > c.gamma(d, 0x1p-24) * o.mag + 0x1p-149) return error.OracleDrift;
            want[at] = if (bf16) c.fromBf16(c.toBf16(o.oracle)) else o.oracle;
            mag[at] = o.mag;
        };
    }

    /// On gate 4's inputs: each output as an accumulator leaves it, the closer of the order and the kernel's lanes.
    pub fn simulate(f: Mv, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const k = f.shape.k;
        const t = try f.a.alloc(f64, k);
        defer f.a.free(t);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const want = f.one(r, j).exact;
            for (0..k) |i| t[i] = @as(f64, f.xv(r, i)) * f.w(j, i);
            const whole = rows.sumIn(acc, order, t);
            const lanes = rows.sumIn(acc, .strided, t);
            out[r * f.shape.n + j] = if (!std.math.isFinite(whole) or @abs(lanes - want) < @abs(whole - want)) lanes else whole;
        };
    }
};
