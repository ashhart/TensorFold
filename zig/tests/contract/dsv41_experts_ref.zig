//! DeepSeek V4.1 routed experts for the contract: inputs, a route plan, the kernels' fp32 order as oracle, float64.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const rows_ = contract.rows;
pub const topk = 6;
pub const limit: f32 = 10.0;
const inv448: f32 = @as(f32, 1.0) / @as(f32, 448.0);
const nan = std.math.nan(f32);
/// Relative error the Metal exp and divide may add to silu, as the oracle's spread.
pub const exp_spread: f64 = 0x1p-20;

pub fn fp4(code: u8) f32 {
    const mags = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return if (code & 8 != 0) -mags[code & 7] else mags[code & 7];
}

pub fn e4m3(code: u8) f32 {
    const e: i32 = (code >> 3) & 15;
    const m: f32 = @floatFromInt(code & 7);
    const v: f32 = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(1 + m / 8, e - 7);
    return if (code & 0x80 != 0) -v else v;
}

pub fn pow2(e: i32) f32 {
    return std.math.ldexp(@as(f32, 1), e);
}

pub fn bf(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

/// The reference's act_quant scale exponent for a block's amax.
fn scaleExp(amax: f32) i32 {
    const b: u32 = @bitCast(@max(amax, 1e-4) * inv448);
    return @as(i32, @intCast((b >> 23) & 0xff)) - 127 + @intFromBool((b & 0x7fffff) != 0);
}

/// A magnitude of at most 448 as its nearest E4M3 code, ties to even.
fn code8(a: f32, neg: bool) u8 {
    var b: u32 = @bitCast(a);
    b +%= 0x7FFFF + ((b >> 20) & 1);
    var cc: u32 = ((b >> 23) -% 120) << 3 | ((b >> 20) & 7);
    if (a < 0.015625) {
        const t = a * 512;
        const f = @floor(t);
        cc = @intFromFloat(if (t - f > 0.5 or (t - f == 0.5 and @mod(f, 2) == 1)) f + 1 else f);
    }
    return @intCast(cc | @as(u32, if (neg) 0x80 else 0));
}

/// One row as act_quant leaves it: unscaled E4M3 values and a power-of-two scale a 32-block (NaN if non-finite).
pub fn quantRow(x: []const u16, v: []f32, s: []f32) void {
    for (0..x.len / 32) |b| {
        var f: [32]f32 = undefined;
        var amax: f32 = 0;
        var bad = false;
        for (&f, 0..) |*y, j| {
            y.* = c.fromBf16(x[32 * b + j]);
            bad = bad or !std.math.isFinite(y.*);
            amax = @max(amax, @abs(y.*));
        }
        const e = scaleExp(amax);
        s[b] = if (bad) nan else pow2(e);
        for (f, 0..) |y, j| v[32 * b + j] = e4m3(code8(@min(@abs(y) * pow2(-e), 448), std.math.signbit(y)));
    }
}

/// A block of 32 bf16-valued activations quantized the same way and read back as bf16.
pub fn quantBlock(v: []const f32, out: []f32) void {
    var amax: f32 = 0;
    var bad = false;
    for (v) |y| {
        bad = bad or !std.math.isFinite(y);
        amax = @max(amax, @abs(y));
    }
    const e = scaleExp(amax);
    for (v, out) |y, *o| o.* = if (bad) nan else bf(e4m3(code8(@min(@abs(y) * pow2(-e), 448), std.math.signbit(y))) * pow2(e));
}

/// silu(clamped gate) * clamped up * route weight, rounded to bf16; `k` scales exp's result (the oracle's spread).
pub fn swiglu(gate: f32, up: f32, wt: f32, k: f64) f32 {
    const g = clip(bf(gate), -std.math.inf(f32), limit);
    const u = clip(bf(up), -limit, limit);
    const e: f32 = @floatCast(@exp(-@as(f64, g)) * k);
    return bf(g / (1 + e) * u * wt);
}

fn clip(x: f32, lo: f32, hi: f32) f32 {
    return if (std.math.isNan(x)) x else std.math.clamp(x, lo, hi);
}

/// `n` rows of `k` MXFP4 codes (two a byte, low nibble first), E8M0 scales a 32 codes, for each of `e` experts.
pub const Mats = struct {
    a: std.mem.Allocator,
    e: usize,
    n: usize,
    k: usize,
    codes: []u8,
    scales: []u8,

    pub fn init(a: std.mem.Allocator, e: usize, n: usize, k: usize) !Mats {
        const codes = try a.alloc(u8, e * n * k / 2);
        errdefer a.free(codes);
        return .{ .a = a, .e = e, .n = n, .k = k, .codes = codes, .scales = try a.alloc(u8, e * n * k / 32) };
    }

    pub fn deinit(m: Mats) void {
        m.a.free(m.codes);
        m.a.free(m.scales);
    }

    pub fn row(m: Mats, ex: usize, j: usize) []const u8 {
        return m.codes[((ex % m.e) * m.n + j) * m.k / 2 ..][0 .. m.k / 2];
    }

    pub fn srow(m: Mats, ex: usize, j: usize) []const u8 {
        return m.scales[((ex % m.e) * m.n + j) * m.k / 32 ..][0 .. m.k / 32];
    }

    /// Random codes; scales 2^-8..2^-2, or with `wide` every byte but 255 (a quarter of the groups at 241..254).
    pub fn fill(m: Mats, rng: std.Random, wide: bool) void {
        rng.bytes(m.codes);
        for (m.scales, 0..) |*s, i| s.* = if (!wide) 119 + rng.uintLessThan(u8, 7) else if (i % 4 == 0) 241 + rng.uintLessThan(u8, 14) else rng.uintLessThan(u8, 255);
    }

    /// Gate 4's weights: integer E2M1 magnitudes, scale 1, each row's second half the first half reversed and negated.
    pub fn stress(m: Mats, rng: std.Random) void {
        const ints = [_]u8{ 0, 2, 4, 5, 6, 7 };
        @memset(m.scales, 127);
        const rb = m.k / 2;
        const half = rb / 2;
        for (0..m.e * m.n) |r| {
            for (0..half) |i| m.codes[r * rb + i] = ints[rng.uintLessThan(usize, ints.len)] | ints[rng.uintLessThan(usize, ints.len)] << 4;
            for (0..half) |i| {
                const b = m.codes[r * rb + half - 1 - i];
                m.codes[r * rb + half + i] = ((b >> 4) | (b << 4)) ^ 0x88;
            }
        }
    }
};

/// The terms of one dot product: the fp32 order of Kimi's chains, the float64 sum and the magnitude of its terms.
pub const Dot = struct { sum: f32, exact: f64, mag: f64 };

/// 8 lane chains over groups c, c+8, ...: 32 fmas from zero, times the activation scale, an fma by the weight scale.
pub fn dot(codes: []const u8, sc: []const u8, xv: []const f32, xs: ?[]const f32, k: usize) Dot {
    var lane: [8]f32 = @splat(0);
    var exact: f64 = 0;
    var mag: f64 = 0;
    for (0..8) |ch| {
        var g = ch;
        while (g < k / 32) : (g += 8) {
            var p: f32 = 0;
            const s: f32 = if (xs) |t| t[g] else 1;
            for (0..32) |j| {
                const i = 32 * g + j;
                const byte = codes[i / 2];
                const w = fp4(if (i % 2 == 0) byte & 15 else byte >> 4);
                p = @mulAdd(f32, xv[i], w, p);
                const t = @as(f64, xv[i]) * w * s * @as(f64, pow2(@as(i32, sc[g]) - 127));
                exact += t;
                mag += @abs(t);
            }
            lane[ch] = @mulAdd(f32, p * s, pow2(@as(i32, sc[g]) - 127), lane[ch]);
        }
    }
    return .{ .sum = ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7])), .exact = exact, .mag = mag };
}

pub fn depth(k: usize) usize {
    return 32 + (k / 32 + 7) / 8 + 3;
}

/// An output's sum must sit inside the fp32 bound of float64 (the oracle is only an oracle if it does).
pub fn sane(d: Dot, k: usize) !void {
    if (!std.math.isFinite(d.exact) or std.math.isNan(d.sum)) return;
    if (@abs(@as(f64, d.sum) - d.exact) > c.gamma(@floatFromInt(depth(k)), 0x1p-24) * d.mag + 0x1p-149) return error.OracleDrift;
}

/// The routes of a call and the plan the kernels take: slots by expert, pairs grouped by expert in ascending order.
pub const Plan = struct {
    a: std.mem.Allocator,
    rows: usize,
    ids: []u32,
    weights: []f32,
    slots: []u32,
    nslots: [1]u32,
    pairs: []u32,
    where: []u32,

    pub fn init(a: std.mem.Allocator, rows: usize, experts: usize, rng: std.Random) !Plan {
        const np = rows * topk;
        const ids = try a.alloc(u32, np);
        errdefer a.free(ids);
        for (0..rows) |r| for (0..topk) |i| {
            ids[r * topk + i] = while (true) {
                const e = rng.uintLessThan(u32, @intCast(experts));
                if (std.mem.indexOfScalar(u32, ids[r * topk ..][0..i], e) == null) break e;
            };
        };
        const weights = try a.alloc(f32, np);
        errdefer a.free(weights);
        for (weights) |*w| w.* = 0.1 + 1.9 * rng.float(f32);
        const slots = try a.alloc(u32, 3 * np);
        errdefer a.free(slots);
        const pairs = try a.alloc(u32, np);
        errdefer a.free(pairs);
        const where = try a.alloc(u32, np);
        var p: Plan = .{ .a = a, .rows = rows, .ids = ids, .weights = weights, .slots = slots, .nslots = .{0}, .pairs = pairs, .where = where };
        p.build(experts);
        return p;
    }

    fn build(p: *Plan, experts: usize) void {
        var count: [1024]u32 = @splat(0);
        for (p.ids) |e| count[e] += 1;
        var start: u32 = 0;
        var cursor: [1024]u32 = @splat(0);
        var ns: u32 = 0;
        for (0..experts) |e| if (count[e] > 0) {
            p.slots[3 * ns] = @intCast(e);
            p.slots[3 * ns + 1] = start;
            p.slots[3 * ns + 2] = count[e];
            cursor[e] = start;
            start += count[e];
            ns += 1;
        };
        p.nslots[0] = ns;
        for (p.ids, 0..) |e, i| {
            p.pairs[cursor[e]] = @intCast(i);
            p.where[i] = cursor[e];
            cursor[e] += 1;
        }
    }

    pub fn deinit(p: Plan) void {
        p.a.free(p.ids);
        p.a.free(p.weights);
        p.a.free(p.slots);
        p.a.free(p.pairs);
        p.a.free(p.where);
    }

    pub fn expertAt(p: Plan, pos: usize) usize {
        return p.ids[p.pairs[pos]];
    }
};

/// The up kernel's activations by pair position (nominal, and how far exp's error can move each), for column blocks.
pub const UpOut = struct { act: []f32, spread: []f64 };

pub fn upAct(a: std.mem.Allocator, x: []const u16, plan: Plan, w1: Mats, w3: Mats, blocks: []const usize, out: UpOut) !void {
    const n = w1.n;
    const k = w1.k;
    const qv = try a.alloc(f32, plan.rows * k);
    defer a.free(qv);
    const qs = try a.alloc(f32, plan.rows * k / 32);
    defer a.free(qs);
    for (0..plan.rows) |r| quantRow(x[r * k ..][0..k], qv[r * k ..][0..k], qs[r * k / 32 ..][0 .. k / 32]);
    const mults = [3]f64{ 1, 1 + exp_spread, 1 - exp_spread };
    for (0..plan.rows * topk) |pos| {
        const p = plan.pairs[pos];
        const r = p / topk;
        const ex = plan.ids[p];
        const xv = qv[r * k ..][0..k];
        const xs = qs[r * k / 32 ..][0 .. k / 32];
        for (blocks) |b| {
            var v: [3][32]f32 = undefined;
            for (0..32) |j| {
                const col = 32 * b + j;
                const g = dot(w1.row(ex, col), w1.srow(ex, col), xv, xs, k);
                const u = dot(w3.row(ex, col), w3.srow(ex, col), xv, xs, k);
                try sane(g, k);
                try sane(u, k);
                for (&v, mults) |*m, mu| m[j] = swiglu(g.sum, u.sum, plan.weights[p], mu);
            }
            var q: [3][32]f32 = undefined;
            for (&q, v) |*o, m| quantBlock(&m, o);
            for (0..32) |j| {
                const at = pos * n + 32 * b + j;
                out.act[at] = q[0][j];
                out.spread[at] = @max(@abs(@as(f64, q[1][j]) - q[0][j]), @abs(@as(f64, q[2][j]) - q[0][j]));
            }
        }
    }
}

/// Down's outputs by pair position for the columns listed: bf16 of the fp32 order, float64 sum, term magnitude.
pub fn downOut(act: []const f32, plan: Plan, w2: Mats, cols: []const usize, out: []f32, exact: []f64, mag: []f64) !void {
    const k = w2.k;
    for (0..plan.rows * topk) |pos| {
        const ex = plan.expertAt(pos);
        for (cols) |j| {
            const d = dot(w2.row(ex, j), w2.srow(ex, j), act[pos * k ..][0..k], null, k);
            try sane(d, k);
            out[pos * w2.n + j] = bf(d.sum);
            exact[pos * w2.n + j] = d.exact;
            mag[pos * w2.n + j] = d.mag;
        }
    }
}

/// Combine in the kernel's order for canonical groups of `per` experts: fmas from zero by group, a tree over 8 groups.
pub fn combineOne(plan: Plan, weights: []const f32, out: []const f32, n: usize, r: usize, j: usize, per: usize) f32 {
    var part: [8]f32 = @splat(0);
    for (0..topk) |i| {
        const p = r * topk + i;
        const g = plan.ids[p] / per;
        part[g] = @mulAdd(f32, weights[p], out[plan.where[p] * n + j], part[g]);
    }
    var w: usize = 8;
    while (w > 1) : (w /= 2) for (0..w / 2) |i| {
        part[i] = part[2 * i] + part[2 * i + 1];
    };
    return part[0];
}

/// Real route-dependent rows of x: finite bf16 in [-2, 2], every 29th zero.
pub fn fillX(x: []u16, rng: std.Random) void {
    for (x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2);
}

/// Planted NaN and infinities at known places of a [rows][k] matrix.
pub fn plantNonfinite(x: []u16, rows: usize, k: usize) void {
    const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
    const put = [_]f32{ nan, std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * k + p[1]] = c.toBf16(v);
    };
}

pub fn rowFinite(x: []const u16, k: usize, r: usize) bool {
    for (x[r * k ..][0..k]) |h| if (!std.math.isFinite(c.fromBf16(h))) return false;
    return true;
}

/// Gate 4's x: a row's second half is its first reversed, the first eight plus one; quantized x are exact E4M3.
pub fn stressX(x: []u16, rows: usize, k: usize, quantized: bool, rng: std.Random) void {
    const h = k / 2;
    for (0..rows) |r| {
        for (0..h) |i| {
            const small = i >= h - 8;
            const v: f32 = if (!quantized) @floatFromInt(1 + rng.uintLessThan(u32, if (small) 254 else 255)) else if (small) @floatFromInt(1 + rng.uintLessThan(u32, 14)) else @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 15))) * pow2(@intCast(rng.uintLessThan(u32, 5)));
            x[r * k + i] = c.toBf16(v);
        }
        for (0..h) |i| x[r * k + h + i] = c.toBf16(c.fromBf16(x[r * k + h - 1 - i]) + @as(f32, if (i < 8) 1 else 0));
    }
}

pub const Fill = rows_.Fill;

/// One output's terms in float64, activation scale and weight scale included: gate 4 replays them in an accumulator.
pub fn terms(codes: []const u8, sc: []const u8, xv: []const f32, xs: ?[]const f32, k: usize, t: []f64) void {
    for (t[0..k], 0..) |*y, i| {
        const byte = codes[i / 2];
        const s: f64 = if (xs) |q| q[i / 32] else 1;
        y.* = @as(f64, xv[i]) * fp4(if (i % 2 == 0) byte & 15 else byte >> 4) * s * pow2(@as(i32, sc[i / 32]) - 127);
    }
}

/// A float64 .npy (version 1) payload: the values after its header.
pub fn npyF64(bytes: []const u8, out: []f64) !void {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const start = 10 + std.mem.readInt(u16, bytes[8..10], .little);
    if (bytes.len != start + 8 * out.len) return error.BadNpy;
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u64, bytes[start + 8 * i ..][0..8], .little));
}

/// The release's mtp.0.ffn.experts.0.w1 rows 0:64, K 0:256 (real bytes) and numpy's float64 matvec against it.
pub const fixture = struct {
    pub const shape: c.Shape = .{ .rows = 1, .n = 64, .k = 256 };
    pub const raw = @embedFile("fixtures/dsv41/fp4_expert_w1_raw.bin");
    pub const scale = @embedFile("fixtures/dsv41/fp4_expert_w1_scale.bin");
    pub const x_npy = @embedFile("fixtures/dsv41/matvec_x_bf16.npy");
    pub const ref_npy = @embedFile("fixtures/dsv41/fp4_expert_w1_matvec_ref.npy");

    /// Every expert of `m` holds the fixture's rows; `x` holds its activation for each of its rows.
    pub fn load(m: Mats, x: []u16) !void {
        var xv: [256]f64 = undefined;
        try npyF64(x_npy, &xv);
        for (x, 0..) |*h, i| h.* = c.toBf16(@floatCast(xv[i % 256]));
        for (0..m.e) |e| {
            @memcpy(m.codes[e * raw.len ..][0..raw.len], raw);
            @memcpy(m.scales[e * scale.len ..][0..scale.len], scale);
        }
    }
};
