//! The prompt kernels in float64: each output at its kernel's bf16 points, the range fp32 allows, gate 4's sums.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const Order = contract.gates.Order;
const sumIn = contract.rows.sumIn;
const ar = contract.attention_range;
const threads = contract.reference_threads;

/// How a norm or SiLU rounds: as the Qwen text model (mode 0) or as H3's refiner blocks (mode 1).
pub const Mode = enum(u32) { qwen = 0, h3 = 1 };

// Metal's error bounds under the launcher's options (safe math, fast functions) and the kernels' `precise::` calls.
pub const div_ulps = 2.5;
pub const rsqrt_ulps = 2;
pub const exp_ulps = 4;

pub fn bf(v: f64) f64 {
    return contract.bf16_round.bf16(v);
}

pub fn val(h: u16) f64 {
    return c.fromBf16(h);
}

/// Every value a step may hold: its fp32 steps widen both ends, its bf16 points round them.
pub const Range = struct {
    lo: f64,
    hi: f64,

    pub fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5).
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn neg(a: Range) Range {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    /// 1 / a, a positive.
    pub fn recip(a: Range) Range {
        return .{ .lo = 1 / a.hi, .hi = 1 / a.lo };
    }
    /// 1 / sqrt(a), a positive.
    pub fn rsqrt(a: Range) Range {
        return .{ .lo = 1 / @sqrt(a.hi), .hi = 1 / @sqrt(a.lo) };
    }
};

/// rsqrt(ss / n + eps) in float64, and its range through fp32 sums `depth` deep (or exact) and the kernel's steps.
pub fn inverse(ss: f64, n: f64, eps: f32, depth: usize, exact: bool) struct { r: f64, range: Range } {
    const e = if (exact) 0 else c.gamma(@floatFromInt(depth), c.unit(.f32)) * ss;
    const mean = (Range{ .lo = (ss - e) / n, .hi = (ss + e) / n }).fp32(div_ulps);
    const eps64: f64 = eps;
    return .{ .r = 1 / @sqrt(ss / n + eps64), .range = mean.add(Range.of(eps64)).fp32(0.5).rsqrt().fp32(rsqrt_ulps) };
}

/// One normed value as hp_normed rounds it: qwen bf16(w bf16(x r)), h3 bf16(x r w).
pub fn normed(x: f64, r: f64, w: f64, mode: Mode) f64 {
    return switch (mode) {
        .qwen => bf(w * bf(x * r)),
        .h3 => bf(x * r * w),
    };
}

pub fn normedRange(x: f64, r: Range, w: f64, mode: Mode) Range {
    const xr = Range.of(x).mul(r).fp32(0.5);
    return switch (mode) {
        .qwen => xr.bf16().mul(Range.of(w)).fp32(0.5).bf16(),
        .h3 => xr.mul(Range.of(w)).fp32(0.5).bf16(),
    };
}

/// A channel turned with its signed partner o: bf16(bf16(n cos) + bf16(o sin)), as hp_heads rounds.
pub fn turned(n: f64, o: f64, cs: f64, sn: f64) f64 {
    return bf(bf(n * cs) + bf(o * sn));
}

pub fn turnedRange(n: Range, o: Range, cs: f64, sn: f64) Range {
    const a = n.mul(Range.of(cs)).fp32(0.5).bf16();
    const b = o.mul(Range.of(sn)).fp32(0.5).bf16();
    return a.add(b).fp32(0.5).bf16();
}

/// 1 + exp(-g) through the kernel's precise exp and add; past fp32's range its exp is +Inf.
fn denominator(g: f64) Range {
    var e = Range.of(@exp(-g)).fp32(exp_ulps);
    if (e.hi > std.math.floatMax(f32)) e.hi = std.math.inf(f64);
    return e.add(Range.of(1)).fp32(0.5);
}

/// hp_glu's output: qwen bf16(bf16(g / (1 + e^-g)) v), h3 bf16(bf16(g bf16(1 / (1 + e^-g))) v).
pub fn glu(g: f64, v: f64, mode: Mode) f64 {
    const e = 1 + @exp(-g);
    const s = switch (mode) {
        .qwen => bf(g / e),
        .h3 => bf(g * bf(1 / e)),
    };
    return bf(s * v);
}

pub fn gluRange(g: f64, v: f64, mode: Mode) Range {
    const e = denominator(g);
    const s = switch (mode) {
        .qwen => Range.of(g).mul(e.recip()).fp32(div_ulps).bf16(),
        .h3 => Range.of(g).mul(e.recip().fp32(div_ulps).bf16()).fp32(0.5).bf16(),
    };
    return s.mul(Range.of(v)).fp32(0.5).bf16();
}

/// Gate 4's simulated sums: an accumulator and an order, or float64 (`acc` none).
pub const Sum = struct {
    acc: c.Accum = .none,
    order: Order = .sequential,

    pub fn of(s: Sum, t: []const f64) f64 {
        if (s.acc != .none) return sumIn(s.acc, s.order, t);
        var x: f64 = 0;
        for (t) |v| x += v;
        return x;
    }
};

/// hp_gemm's outputs in float64 (each sum |x w|, plus |b|, its magnitude), or gate 4's sums in `sim`, on 8 threads.
pub const Gemm = struct {
    x: []const u16,
    w: []const u16,
    b: ?[]const u16,
    M: usize,
    N: usize,
    K: usize,

    const Job = struct {
        g: *const Gemm,
        from: usize,
        to: usize,
        sim: ?Sum,
        out: []f64,
        mag: []f64,
        row: []f64,
        terms: []f64,

        pub fn run(j: *Job) void {
            const g = j.g;
            var have: usize = std.math.maxInt(usize);
            for (j.from..j.to) |i| {
                const m = i / g.N;
                const n = i % g.N;
                if (m != have) {
                    for (j.row, g.x[m * g.K ..][0..g.K]) |*o, h| o.* = val(h);
                    have = m;
                }
                const bias: f64 = if (g.b) |b| val(b[n]) else 0;
                const w = g.w[n * g.K ..][0..g.K];
                if (j.sim) |s| {
                    for (j.terms, j.row, w) |*t, x, h| t.* = x * val(h);
                    j.out[i] = s.of(j.terms) + bias;
                    continue;
                }
                const d = dot(j.row, w);
                j.out[i] = d[0] + bias;
                j.mag[i] = d[1] + @abs(bias);
            }
        }
    };

    /// Each output, and its magnitude unless `sim`; `scratch` holds 8 rows and, for `sim`, 8 rows of terms.
    pub fn run(g: *const Gemm, a: std.mem.Allocator, sim: ?Sum, out: []f64, mag: []f64) !void {
        const total = g.M * g.N;
        const n = threads.count(threads.maximum, total);
        const scratch = try a.alloc(f64, 2 * n * g.K);
        defer a.free(scratch);
        var jobs: [threads.maximum]Job = undefined;
        for (jobs[0..n], 0..) |*j, t| j.* = .{
            .g = g,
            .from = total * t / n,
            .to = total * (t + 1) / n,
            .sim = sim,
            .out = out,
            .mag = mag,
            .row = scratch[2 * t * g.K ..][0..g.K],
            .terms = scratch[(2 * t + 1) * g.K ..][0..g.K],
        };
        threads.run(Job, jobs[0..n], .normal);
    }
};

/// x . w in float64 and sum |x w|, w as bf16 bits.
fn dot(x: []const f64, w: []const u16) [2]f64 {
    const V = @Vector(8, f64);
    var s: V = @splat(0);
    var m: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= w.len) : (i += 8) {
        const bits: @Vector(8, u32) = w[i..][0..8].*;
        const f: @Vector(8, f32) = @bitCast(bits << @splat(16));
        const p = @as(V, x[i..][0..8].*) * @as(V, @floatCast(f));
        s += p;
        m += @abs(p);
    }
    var t = [2]f64{ @reduce(.Add, s), @reduce(.Add, m) };
    while (i < w.len) : (i += 1) {
        const p = x[i] * val(w[i]);
        t[0] += p;
        t[1] += @abs(p);
    }
    return t;
}

/// hp_attention's fp32 steps: lane dots, simd_sum, precise exp, `n` keys in order with `raised` rescales, division.
pub fn form(n: usize, raised: usize) ar.Form {
    const chain: f64 = @floatFromInt(n + raised + 3);
    return .{
        .score = 4 + 6,
        .fast_exp = false,
        .chunk_exp = true,
        .p_rel = 0,
        .num = chain,
        .den = chain,
        .score_round = 1,
        .div = 2 * div_ulps,
        .exp_ulps = .{ exp_ulps * @as(f64, @floatFromInt(1 + raised)), 0 },
    };
}

/// Blocks past the first that may raise the running max; one that does not rescales by exp(0) = 1, changing nothing.
pub fn rises(q: []const f64, k: []const f64, n: usize, scale: f64, s: []f64) usize {
    const f = form(n, 0);
    const u: f64 = 0x1p-24;
    var spread: f64 = 0;
    for (0..n) |j| {
        var dot_: f64 = 0;
        var abs: f64 = 0;
        for (q, k[j * 128 ..][0..128]) |x, y| {
            dot_ += x * y;
            abs += @abs(x * y);
        }
        s[j] = dot_ * scale;
        const t = f.score * u * abs * @abs(scale) + f.score_round * u * @abs(s[j]);
        if (std.math.isFinite(t)) spread = @max(spread, t);
    }
    var top = -std.math.inf(f64);
    var count: usize = 0;
    var j0: usize = 0;
    while (j0 < n) : (j0 += 4) {
        var most = -std.math.inf(f64);
        for (s[j0..@min(j0 + 4, n)]) |v| most = @max(most, v);
        if (j0 > 0 and most + spread >= top - spread) count += 1;
        top = @max(top, most);
    }
    return count;
}
