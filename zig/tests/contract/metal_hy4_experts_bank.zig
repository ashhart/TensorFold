//! Hy4-flash's routed experts for the contract: a call's routes and jobs, an expert bank, input rows, float64 dots.
const std = @import("std");
const contract = @import("contract");
const h = @import("metal_hy4.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const Range = fl.Range;
const bf = fl.bf;
const threads = contract.reference_threads;

/// MLX's affine group in Hy4-flash's expert tensors.
pub const group = 64;
/// The widest reduction a test takes (the routed widths); each float64 job keeps a row of codes on its stack.
pub const max_k = 1536;
/// Experts a bank holds: a route picks expert 1 in every row and 15 of experts 2 to 23.
pub const bank_experts = 24;

/// Where a call's jobs come from: one row's picks, a router window's active experts, or a prompt chunk's grouping.
pub const Mode = enum { decode, window, prompt };

/// metal_routed's choice: one row reads its picks directly, a window of 16 rows at most its active experts.
pub fn modeOf(rows: usize) Mode {
    return if (rows == 1) .decode else if (rows <= h.hy.metal_routed.max_rows) .window else .prompt;
}

/// A call's routes and the jobs metal_tensor_experts builds from them: offsets, order and the job-to-expert map.
pub const Route = struct {
    a: std.mem.Allocator,
    rows: usize,
    topk: usize,
    /// Each pair's expert (pair p = row * topk + slot).
    pick: []i32,
    /// Each job's first sorted slot (past the active experts: every pair).
    offsets: []i32,
    /// Each sorted slot's pair.
    order: []i32,
    /// Each job's expert.
    ids: []i32,
    /// Host only: each sorted slot's job, and each pair's sorted slot.
    job: []u32,
    slot: []u32,

    /// Row r's slot r % topk picks expert 1, the rest distinct ones of 2 .. among - 1 (0 never: no map is identity).
    pub fn init(a: std.mem.Allocator, rows: usize, topk: usize, among: usize, experts: usize, mode: Mode, seed: u64) !Route {
        const np = rows * topk;
        const count: usize = switch (mode) {
            .decode => topk,
            .window => np,
            .prompt => experts,
        };
        var r: Route = .{ .a = a, .rows = rows, .topk = topk, .pick = &.{}, .offsets = &.{}, .order = &.{}, .ids = &.{}, .job = &.{}, .slot = &.{} };
        errdefer r.deinit();
        r.pick = try a.alloc(i32, np);
        r.offsets = try a.alloc(i32, count);
        r.ids = try a.alloc(i32, count);
        r.order = try a.alloc(i32, np);
        r.job = try a.alloc(u32, np);
        r.slot = try a.alloc(u32, np);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const pool = try a.alloc(i32, among - 2);
        defer a.free(pool);
        for (0..rows) |row| {
            for (pool, 2..) |*p, e| p.* = @intCast(e);
            // the first topk - 1 places of a shuffle of 2 .. among - 1
            for (0..topk - 1) |i| std.mem.swap(i32, &pool[i], &pool[i + rng.uintLessThan(usize, pool.len - i)]);
            var next: usize = 0;
            for (r.pick[row * topk ..][0..topk], 0..) |*p, s| {
                if (s == row % topk) {
                    p.* = 1;
                } else {
                    p.* = pool[next];
                    next += 1;
                }
            }
        }
        switch (mode) {
            .decode => for (0..topk) |j| {
                r.offsets[j] = @intCast(j);
                r.order[j] = @intCast(j);
                r.ids[j] = r.pick[j];
            },
            // the router's active experts ascending, each one's pairs in row order; later jobs empty
            .window, .prompt => {
                var at: usize = 0;
                var job: usize = 0;
                for (0..experts) |e| {
                    const first = at;
                    for (r.pick, 0..) |p, pair| if (p == e) {
                        r.order[at] = @intCast(pair);
                        at += 1;
                    };
                    if (mode == .window and at == first) continue;
                    r.offsets[job] = @intCast(first);
                    r.ids[job] = @intCast(e);
                    job += 1;
                }
                for (r.offsets[job..], r.ids[job..]) |*o, *i| {
                    o.* = @intCast(np);
                    i.* = 0;
                }
            },
        }
        for (0..count) |u| {
            const end: usize = if (u + 1 < count) @intCast(r.offsets[u + 1]) else np;
            for (@as(usize, @intCast(r.offsets[u]))..end) |i| r.job[i] = @intCast(u);
        }
        for (r.order, 0..) |p, i| r.slot[@intCast(p)] = @intCast(i);
        return r;
    }

    pub fn deinit(r: Route) void {
        inline for (.{ "pick", "offsets", "order", "ids", "job", "slot" }) |f| r.a.free(@field(r, f));
    }

    pub fn pairs(r: Route) usize {
        return r.rows * r.topk;
    }

    /// The expert sorted slot i reads.
    pub fn expertAt(r: Route, i: usize) usize {
        return @intCast(r.ids[r.job[i]]);
    }
};

/// An expert table [experts, n, k] in MLX's affine layout: codes, bf16 scales and biases a group, packed words.
pub const Bank = struct {
    a: std.mem.Allocator,
    experts: usize,
    n: usize,
    k: usize,
    bits: u8,
    stress: bool,
    codes: []u8,
    scales: []u16,
    biases: []u16,
    words: []u32,

    /// contract.affine's weights: uniform codes, scales in [0.001, 0.021], biases in [-0.099, 0.051]; or its stress.
    pub fn init(a: std.mem.Allocator, experts: usize, n: usize, k: usize, bits: u8, stress: bool, seed: u64) !Bank {
        const rows = experts * n;
        const codes = try a.alloc(u8, rows * k);
        errdefer a.free(codes);
        const scales = try a.alloc(u16, rows * (k / group));
        errdefer a.free(scales);
        const biases = try a.alloc(u16, scales.len);
        errdefer a.free(biases);
        const words = try a.alloc(u32, rows * k * bits / 32);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const top: u32 = @as(u32, 1) << @intCast(bits);
        if (stress) {
            // contract.affine's stress, its 8 nudged inputs at weights -1 or 0: sums of 1 to 8, where SiLU keeps errors
            const half: u32 = top / 2;
            @memset(scales, c.toBf16(1));
            @memset(biases, c.toBf16(-@as(f32, @floatFromInt(half))));
            for (0..rows) |j| for (0..k / 2) |i| {
                const q = if (i >= 8) half + 1 + rng.uintLessThan(u32, half - 1) else if (i == 0 or rng.boolean()) half - 1 else half;
                codes[j * k + i] = @intCast(q);
                codes[j * k + k / 2 + i] = @intCast(2 * half - q);
            };
        } else {
            for (codes) |*q| q.* = @intCast(rng.uintLessThan(u32, top));
            for (scales) |*s| s.* = c.toBf16(rng.float(f32) * 0.02 + 0.001);
            for (biases) |*b| b.* = c.toBf16((rng.float(f32) - 0.66) * 0.15);
        }
        // MLX's little-endian bit stream: 4 and 8 bits never cross a word
        @memset(words, 0);
        const per = 32 / bits;
        for (codes, 0..) |q, i| words[i / per] |= @as(u32, q) << @intCast((i % per) * bits);
        return .{ .a = a, .experts = experts, .n = n, .k = k, .bits = bits, .stress = stress, .codes = codes, .scales = scales, .biases = biases, .words = words };
    }

    pub fn deinit(b: Bank) void {
        b.a.free(b.codes);
        b.a.free(b.scales);
        b.a.free(b.biases);
        b.a.free(b.words);
    }

    /// Scale and bias of group `grp` of weight row `row` (expert e's output j is row e * n + j).
    pub fn sb(b: *const Bank, row: usize, grp: usize) [2]f64 {
        const at = row * (b.k / group) + grp;
        return .{ c.fromBf16(b.scales[at]), c.fromBf16(b.biases[at]) };
    }

    /// The portable kernels' weight: the code's fma with its scale and bias in fp32, rounded to bf16.
    pub fn rounded(b: *const Bank, row: usize, i: usize) f64 {
        const at = row * (b.k / group) + i / group;
        const w = @mulAdd(f32, @floatFromInt(b.codes[row * b.k + i]), c.fromBf16(b.scales[at]), c.fromBf16(b.biases[at]));
        return c.fromBf16(c.toBf16(w));
    }
};

/// Input rows [rows, k] bf16 as contract.affine fills them, 8-bit calls' scaled by 1/16 so dots stay near unit size.
pub const Rows = struct {
    a: std.mem.Allocator,
    rows: usize,
    k: usize,
    x: []u16,
    /// Each value and its magnitude in float64, and each group's sum of both.
    xf: []f64,
    xa: []f64,
    gs: []f64,
    ga: []f64,

    /// `off`: what the kernel adds to each code before its sums, which bounds gate 4's inputs.
    pub fn init(a: std.mem.Allocator, rows: usize, k: usize, bits: u8, fill: g.Fill, off: f64, seed: u64) !Rows {
        var r: Rows = .{ .a = a, .rows = rows, .k = k, .x = &.{}, .xf = &.{}, .xa = &.{}, .gs = &.{}, .ga = &.{} };
        errdefer r.deinit();
        r.x = try a.alloc(u16, rows * k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const scale: f32 = if (bits == 8) 1.0 / 16.0 else 1;
        for (r.x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else c.toBf16((rng.float(f32) * 4 - 2) * scale);
        switch (fill) {
            .random => {},
            // contract.affine's places: NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in row 3
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                for (at, [_]f32{ fl.nan32, fl.inf32, -fl.inf32, fl.inf32, -fl.inf32 }) |p, v| if (p[0] < rows) {
                    r.x[p[0] * k + p[1]] = c.toBf16(v);
                };
            },
            // integers mirroring the bank's stress codes, every sum of |(q + off) x| under 2^24 (contract.affine's)
            .stress => {
                const top: u64 = (@as(u64, 1) << @intCast(bits)) + @as(u64, @intFromFloat(off));
                const xmax: u64 = @min(255, ((1 << 24) - 1) / (top * k));
                if (xmax == 0) return error.StressPastFp32;
                for (0..rows) |row| for (0..k / 2) |i| {
                    const v = 1 + rng.uintLessThan(u64, xmax);
                    r.x[row * k + i] = c.toBf16(@floatFromInt(v));
                    r.x[row * k + k / 2 + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
                };
            },
        }
        r.xf = try a.alloc(f64, rows * k);
        r.xa = try a.alloc(f64, rows * k);
        r.gs = try a.alloc(f64, rows * (k / group));
        r.ga = try a.alloc(f64, rows * (k / group));
        for (r.x, r.xf, r.xa) |v, *f, *m| {
            f.* = c.fromBf16(v);
            m.* = @abs(f.*);
        }
        for (r.gs, r.ga, 0..) |*s, *m, at| {
            s.* = 0;
            m.* = 0;
            for (r.xf[at * group ..][0..group], r.xa[at * group ..][0..group]) |v, w| {
                s.* += v;
                m.* += w;
            }
        }
        return r;
    }

    pub fn deinit(r: Rows) void {
        inline for (.{ "x", "xf", "xa", "gs", "ga" }) |f| r.a.free(@field(r, f));
    }
};

/// The formula a kernel sums: MLX's s * sum(q x) + b * sum(x) a group (tensor units), or bf16 weights times x.
pub const Form = enum { factored, rounded };

/// A weight row in float64 for every input row that reads it: its codes and, for the rounded form, its weights.
pub const Weights = struct {
    q: [max_k]f64 = undefined,
    w: [max_k]f64 = undefined,
    wa: [max_k]f64 = undefined,

    pub fn load(self: *Weights, b: *const Bank, row: usize, form: Form) void {
        for (self.q[0..b.k], b.codes[row * b.k ..][0..b.k]) |*q, v| q.* = @floatFromInt(v);
        if (form == .rounded) for (0..b.k) |i| {
            self.w[i] = b.rounded(row, i);
            self.wa[i] = @abs(self.w[i]);
        };
    }

    /// Input row xr through weight row `row` in float64, and the magnitude its fp32 sums carry (off: the code offset).
    pub fn dot(self: *const Weights, b: *const Bank, row: usize, x: *const Rows, xr: usize, form: Form, off: f64) [2]f64 {
        const k = b.k;
        const xf = x.xf[xr * k ..][0..k];
        const xa = x.xa[xr * k ..][0..k];
        var y: f64 = 0;
        var mag: f64 = 0;
        if (form == .rounded) {
            for (self.w[0..k], self.wa[0..k], xf, xa) |w, wa, v, va| {
                y += w * v;
                mag += wa * va;
            }
            return .{ y, mag };
        }
        for (0..k / group) |grp| {
            const p = b.sb(row, grp);
            var qx: f64 = 0;
            var qa: f64 = 0;
            for (self.q[grp * group ..][0..group], xf[grp * group ..][0..group], xa[grp * group ..][0..group]) |q, v, va| {
                qx += q * v;
                qa += (q + off) * va;
            }
            y += p[0] * qx + p[1] * x.gs[xr * (k / group) + grp];
            mag += @abs(p[0]) * qa + @abs(p[1] - off * p[0]) * x.ga[xr * (k / group) + grp];
        }
        return .{ y, mag };
    }
};

/// The dot as an accumulator in `order` leaves it on gate 4's inputs: the closest of contract.affine's structures.
pub fn simDot(acc: c.Accum, order: g.Order, b: *const Bank, row: usize, x: *const Rows, xr: usize, off: f64, want: f64) f64 {
    const k = b.k;
    var t: [max_k]f64 = undefined;
    var xs: [max_k]f64 = undefined;
    var gs: [max_k / group]f64 = undefined;
    const sums = contract.rows.sumIn;
    for (0..k) |i| {
        t[i] = (@as(f64, @floatFromInt(b.codes[row * k + i])) + off) * x.xf[xr * k + i];
        xs[i] = x.xf[xr * k + i];
    }
    const p0 = b.sb(row, 0);
    const whole = sums(acc, order, t[0..k]) * p0[0] + sums(acc, order, xs[0..k]) * (p0[1] - off * p0[0]);
    for (gs[0 .. k / group], 0..) |*y, grp| {
        const p = b.sb(row, grp);
        y.* = sums(acc, .sequential, t[grp * group ..][0..group]) * p[0] + sums(acc, .sequential, xs[grp * group ..][0..group]) * (p[1] - off * p[0]);
    }
    const grouped = sums(acc, order, gs[0 .. k / group]);
    for (0..k) |i| {
        const p = b.sb(row, i / group);
        t[i] = (p[0] * @as(f64, @floatFromInt(b.codes[row * k + i])) + p[1]) * x.xf[xr * k + i];
    }
    const direct = sums(acc, order, t[0..k]);
    var best = whole;
    for ([_]f64{ grouped, direct }) |v| if (!std.math.isFinite(best) or @abs(v - want) < @abs(best - want)) {
        best = v;
    };
    return best;
}

/// Each expert's readers: the input row and output row of every pair it serves, in call order.
pub const Uses = struct {
    a: std.mem.Allocator,
    start: []usize,
    xrow: []u32,
    orow: []u32,

    /// `of(i)` gives entry i's expert, input row and output row, for i below `n`.
    pub fn init(a: std.mem.Allocator, experts: usize, n: usize, ctx: anytype, comptime of: fn (@TypeOf(ctx), usize) [3]usize) !Uses {
        const start = try a.alloc(usize, experts + 1);
        errdefer a.free(start);
        const xrow = try a.alloc(u32, n);
        errdefer a.free(xrow);
        const orow = try a.alloc(u32, n);
        @memset(start, 0);
        for (0..n) |i| start[of(ctx, i)[0] + 1] += 1;
        for (1..experts + 1) |e| start[e] += start[e - 1];
        const fill = try a.alloc(usize, experts);
        defer a.free(fill);
        @memcpy(fill, start[0..experts]);
        for (0..n) |i| {
            const u = of(ctx, i);
            xrow[fill[u[0]]] = @intCast(u[1]);
            orow[fill[u[0]]] = @intCast(u[2]);
            fill[u[0]] += 1;
        }
        return .{ .a = a, .start = start, .xrow = xrow, .orow = orow };
    }

    pub fn deinit(u: Uses) void {
        u.a.free(u.start);
        u.a.free(u.xrow);
        u.a.free(u.orow);
    }
};

/// `ctx.item(i)` for i below `items` on the contract's reference threads (each item writes its own outputs).
pub fn sweep(ctx: anytype, items: usize) void {
    const Ctx = @TypeOf(ctx);
    const Job = struct {
        ctx: Ctx,
        first: usize,
        end: usize,
        pub fn run(job: *@This()) void {
            for (job.first..job.end) |i| job.ctx.item(i);
        }
    };
    const n = threads.count(threads.maximum, items);
    var jobs: [threads.maximum]Job = undefined;
    for (jobs[0..n], 0..) |*j, w| j.* = .{ .ctx = ctx, .first = w * items / n, .end = (w + 1) * items / n };
    threads.run(Job, jobs[0..n], .normal);
}

// --- Hy4's SwiGLU at the checkpoint's bf16 points ---------------------------------------------------------------

fn keepMin(x: f64, lim: f64) f64 {
    return if (std.math.isNan(x)) x else @min(x, lim);
}

fn keepClamp(x: f64, lim: f64) f64 {
    return if (std.math.isNan(x)) x else std.math.clamp(x, -lim, lim);
}

/// bf16(bf16(silu(min(bf16(g), L))) * clamp(bf16(u), -L, L)) in float64, a NaN kept as float64 keeps it.
pub fn swiglu(gate: f64, up: f64, lim: f64) f64 {
    const gt = keepMin(bf(gate), lim);
    const a = bf(gt / (1 + @exp(-gt)));
    return bf(a * keepClamp(bf(up), lim));
}

/// SiLU's least value and where it is: SiLU falls to it from 0 below, and rises from it above.
const silu_low_at = -1.2784645427610738;

fn past(v: f64) f64 {
    return if (v > std.math.floatMax(f32)) std.math.inf(f64) else v;
}

/// silu(v) through fp32 steps: precise::exp within 4 ulps (infinite past fp32's largest), 1 + e and the divide rounded.
fn siluAt(v: f64) Range {
    const e = @exp(-v);
    const rel = 0x1p-21 + 0x1p-40;
    const d_lo = past((1 + past(e * (1 - rel))) * (1 - fl.u32_ulp));
    const d_hi = past((1 + past(e * (1 + rel))) * (1 + fl.u32_ulp));
    const q = if (v >= 0) Range{ .lo = v / d_hi, .hi = v / d_lo } else Range{ .lo = v / d_lo, .hi = v / d_hi };
    return q.fp32();
}

/// silu over [lo, hi] through fp32 steps: the ends' ranges, and SiLU's least value when the span holds it.
fn siluSpan(r: Range) Range {
    const a = siluAt(r.lo);
    const b = siluAt(r.hi);
    var out = Range{ .lo = @min(a.lo, b.lo), .hi = @max(a.hi, b.hi) };
    if (r.lo < silu_low_at and r.hi > silu_low_at) {
        const low = silu_low_at / (1 + @exp(-silu_low_at));
        out.lo = @min(out.lo, low * (1 + 0x1p-20) - fl.tiny);
    }
    return out;
}

/// The same SwiGLU through the kernels' fp32 steps, the gate's and up's fp32 sums within `eg` and `eu` of float64's.
pub fn swigluRange(gate: f64, eg: f64, up: f64, eu: f64, lim: f64) Range {
    const gr = Range.point(gate).widen(eg + fl.tiny).bf16();
    const a = siluSpan(.{ .lo = @min(gr.lo, lim), .hi = @min(gr.hi, lim) }).bf16();
    const ur = Range.point(up).widen(eu + fl.tiny).bf16();
    const v = Range{ .lo = std.math.clamp(ur.lo, -lim, lim), .hi = std.math.clamp(ur.hi, -lim, lim) };
    // bf16 times bf16 is exact in fp32, but for a flushed subnormal
    return a.mul(v).widen(fl.tiny).bf16();
}
