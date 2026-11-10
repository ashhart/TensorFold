//! The decoder kernels' tests' shared parts: geometry, float64 ranges at bf16 points, fills, threads and the test type.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;
const bfr = contract.bf16_round;

pub const W = 1024;
pub const heads = 16;
pub const D = 64;
pub const F = 4096;
pub const channels = 24;
pub const registers = 4;
/// Tokens after a tile's cells: the registers, then one zero token.
pub const suffix = registers + 1;
/// A tile's most cells a side, and so its most tokens.
pub const most_cells = 16;
pub const most_tokens = most_cells * most_cells + suffix;
/// The norms' epsilon as the kernels hold it (fp32), and the attention scale 1 / sqrt(64).
pub const eps: f64 = @as(f32, 1e-5);
pub const scale: f64 = 0.125;

// Metal's error bounds with the launcher's options (safe math, fast fp32 functions) and the kernels' `precise::` calls.
pub const div_ulps = 2.5;
pub const rsqrt_ulps = 2;
pub const exp_ulps = 4;

/// image_decoder.metal's H3dShape.
pub const Shape = extern struct {
    tiles: u32 = 0,
    h: u32 = 0,
    w: u32 = 0,
    tokens: u32 = 0,
    rows: u32 = 0,
    cols: u32 = 0,
    m: u32 = 0,
    n: u32 = 0,
    k: u32 = 0,
    mode: u32 = 0,
    eps: f32 = 1e-5,
    scale: f32 = 0.125,
};

/// A value rounded to bf16 in one step, to nearest even (NaN and the infinities kept).
pub fn bf(v: f64) f64 {
    return bfr.bf16(v);
}

pub fn value(b: u16) f64 {
    return c.fromBf16(b);
}

/// A tile of `cells` cells as rows and columns, each 1 to 16 (the widest rows first); null when none holds it.
pub fn geometry(cells: usize) ?[2]usize {
    var w: usize = @min(cells, most_cells);
    while (w > 0) : (w -= 1) if (cells % w == 0 and cells / w <= most_cells) return .{ cells / w, w };
    return null;
}

/// Every value a kernel may hold at a step: its fp32 steps widen both ends, its bf16 points round them.
pub const Range = struct {
    lo: f64,
    hi: f64,

    pub fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    pub fn around(v: f64, e: f64) Range {
        return .{ .lo = v - e, .hi = v + e };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5).
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn sub(a: Range, b: Range) Range {
        return .{ .lo = a.lo - b.hi, .hi = a.hi - b.lo };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    pub fn div(a: Range, n: f64) Range {
        return .{ .lo = a.lo / n, .hi = a.hi / n };
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

/// 1 + exp(-x) through precise::exp and an fp32 add; past fp32's range exp's result is +Inf.
fn denominator(x: f64) Range {
    var e = Range.of(@exp(-x)).fp32(exp_ulps);
    if (e.hi > std.math.floatMax(f32)) e.hi = std.math.inf(f64);
    return e.add(Range.of(1)).fp32(0.5);
}

/// x / (1 + exp(-x)) through the kernel's precise exp and fast division.
pub fn silu(x: f64) Range {
    return Range.of(x).mul(denominator(x).recip()).fp32(div_ulps);
}

/// rsqrt(ss / n + eps) through the kernel's division, add and precise rsqrt, for a sum of squares in `ss`.
pub fn invNorm(ss: Range, n: f64) Range {
    return ss.div(n).fp32(div_ulps).add(Range.of(eps)).fp32(0.5).rsqrt().fp32(rsqrt_ulps);
}

/// Each output's float64 value at the kernel's rounding points and its range, kept for the bounds and intervals.
pub const Ends = struct {
    a: std.mem.Allocator,
    lo: []f64 = &.{},
    hi: []f64 = &.{},

    pub fn deinit(e: *Ends) void {
        e.a.free(e.lo);
        e.a.free(e.hi);
        e.lo = &.{};
        e.hi = &.{};
    }

    pub fn size(e: *Ends, n: usize) !void {
        if (e.lo.len == n) return;
        e.deinit();
        e.lo = try e.a.alloc(f64, n);
        e.hi = try e.a.alloc(f64, n);
    }

    pub fn put(e: *Ends, want: []f64, mag: []f64, i: usize, w: f64, r: Range) void {
        want[i] = w;
        mag[i] = if (std.math.isFinite(w)) @abs(w) else 0;
        e.lo[i] = r.lo;
        e.hi[i] = r.hi;
    }

    /// The allowance about the float64 value: its range's farther end (a non-finite value is checked by class).
    pub fn bounds(e: *const Ends, want: []const f64, allow: []f64) void {
        for (allow, want, e.lo, e.hi) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(@abs(w - lo), @abs(hi - w)) else 0;
    }

    pub fn intervals(e: *const Ends, low: []f64, high: []f64) void {
        @memcpy(low, e.lo);
        @memcpy(high, e.hi);
    }
};

/// How a reference takes its sums: float64 with the spread fp32 allows (none when exact), or replayed (gate 4).
pub const Sum = struct {
    acc: ?c.Accum = null,
    order: g.Order = .sequential,
    exact: bool = false,

    pub const Out = struct { v: f64, r: Range };

    /// The terms' sum and its range: float64's widened by gamma(depth) of their magnitude, or the replay's.
    pub fn of(s: Sum, t: []const f64, depth: usize) Out {
        if (s.acc) |acc| {
            const v = contract.rows.sumIn(acc, s.order, t);
            return .{ .v = v, .r = Range.of(v) };
        }
        var v: f64 = 0;
        var m: f64 = 0;
        for (t) |x| {
            v += x;
            m += @abs(x);
        }
        return .{ .v = v, .r = s.spread(v, m, depth) };
    }

    pub fn spread(s: Sum, v: f64, m: f64, depth: usize) Range {
        if (s.exact or s.acc != null or !std.math.isFinite(m)) return Range.of(v);
        return Range.around(v, c.gamma(@floatFromInt(depth), c.unit(.f32)) * m);
    }
};

pub fn uniform(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

pub fn integer(v: i32) u16 {
    return c.toBf16(@floatFromInt(v));
}

pub fn integers(rng: std.Random, out: []u16, lo: i32, hi: i32) void {
    for (out) |*v| v.* = integer(rng.intRangeAtMost(i32, lo, hi));
}

pub const nan16: u16 = 0x7fc0;
pub const inf16: u16 = 0x7f80;
pub const ninf16: u16 = 0xff80;
/// The plants each gate-1 input takes in turn: NaN, +Inf, -Inf, then both infinities in one place.
pub const plants = [_]u16{ nan16, inf16, ninf16, inf16, ninf16 };

/// `f(ctx, from, to)` over [0, n) in slices, each on its own thread (reference_threads' most).
pub fn parallel(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
    const Job = struct {
        ctx: @TypeOf(ctx),
        from: usize,
        to: usize,
        pub fn run(j: *@This()) void {
            f(j.ctx, j.from, j.to);
        }
    };
    var jobs: [contract.reference_threads.maximum]Job = undefined;
    const parts = contract.reference_threads.count(std.Thread.getCpuCount() catch 1, n);
    for (jobs[0..parts], 0..) |*j, i| j.* = .{ .ctx = ctx, .from = n * i / parts, .to = n * (i + 1) / parts };
    contract.reference_threads.run(Job, jobs[0..parts], .normal);
}

/// A decoder kernel's test: rig, pipeline, ranges; `K` gives spec, prepare, launch, count and outputs (or replays).
pub fn Case(comptime K: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: rig_.Rig,
        pipe: mtl.Pipeline,
        name: []const u8,
        k: K,
        shape: c.Shape = undefined,
        fill: g.Fill = .random,
        /// Host copies of the inputs, dropped at each prepare.
        arena: std.heap.ArenaAllocator,
        bufs: [8]rig_.Ref = undefined,
        nbufs: usize = 0,
        ends: Ends,

        pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, name: []const u8, k: K) Self {
            return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .name = name, .k = k, .arena = std.heap.ArenaAllocator.init(a), .ends = .{ .a = a } };
        }

        pub fn deinit(t: *Self) void {
            t.ends.deinit();
            t.rig.deinit();
            t.arena.deinit();
            t.pipe.deinit();
        }

        pub fn kernel(t: *Self, control: ?g.Gate) g.Kernel {
            return .{ .spec = K.spec(t.name, t.k), .ctx = t, .vt = &vt, .control = control };
        }

        /// The hooks, with a release that also drops the host copies and ranges once a kernel's gates are done.
        const vt = blk: {
            var v = g.hooks(Self);
            v.release = release;
            break :blk v;
        };

        fn release(ctx: *anyopaque) void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            t.rig.reset();
            _ = t.arena.reset(.free_all);
            t.ends.deinit();
        }

        pub fn host(t: *Self) std.mem.Allocator {
            return t.arena.allocator();
        }

        pub fn input(t: *Self, v: anytype) !void {
            t.bufs[t.nbufs] = try t.rig.input(std.mem.sliceAsBytes(v));
            t.nbufs += 1;
        }

        pub fn output(t: *Self, bytes: usize) !void {
            t.bufs[t.nbufs] = try t.rig.output(bytes);
            t.nbufs += 1;
        }

        /// Buffer `i` at the launch's index `at`.
        pub fn bind(t: *Self, e: mtl.ComputeEncoder, i: usize, at: usize) void {
            e.setBuffer(t.bufs[i].buf, t.bufs[i].off, at);
        }

        pub fn prepare(t: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            t.rig.reset();
            _ = t.arena.reset(.retain_capacity);
            t.nbufs = 0;
            t.shape = shape;
            t.fill = fill;
            var prng = std.Random.DefaultPrng.init(seed);
            return t.k.prepare(t, prng.random());
        }

        pub fn run(t: *Self) !void {
            const p = t.rig.begin();
            p.e.setPipeline(t.pipe);
            t.k.launch(t, p.e);
            try t.rig.finish(p);
        }

        pub fn count(t: *Self) usize {
            return t.k.count(t.shape);
        }

        pub fn reference(t: *Self, want: []f64, mag: []f64) !void {
            try t.ends.size(want.len);
            try t.k.outputs(t, .{ .exact = t.fill == .stress }, &t.ends, want, mag);
        }

        pub fn bounds(t: *Self, want: []const f64, _: []const f64, allow: []f64) !void {
            t.ends.bounds(want, allow);
        }

        pub fn intervals(t: *Self, low: []f64, high: []f64) !void {
            t.ends.intervals(low, high);
        }

        pub fn simulate(t: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            var scratch: Ends = .{ .a = t.a };
            defer scratch.deinit();
            try scratch.size(out.len);
            const mag = try t.a.alloc(f64, out.len);
            defer t.a.free(mag);
            try t.k.outputs(t, .{ .acc = acc, .order = order }, &scratch, out, mag);
        }
    };
}
