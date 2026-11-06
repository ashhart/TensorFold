//! The drafter's attention under the contract: rc_df_prep (head norms, rotary or NoPE, the ring) and rc_df_vec.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ricursive_controls.zig");
const d = @import("metal_ricursive_dflash2.zig");

const c = contract.spec;
const g = contract.gates;
const D = d.D;
const Range = d.Range;
const Mutation = d.Mutation;
const bf = d.bf;
const f32r = d.f32r;
const u = c.unit(.f32);
const Sim = struct { c.Accum, g.Order };

pub fn collect(s: *d.Suite, x: d.Ctx) !void {
    for (std.enums.values(Prep.Variant)) |v| try s.add(x, Prep, v, .none);
    for (std.enums.values(Vec.Variant)) |v| try s.add(x, Vec, v, .none);
}

/// cos over [lo, hi] (sin as cos(x - pi/2)): the ends' values, or +-1 where an extremum falls inside.
fn cosRange(lo: f64, hi: f64) Range {
    if (hi - lo >= 2 * std.math.pi) return .{ .lo = -1, .hi = 1 };
    var r = Range{ .lo = @min(@cos(lo), @cos(hi)), .hi = @max(@cos(lo), @cos(hi)) };
    var k = @ceil(lo / std.math.pi);
    while (k * std.math.pi <= hi) : (k += 1) {
        if (@mod(k, 2) == 0) r.hi = 1 else r.lo = -1;
    }
    return r;
}

/// The last position before the served context's end at 3 keys before a wrap of `ring`.
fn farPos(ring: u32) u32 {
    return D.context / ring * ring - 3;
}

// --- rc_df_prep: q and k heads normed and rotated (or not, on the NoPE layer), k and v into the ring -------------

pub const Prep = struct {
    /// A block on a sliding layer near the ring's wrap or past 261k positions, on the global NoPE layer; and absorbs.
    pub const Variant = enum { sliding_block, sliding_far, global_block, sliding_context, global_context };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    qw: []u16 = &.{},
    kw: []u16 = &.{},
    old: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    fn context(v: Variant) bool {
        return v == .sliding_context or v == .global_context;
    }
    fn global(t: *const Prep) bool {
        return t.v == .global_block or t.v == .global_context;
    }
    fn ring(t: *const Prep) u32 {
        return D.ring(if (t.global()) D.global else D.window);
    }
    fn heads(t: *const Prep) u32 {
        return if (context(t.v)) 0 else D.heads;
    }

    /// Rows: positions; n: the heads launched (q then kv); k: the head dim.
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = switch (v) {
                .sliding_block => "ricursive dflash2 rc_df_prep sliding block",
                .sliding_far => "ricursive dflash2 rc_df_prep sliding block past 261k",
                .global_block => "ricursive dflash2 rc_df_prep NoPE block",
                .sliding_context => "ricursive dflash2 rc_df_prep sliding context",
                .global_context => "ricursive dflash2 rc_df_prep NoPE context",
            },
            .accum = .f32,
            .depth = prepDepth,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            .served = if (context(v)) &.{ .{ .rows = 2048, .n = D.kv, .k = 128 }, .{ .rows = 5, .n = D.kv, .k = 128 } } else &.{ .{ .rows = D.block, .n = D.heads + D.kv, .k = 128 }, .{ .rows = 1, .n = D.heads + D.kv, .k = 128 } },
            .sums = .{ .bf16 = 0, .why = d.norm_why },
        };
    }
    /// Each lane's 4 fmas, then a 5-level simd_sum.
    fn prepDepth(_: c.Shape) usize {
        return 9;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, m: Mutation) !Prep {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, a, "rc_df_prep", m), .v = v, .m = m };
    }
    pub fn deinit(t: *Prep) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Prep) void {
        inline for (.{ "x", "qw", "kw", "old" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Prep) g.Kernel {
        return controls.mark(.{ .spec = spec(t.v), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Prep);

    fn pos0(t: *const Prep) u32 {
        if (t.fill == .stress) return 0; // gate 4 at positions 0-3: the rotation's own spread stays far under a sum's
        return if (t.v == .sliding_far) farPos(t.ring()) else t.ring() - 3;
    }
    fn rope(t: *const Prep) bool {
        return !t.global();
    }
    /// X's row as dflash2.zig lays it: a block's q, k and v heads; a context's k and v projections.
    fn layout(t: *const Prep) struct { stride: usize, q: usize, k: usize, v: usize } {
        const kvw: usize = D.kv * 128;
        const qw: usize = t.heads() * 128;
        return if (t.heads() == 0) .{ .stride = 2 * kvw, .q = 0, .k = 0, .v = kvw } else .{ .stride = qw + 2 * kvw, .q = 0, .k = qw, .v = qw + kvw };
    }
    fn sizes(t: *const Prep) [3]usize {
        const ringed: usize = @as(usize, t.ring()) * D.kv * 128;
        return .{ @max(1, t.shape.rows * t.heads() * 128), ringed, ringed };
    }

    pub fn prepare(t: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != t.heads() + D.kv or shape.k != 128) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = t.layout();
        t.x = try t.a.alloc(u16, shape.rows * l.stride);
        d.fillBf(rng, t.x, -3, 3);
        if (fill == .stress) for (t.x) |*v| {
            v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32);
        };
        if (fill == .nonfinite) for ([_]f32{ d.nan, d.inf, -d.inf, d.inf }, 0..) |v, i| {
            // a q head (or a context's k head) a value each, then a k head's
            const r = i % shape.rows;
            const at = if (i < 2) l.q + (if (t.heads() == 0) l.k else 0) + 5 + 31 * i else l.k + 128 + 7 * i;
            t.x[r * l.stride + at] = c.toBf16(v);
        };
        t.qw = try t.a.alloc(u16, 128);
        t.kw = try t.a.alloc(u16, 128);
        d.fillBf(rng, t.qw, -0.5, 0.5);
        d.fillBf(rng, t.kw, -0.5, 0.5);
        const sz = t.sizes();
        t.old = try t.a.alloc(u16, sz[0] + sz[1] + sz[2]);
        d.fillBf(rng, t.old, -1, 1);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.qw));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.kw));
        for (&t.outs, sz) |*o, n| o.* = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Prep) !void {
        const sz = t.sizes();
        var at: usize = 0;
        for (t.outs, sz) |o, n| {
            @memcpy(t.rig.bytes(o, n * 2), std.mem.sliceAsBytes(t.old[at..][0..n]));
            at += n;
        }
        const l = t.layout();
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        const Args = extern struct { rows: u32, heads: u32, kv_heads: u32, pos0: u32, stride: u32, q_off: u32, k_off: u32, v_off: u32, ring: u32, rope: u32, eps: f32, theta: f32 };
        p.e.setValue(Args{ .rows = @intCast(t.shape.rows), .heads = t.heads(), .kv_heads = D.kv, .pos0 = t.pos0(), .stride = @intCast(l.stride), .q_off = @intCast(l.q), .k_off = @intCast(l.k), .v_off = @intCast(l.v), .ring = t.ring(), .rope = @intFromBool(t.rope()), .eps = D.eps, .theta = D.theta }, 3);
        for (t.outs, 4..) |o, i| p.e.setBuffer(o.buf, o.off, i);
        p.e.dispatchGroups(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Prep) usize {
        const sz = t.sizes();
        return sz[0] + sz[1] + sz[2];
    }

    /// Where head (row, unit)'s outputs land: its q row, or its ring slot's k (v beside it, in the third buffer).
    fn dest(t: *const Prep, row: usize, unit: usize) usize {
        const sz = t.sizes();
        const h = t.heads();
        if (unit < h) return (row * h + unit) * 128;
        const slot = (t.pos0() + row) % t.ring();
        return sz[0] + (slot * D.kv + unit - h) * 128;
    }

    /// One head's 128 outputs: float64 at the kernel's rounding points (fp32's angle), and the range fp32 allows.
    fn head(t: *const Prep, row: usize, unit: usize, sim: ?Sim, want: *[128]f64, low: *[128]f64, high: *[128]f64) void {
        const l = t.layout();
        const is_q = unit < t.heads();
        const src = t.x[row * l.stride + (if (is_q) l.q + unit * 128 else l.k + (unit - t.heads()) * 128) ..][0..128];
        const w = if (is_q) t.qw else t.kw;
        var sq: [128]f64 = undefined;
        var ss: f64 = 0;
        for (&sq, src) |*s, b| {
            s.* = @as(f64, c.fromBf16(b)) * c.fromBf16(b);
            ss += s.*;
        }
        if (sim) |o| ss = contract.rows.sumIn(o[0], o[1], &sq);
        const inv = 1 / @sqrt(ss / 128 + @as(f64, D.eps));
        // the sum's chain (none for gate 4's exact sums), + eps and precise rsqrt's 2 ulps, halved through ss^-1/2
        const e_inv = (if (t.fill == .stress or sim != null) 0 else c.gamma(9, u)) / 2 + 3 * u;
        var nrm: [128]f64 = undefined;
        var nr: [128]Range = undefined;
        for (src, w, 0..) |b, wb, j| {
            const v: f64 = c.fromBf16(b);
            const wv: f64 = c.fromBf16(wb);
            nrm[j] = bf(wv * bf(v * inv));
            nr[j] = Range.point(v * inv).fp32(e_inv / u + 1).bf16().mul(Range.point(wv)).bf16();
        }
        if (!t.rope()) {
            want.* = nrm;
            for (nr, low, high) |r, *lo, *hi| {
                lo.* = r.lo;
                hi.* = r.hi;
            }
            return;
        }
        const p: u32 = t.pos0() + @as(u32, @intCast(row));
        for (0..128) |j| {
            const i = j % 64;
            const e = f32r(@as(f64, @floatFromInt(2 * i)) / 128);
            const freq = std.math.pow(f64, D.theta, e);
            const angle = f32r(@as(f64, @floatFromInt(p)) * f32r(1 / f32r(freq)));
            // precise pow (16 ulps), the reciprocal (2.5), the product (1); then precise cos and sin (4 ulps of 1)
            const exact = @as(f64, @floatFromInt(p)) / freq;
            const da = exact * 19.5 * u + 0x1p-149;
            const cr = cosRange(exact - da, exact + da);
            const sr = cosRange(exact - da - std.math.pi / 2.0, exact + da - std.math.pi / 2.0);
            const cs = (Range{ .lo = cr.lo - 0x1p-22, .hi = cr.hi + 0x1p-22 }).bf16();
            const sn = (Range{ .lo = sr.lo - 0x1p-22, .hi = sr.hi + 0x1p-22 }).bf16();
            const rot = if (j < 64) -nrm[j + 64] else nrm[j - 64];
            const rr = if (j < 64) nr[j + 64].neg() else nr[j - 64];
            want[j] = bf(bf(nrm[j] * bf(@cos(angle))) + bf(rot * bf(@sin(angle))));
            const y = nr[j].mul(cs).bf16().add(rr.mul(sn).bf16()).fp32(1).bf16();
            low[j] = y.lo;
            high[j] = y.hi;
        }
    }

    fn all(t: *Prep, want: []f64, low: ?[]f64, high: ?[]f64, sim: ?Sim) void {
        const sz = t.sizes();
        for (want, t.old) |*w, o| w.* = c.fromBf16(o);
        if (low) |lo| @memcpy(lo, want);
        if (high) |hi| @memcpy(hi, want);
        const l = t.layout();
        for (0..t.shape.rows) |row| for (0..t.shape.n) |unit| {
            var hw: [128]f64 = undefined;
            var hl: [128]f64 = undefined;
            var hh: [128]f64 = undefined;
            t.head(row, unit, sim, &hw, &hl, &hh);
            const at = t.dest(row, unit);
            @memcpy(want[at..][0..128], &hw);
            if (low) |lo| @memcpy(lo[at..][0..128], &hl);
            if (high) |hi| @memcpy(hi[at..][0..128], &hh);
            if (unit < t.heads()) continue;
            const vs = t.x[row * l.stride + l.v + (unit - t.heads()) * 128 ..][0..128];
            for (want[at + sz[1] ..][0..128], vs, 0..) |*w, b, j| {
                w.* = c.fromBf16(b);
                if (low) |lo| lo[at + sz[1] + j] = w.*;
                if (high) |hi| hi[at + sz[1] + j] = w.*;
            }
        };
    }

    pub fn reference(t: *Prep, want: []f64, mag: []f64) !void {
        t.all(want, null, null, null);
        @memset(mag, 0);
    }
    pub fn intervals(t: *Prep, low: []f64, high: []f64) !void {
        const want = try t.a.alloc(f64, t.count());
        defer t.a.free(want);
        t.all(want, low, high, null);
    }
    pub fn bounds(t: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, 2 * t.count());
        defer t.a.free(buf);
        try t.intervals(buf[0..t.count()], buf[t.count()..]);
        for (allow, want, buf[0..t.count()], buf[t.count()..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) d.spread(w, .{ .lo = lo, .hi = hi }) else 0;
    }
    pub fn simulate(t: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.all(out, null, null, .{ acc, order });
    }
};

// --- rc_df_vec: a block's attention partials over 128-key blocks of a layer's ring --------------------------------

pub const Vec = struct {
    /// A sliding layer, the global layer, and the global layer over a resumed prompt's context (keys from a floor).
    pub const Variant = enum { sliding, global, floor };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    k: []u16 = &.{},
    val: []u16 = &.{},
    old: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    fn ring(t: *const Vec) u32 {
        return D.ring(if (t.v == .sliding) D.window else D.global);
    }

    /// Rows: the block's positions; n: q heads; k: the window (the layer's, and a short one with partial blocks).
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = switch (v) {
                .sliding => "ricursive dflash2 rc_df_vec sliding layer",
                .global => "ricursive dflash2 rc_df_vec global layer",
                .floor => "ricursive dflash2 rc_df_vec global layer above a floor",
            },
            .accum = .f32,
            .depth = vecDepth,
            .out = .f32,
            .nan = .{ .stated = "a NaN score is skipped by the running max (Metal's max, as fmax) and makes the block's mass and output NaN" },
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            .served = switch (v) {
                .sliding => &.{ .{ .rows = D.block, .n = D.heads, .k = D.window }, .{ .rows = D.block, .n = D.heads, .k = 300 } },
                .global, .floor => &.{ .{ .rows = D.block, .n = D.heads, .k = D.global }, .{ .rows = D.block, .n = D.heads, .k = 1300 } },
            },
            .sums = .{ .f16 = 100, .bf16 = 100 },
        };
    }
    /// A score's: q's scale, a lane's 4-term dot, a 5-level simd_sum.
    fn vecDepth(_: c.Shape) usize {
        return 10;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, m: Mutation) !Vec {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, a, "rc_df_vec", m), .v = v, .m = m };
    }
    pub fn deinit(t: *Vec) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Vec) void {
        inline for (.{ "q", "k", "val", "old" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Vec) g.Kernel {
        return controls.mark(.{ .spec = spec(t.v), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Vec);

    const Plan = struct { anchor: u32, hi: u32, floor: u32, first: u32, blocks: u32 };
    /// The anchor near the context's end, row 0's window from 3 keys before a block's end, the floor 1,000 keys in.
    fn plan(t: *const Vec) Plan {
        const w: u32 = @intCast(t.shape.k);
        const lo0 = (farPos(t.ring()) + 3 - w) / 128 * 128 + 125;
        const anchor = lo0 + w - 1;
        const hi = anchor + @as(u32, @intCast(t.shape.rows));
        const floor: u32 = if (t.v == .floor) lo0 + 1000 else 0;
        const lo = @max(lo0, floor);
        return .{ .anchor = anchor, .hi = hi, .floor = floor, .first = lo / 128, .blocks = (hi - 1) / 128 - lo / 128 + 1 };
    }
    fn slots(t: *const Vec) usize {
        return t.shape.n * t.shape.rows * t.plan().blocks;
    }
    fn scale(t: *const Vec) f32 {
        return if (t.fill == .stress) 1 else 1.0 / @sqrt(128.0);
    }

    pub fn prepare(t: *Vec, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > D.block or shape.n != D.heads or shape.k > t.ring() - D.block) return false;
        if (t.v == .floor and shape.k <= 1000) return false; // the floor lies inside row 0's window
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const kvn: usize = @as(usize, t.ring()) * D.kv * 128;
        t.q = try t.a.alloc(u16, shape.rows * D.heads * 128);
        t.k = try t.a.alloc(u16, kvn);
        t.val = try t.a.alloc(u16, kvn);
        d.fillBf(rng, t.q, -3, 3);
        d.fillBf(rng, t.k, -3, 3);
        d.fillBf(rng, t.val, -2, 2);
        if (fill == .stress) {
            // big products, mirrored halves (k negated, a few nudged): partials near 2^20, scores in the hundreds
            for (0..t.q.len / 128) |h| for (0..64) |j| {
                const qv: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
                t.q[h * 128 + j] = c.toBf16(qv);
                t.q[h * 128 + 64 + j] = c.toBf16(qv);
            };
            for (0..kvn / 128) |h| for (0..64) |j| {
                const kv: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
                const nudge: f32 = if (j < 2) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 3))) - 1) else 0;
                t.k[h * 128 + j] = c.toBf16(kv);
                t.k[h * 128 + 64 + j] = c.toBf16(-kv + nudge);
            };
            for (t.val) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 9))) - 4);
        }
        if (fill == .nonfinite) {
            const p = t.plan();
            const r: usize = t.ring();
            // a key's NaN component, +inf in a query, -inf in a value (keys of the last block, kv head 0)
            t.k[((p.hi - 2) % r) * D.kv * 128 + 3] = c.toBf16(d.nan);
            t.q[(1 * D.heads + 5) * 128 + 9] = c.toBf16(d.inf);
            t.val[((p.hi - 1) % r) * D.kv * 128 + 2] = c.toBf16(-d.inf);
        }
        t.old = try t.a.alloc(f32, t.slots() * 128);
        for (t.old) |*o| o.* = rng.float(f32);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.val));
        t.outs[0] = try t.rig.output(t.slots() * 4);
        t.outs[1] = try t.rig.output(t.slots() * 4);
        t.outs[2] = try t.rig.output(t.slots() * 128 * 4);
        return true;
    }

    pub fn run(t: *Vec) !void {
        @memcpy(t.rig.bytes(t.outs[2], t.slots() * 128 * 4), std.mem.sliceAsBytes(t.old));
        const p = t.plan();
        const e = t.rig.begin();
        e.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| e.e.setBuffer(r.buf, r.off, i);
        const Args = extern struct { rows: u32, pos0: u32, heads: u32, groups: u32, blocks: u32, chunk: u32, first: u32, scale: f32, hi: u32, window: u32, ring: u32, floor: u32 };
        e.e.setValue(Args{ .rows = @intCast(t.shape.rows), .pos0 = p.anchor, .heads = D.heads, .groups = D.kv, .blocks = p.blocks, .chunk = 128, .first = p.first, .scale = t.scale(), .hi = p.hi, .window = @intCast(t.shape.k), .ring = t.ring(), .floor = p.floor }, 3);
        for (t.outs, 4..) |o, i| e.e.setBuffer(o.buf, o.off, i);
        e.e.dispatchGroups(mtl.Size.of(D.kv, t.shape.rows, p.blocks), mtl.Size.of(32, D.heads / D.kv, 1));
        try t.rig.finish(e);
    }

    pub fn count(t: *Vec) usize {
        return t.slots() * 130;
    }

    const Out = struct { want: []f64, allow: ?[]f64 = null, sim: ?Sim = null };

    /// Every slot's (m, l, o) in float64 (fmax's NaN rule), and its allowance: spread, fast exp, the sums' chains.
    fn outputs(t: *Vec, o: Out) !void {
        const p = t.plan();
        const n = t.slots();
        const r: usize = t.ring();
        const G = D.heads / D.kv;
        const sc: f64 = t.scale();
        for (o.want[2 * n ..], t.old) |*w, v| w.* = v;
        if (o.allow) |al| @memset(al, 0);
        var s: [128]f64 = undefined;
        var e: [128]f64 = undefined;
        var terms: [128]f64 = undefined;
        for (0..D.heads) |h| for (0..t.shape.rows) |row| for (0..p.blocks) |z| {
            const slot = (h * t.shape.rows + row) * p.blocks + z;
            const pos = p.anchor + row;
            const lo = @max(if (pos + 1 > t.shape.k) pos + 1 - t.shape.k else 0, p.floor);
            const blk = p.first + z;
            const from = @max(lo, blk * 128);
            const to = @min(p.hi, blk * 128 + 128);
            if (from >= to) {
                o.want[slot] = -std.math.inf(f64);
                o.want[n + slot] = 0;
                continue;
            }
            const kv = h / G;
            const keys = to - from;
            var m: f64 = -std.math.inf(f64);
            var least: f64 = std.math.inf(f64);
            var big_e: f64 = 0;
            for (0..keys) |i| {
                const at = ((from + i) % r) * D.kv * 128 + kv * 128;
                var abs: f64 = 0;
                var sum: f64 = 0;
                for (0..128) |j| {
                    terms[j] = sc * c.fromBf16(t.q[(row * D.heads + h) * 128 + j]) * c.fromBf16(t.k[at + j]);
                    sum += terms[j];
                    abs += @abs(terms[j]);
                }
                s[i] = if (o.sim) |q| contract.rows.sumIn(q[0], q[1], &terms) else sum;
                e[i] = if (t.fill == .stress) 0 else c.gamma(10, u) * abs;
                big_e = @max(big_e, e[i]);
                if (!std.math.isNan(s[i])) {
                    m = @max(m, s[i]);
                    least = @min(least, s[i]);
                }
            }
            var l: f64 = 0;
            var wabs: [128]f64 = @splat(0);
            var po: [128]f64 = @splat(0);
            var vmax: [128]f64 = @splat(0);
            for (0..keys) |i| {
                const w = @exp(s[i] - m);
                l += w;
                const at = ((from + i) % r) * D.kv * 128 + kv * 128;
                for (&po, &wabs, 0..) |*y, *wa, j| {
                    const vv: f64 = c.fromBf16(t.val[at + j]);
                    y.* += w * vv;
                    wa.* += w * @abs(vv);
                    vmax[j] = @max(vmax[j], @abs(vv));
                }
            }
            o.want[slot] = m;
            o.want[n + slot] = l;
            for (po, 0..) |y, j| o.want[2 * n + slot * 128 + j] = y;
            const al = o.allow orelse continue;
            const steps: f64 = @floatFromInt((keys + 3) / 4);
            const range = if (std.math.isFinite(m - least)) m - least else 0;
            const rel = (@exp(2 * big_e) - 1) + (3 + 4 * range + 5 * steps) * u + c.gamma(@as(f64, @floatFromInt(keys)) + steps, u);
            // Metal flushes subnormals: a weight or product under 2^-126 may be 0
            const flush = @as(f64, @floatFromInt(keys)) * 0x1p-126;
            al[slot] = big_e + 0x1p-149;
            al[n + slot] = rel * l + flush;
            for (wabs, vmax, 0..) |wa, vm, j| al[2 * n + slot * 128 + j] = rel * wa + flush * (1 + vm);
        };
    }

    pub fn reference(t: *Vec, want: []f64, mag: []f64) !void {
        try t.outputs(.{ .want = want });
        @memset(mag, 0);
    }
    pub fn bounds(t: *Vec, _: []const f64, _: []const f64, allow: []f64) !void {
        const want = try t.a.alloc(f64, t.count());
        defer t.a.free(want);
        try t.outputs(.{ .want = want, .allow = allow });
    }
    pub fn simulate(t: *Vec, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(.{ .want = out, .sim = .{ acc, order } });
    }
};
