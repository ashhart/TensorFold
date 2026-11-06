//! moss_image_encoder.cu's tf_me_gdn_prep: the DeltaNet's normed q and k, its decay g and its write strength beta.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const me = moss.encoder;
const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;

/// The DeltaNet's key and value geometry: 16 key heads of 128, 32 value heads, two a key head.
const kheads = 16;
const vheads = 32;
const kdim = 128;

/// The scale the launcher passes q: 1 / sqrt(128).
fn qScale() f32 {
    return 1.0 / @sqrt(@as(f32, kdim));
}

/// logf's error, as expf's is counted here: two units in the last place.
const log_ulps = 2;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The prep's launches: each served token count, its 16 key heads and its 32 value heads.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = kheads, .k = vheads };
    return out;
}

/// A head's four squares a lane in turn, then the warp's five shuffles.
const depth = 9;

/// One key head's sum of squares in float64 and its range, and the reciprocal square root the kernel takes of it.
const Inv = struct { v: f64, r: Range };

fn ln(x: f64) f64 {
    return @log(x);
}

fn root(x: f64) f64 {
    return @sqrt(x);
}

pub const Prep = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// the mixed row [rows, 2 x 128 key channels then 32 x 128 value channels], as bf16 bits
    mix: []u16 = &.{},
    /// the a and b projections [rows, 32], a_log and dt_bias [32]: the encoder's bf16 tensors
    a: []u16 = &.{},
    b: []u16 = &.{},
    a_log: []u16 = &.{},
    dt_bias: []u16 = &.{},
    ins: [5]u64 = @splat(0),
    /// qn, kn [rows, 16, 128] then g, beta [rows, 32], all fp32 in one output
    out: u64 = 0,
    ends: Ends,

    /// Rows: the prompt's rows; n: the key heads; k: the value heads.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = expectDepth,
            // the sum's eps, its square root and reciprocal, the two products, and the gate's exponentials
            .ops = 3 * 0.5 + mi.sqrt_ulps + mi.div_ulps + mi.exp_ulps + log_ulps,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{}, .{} },
            .served = &served,
        };
    }
    fn expectDepth(_: c.Shape) usize {
        return depth;
    }

    pub fn deinit(t: *Prep) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Prep) void {
        const a = t.rig.gpu.gpa;
        for ([_][]u16{ t.mix, t.a, t.b, t.a_log, t.dt_bias }) |v| a.free(v);
        t.mix = &.{};
        t.a = &.{};
        t.b = &.{};
        t.a_log = &.{};
        t.dt_bias = &.{};
    }
    pub fn kernel(t: *Prep) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Prep);

    fn mixStride() usize {
        return 2 * kheads * kdim + vheads * kdim;
    }

    fn qnAt() usize {
        return 0;
    }

    fn knAt(t: *const Prep) usize {
        return t.shape.rows * kheads * kdim;
    }

    fn gAt(t: *const Prep) usize {
        return 2 * t.shape.rows * kheads * kdim;
    }

    fn betaAt(t: *const Prep) usize {
        return t.gAt() + t.shape.rows * vheads;
    }

    pub fn prepare(t: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != kheads or shape.k != vheads or shape.rows == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const rows = shape.rows;
        t.mix = try a.alloc(u16, rows * mixStride());
        t.a = try a.alloc(u16, rows * vheads);
        t.b = try a.alloc(u16, rows * vheads);
        t.a_log = try a.alloc(u16, vheads);
        t.dt_bias = try a.alloc(u16, vheads);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the mixed row a projection's logits, a and b a projection's, a_log and dt_bias the model's own
        mi.randomBf16(rng, t.mix, -4, 4);
        mi.randomBf16(rng, t.a, -8, 8);
        mi.randomBf16(rng, t.b, -8, 8);
        for (t.a_log, t.dt_bias) |*al, *dt| {
            al.* = c.toBf16(rng.float(f32) * 2 - 1);
            dt.* = c.toBf16(rng.floatNorm(f32));
        }
        switch (fill) {
            .random => {},
            .nonfinite => t.plant(),
            .stress => t.stress(rng),
        }
        t.ins[0] = try mi.input(&t.rig, t.mix);
        t.ins[1] = try mi.input(&t.rig, t.a);
        t.ins[2] = try mi.input(&t.rig, t.b);
        t.ins[3] = try mi.input(&t.rig, t.a_log);
        t.ins[4] = try mi.input(&t.rig, t.dt_bias);
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    /// Gate 1: a NaN and both infinities in the mixed channels (the norms' sums), and in a, b, a_log and dt_bias.
    fn plant(t: *Prep) void {
        const rows = t.shape.rows;
        const stride = mixStride();
        for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
            t.mix[(i * 37 + 5) % (rows * stride)] = v;
            t.a[i * vheads % (rows * vheads)] = v;
            t.b[(i * 13 + rows) % (rows * vheads)] = v;
        }
        for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, [_]usize{ 0, 1, 2 }) |v, i| {
            t.a_log[i] = v;
            t.dt_bias[i] = if (i == 0) c.toBf16(0) else if (v == mi.inf16) mi.ninf16 else mi.inf16;
        }
    }

    /// Gate 4: whole values, so the heads' sums of squares are exact in fp32 and overflow an fp16 sum: 128 x 32^2.
    fn stress(t: *Prep, rng: std.Random) void {
        mi.integers(rng, t.mix, 32);
        mi.integers(rng, t.a, 4);
        mi.integers(rng, t.b, 4);
        for (t.a_log, t.dt_bias) |*al, *dt| {
            al.* = c.toBf16(0);
            dt.* = c.toBf16(0);
        }
    }

    pub fn run(t: *Prep) !void {
        const geo: me.Gdn = .{ .rows = t.shape.rows, .kheads = kheads, .vheads = vheads, .mix_stride = mixStride(), .scale = qScale() };
        try me.gdnPrep(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.ins[4], t.out + qnAt() * 4, t.out + t.knAt() * 4, t.out + t.gAt() * 4, t.out + t.betaAt() * 4, geo);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Prep) usize {
        return t.betaAt() + t.shape.rows * vheads;
    }

    /// A key head's four channels a lane in turn, then the warp's five shuffles: their squares' sum in float64.
    fn squares(t: *Prep, i: usize, h: usize, second: bool) mi.Squares {
        return mi.squares(t.mix[head(i, h, second)..][0..kdim], depth, t.fill == .stress);
    }

    /// 1 / sqrtf(ss + eps) through the kernel's add, sqrtf and division, and the float64 value.
    fn inv(ss: f64, r: Range) Inv {
        const eps: f64 = mi.eps;
        const d = r.add(Range.of(eps)).fp32(0.5);
        return .{ .v = 1 / @sqrt(ss + eps), .r = d.up(root).fp32(mi.sqrt_ulps).recip().fp32(mi.div_ulps) };
    }

    /// The row's key head's first channel: q's, or k's a key head further on.
    fn head(r: usize, h: usize, second: bool) usize {
        return r * mixStride() + @as(usize, if (second) kheads * kdim else 0) + h * kdim;
    }

    /// -expf(a_log) softplus(a + dt_bias) of value head v, and its points.
    fn decay(t: *Prep, i: usize, v: usize, ends: *Ends, want: []f64, mag: []f64) void {
        const at = i * vheads + v;
        const av: f64 = c.fromBf16(t.a[at]);
        const al: f64 = c.fromBf16(t.a_log[v]);
        const dt: f64 = c.fromBf16(t.dt_bias[v]);
        const x = av + dt;
        const soft = if (x > 20) x else @log(1 + @exp(x));
        const ex = Range.of(av).add(Range.of(dt)).fp32(0.5);
        const e = ex.up(mi.exp).fp32(mi.exp_ulps);
        const sp = Range.of(1.0).add(e).fp32(0.5).up(ln).fp32(log_ulps);
        const sr: Range = .{ .lo = if (ex.lo > 20) ex.lo else sp.lo, .hi = if (ex.hi > 20) @max(sp.hi, ex.hi) else sp.hi };
        const er = Range.of(al).up(mi.exp).fp32(mi.exp_ulps);
        const gr = Range.of(-1.0).mul(er).mul(sr).fp32(0.5);
        const y = -@exp(al) * soft;
        ends.put(want, mag, t.gAt() + at, y, gr);
    }

    /// bf16(sigmoid(b)) of value head v, and its points: expf, the add, the division, then the bf16 round.
    fn beta(t: *Prep, i: usize, v: usize, ends: *Ends, want: []f64, mag: []f64) void {
        const at = i * vheads + v;
        const bv: f64 = c.fromBf16(t.b[at]);
        const e = Range.of(-bv).up(mi.exp).fp32(mi.exp_ulps);
        const r = Range.of(1.0).add(e).fp32(0.5).recip().fp32(mi.div_ulps).bf16();
        ends.put(want, mag, t.betaAt() + at, bf(1 / (1 + @exp(-bv))), r);
    }

    pub fn reference(t: *Prep, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const scale = @as(f64, qScale());
        for (0..t.shape.rows) |i| {
            for (0..kheads) |h| {
                const q = t.squares(i, h, false);
                const kd = t.squares(i, h, true);
                const qn = inv(q.ss, q.r);
                const kn = inv(kd.ss, kd.r);
                for (0..kdim) |ch| {
                    const at = (i * kheads + h) * kdim + ch;
                    const qv: f64 = c.fromBf16(t.mix[head(i, h, false) + ch]);
                    const kv: f64 = c.fromBf16(t.mix[head(i, h, true) + ch]);
                    t.ends.put(want, mag, qnAt() + at, qv * qn.v * scale, Range.of(qv).mul(qn.r).fp32(0.5).mul(Range.of(scale)).fp32(0.5));
                    t.ends.put(want, mag, t.knAt() + at, kv * kn.v, Range.of(kv).mul(kn.r).fp32(0.5));
                }
            }
            for (0..vheads) |v| {
                t.decay(i, v, &t.ends, want, mag);
                t.beta(i, v, &t.ends, want, mag);
            }
        }
    }
    pub fn bounds(t: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Prep, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, kdim);
        defer a.free(terms);
        const scale = @as(f64, qScale());
        for (0..t.shape.rows) |i| {
            for (0..kheads) |h| {
                for ([_]bool{ false, true }) |second| {
                    for (terms, t.mix[head(i, h, second)..][0..kdim]) |*v, x| {
                        const f: f64 = c.fromBf16(x);
                        v.* = f * f;
                    }
                    const n = inv(mi.rows.sumIn(acc, order, terms), Range.of(0));
                    for (0..kdim) |ch| {
                        const at = (i * kheads + h) * kdim + ch;
                        const x: f64 = c.fromBf16(t.mix[head(i, h, second) + ch]);
                        if (second) out[t.knAt() + at] = x * n.v else out[qnAt() + at] = x * n.v * scale;
                    }
                }
            }
            for (0..vheads) |v| {
                const at = i * vheads + v;
                const x: f64 = c.fromBf16(t.a[at]) + c.fromBf16(t.dt_bias[v]);
                const soft = if (x > 20) x else @log(1 + @exp(x));
                out[t.gAt() + at] = -@exp(@as(f64, c.fromBf16(t.a_log[v]))) * soft;
                out[t.betaAt() + at] = bf(1 / (1 + @exp(-@as(f64, c.fromBf16(t.b[at])))));
            }
        }
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Prep = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.gdn_prep = try ctl.function("tf_ctl_me_gdn_prep_beta");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_gdn_prep", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_gdn_prep keeps beta in fp32", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
