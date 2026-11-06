//! moss_image_encoder.cu's tf_me_gdn: the gated delta rule's state, a key channel's row a thread, column by column.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const me = moss.encoder;
const c = mi.c;
const g = mi.g;
const Ends = mi.Ends;
const bf = mi.bf;

/// The DeltaNet's key and value geometry: 16 key heads of 128, 32 value heads, two a key head.
const kheads = 16;
const vheads = 32;
const kdim = 128;
const per = vheads / kheads;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The recurrence's launches: every served token count, its 16 key heads and its 32 value heads.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = kheads, .k = vheads };
    return out;
}

const why = "the state is fp32 and the output rounds once to bf16: a bf16 state's error is far under the output's own ulp";

/// A step's two 128-term key sums and the state's update, over the rows in order.
fn depth(s: c.Shape) usize {
    return 3 * s.rows * (kdim + 1);
}

pub const Gdn = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// qn and kn [rows, 16, 128] fp32; the mixed row [rows, 2 x 2048 then 32 x 128] as bf16 bits
    qn: []f32 = &.{},
    kn: []f32 = &.{},
    mix: []u16 = &.{},
    /// the decay's exponent and the write strength [rows, 32], fp32
    g: []f32 = &.{},
    beta: []f32 = &.{},
    ins: [5]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: the prompt's rows; n: the key heads; k: the value heads.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the decay's expf, the two subtractions and the products that carry the state forward
            .ops = 8 * 0.5 + mi.exp_ulps,
            .out = .bf16,
            .tiles = .{ .{ .size = 8 }, .{}, .{} },
            .served = &served,
            .sums = .{ .bf16 = 0, .why = why },
        };
    }

    pub fn deinit(t: *Gdn) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Gdn) void {
        const a = t.rig.gpu.gpa;
        for ([_][]f32{ t.qn, t.kn, t.g, t.beta }) |v| a.free(v);
        a.free(t.mix);
        t.qn = &.{};
        t.kn = &.{};
        t.mix = &.{};
        t.g = &.{};
        t.beta = &.{};
    }
    pub fn kernel(t: *Gdn) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gdn);

    fn mixStride() usize {
        return 2 * kheads * kdim + vheads * kdim;
    }

    fn valueOffset(h: usize, j: usize) usize {
        return 2 * kheads * kdim + h * kdim + j;
    }

    fn keyAt(i: usize, h: usize, ch: usize) usize {
        return (i * kheads + h) * kdim + ch;
    }

    pub fn prepare(t: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != kheads or shape.k != vheads or shape.rows == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const rows = shape.rows;
        t.qn = try a.alloc(f32, rows * kheads * kdim);
        t.kn = try a.alloc(f32, rows * kheads * kdim);
        t.mix = try a.alloc(u16, rows * mixStride());
        t.g = try a.alloc(f32, rows * vheads);
        t.beta = try a.alloc(f32, rows * vheads);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // a normed key head and a normed query head (the prep's qn, scaled), the mixed row a projection's values
        for (0..rows) |i| for (0..kheads) |h| t.normal(rng, i, h);
        mi.randomBf16(rng, t.mix, -4, 4);
        for (0..rows * vheads) |i| {
            // a decay that forgets (exp g below one) and a write strength a sigmoid's
            t.g[i] = -(0.1 + rng.float(f32) * 2.2);
            t.beta[i] = @floatCast(1.0 / (1.0 + @exp(-@as(f64, rng.floatNorm(f32) * 2))));
        }
        switch (fill) {
            .random => {},
            .nonfinite => t.plant(),
            .stress => t.stress(),
        }
        t.ins[0] = try mi.input(&t.rig, t.qn);
        t.ins[1] = try mi.input(&t.rig, t.kn);
        t.ins[2] = try mi.input(&t.rig, t.mix);
        t.ins[3] = try mi.input(&t.rig, t.g);
        t.ins[4] = try mi.input(&t.rig, t.beta);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    /// A row's key head at unit length, and its query head at unit length times the launcher's scale.
    fn normal(t: *Gdn, rng: std.Random, i: usize, h: usize) void {
        const at = keyAt(i, h, 0);
        for ([_][]f32{ t.kn, t.qn }, [_]f32{ 1, 1.0 / @sqrt(@as(f32, kdim)) }) |dst, scale| {
            var ss: f32 = 0;
            for (0..kdim) |ch| {
                const x = rng.floatNorm(f32);
                dst[at + ch] = x;
                ss += x * x;
            }
            const inv = scale / @sqrt(ss);
            for (0..kdim) |ch| dst[at + ch] *= inv;
        }
    }

    /// Gate 1: a NaN query channel, an infinite key channel (its state stays infinite), decay and beta non-finites.
    fn plant(t: *Gdn) void {
        const rows = t.shape.rows;
        t.qn[keyAt(rows - 1, 0, 5)] = std.math.nan(f32);
        t.kn[keyAt(0, 0, 7)] = std.math.inf(f32);
        t.kn[keyAt(rows / 2, 0, 9)] = -std.math.inf(f32);
        t.g[(rows - 1) * vheads + 1] = std.math.nan(f32);
        t.beta[(rows - 1) * vheads + 2] = std.math.inf(f32);
        t.mix[(rows - 1) * mixStride() + valueOffset(3, 3)] = mi.nan16;
    }

    /// Gate 4: positive dyadic values, so fp32 is exact: decay 1, beta 1/128, the last row's sum past fp16's 65504.
    fn stress(t: *Gdn) void {
        for (t.kn) |*x| x.* = 4;
        for (t.qn) |*x| x.* = 1;
        for (t.g) |*x| x.* = 0;
        for (t.beta) |*x| x.* = 1.0 / 128.0;
        for (0..t.shape.rows) |i| for (0..vheads) |h| for (0..kdim) |j| {
            t.mix[i * mixStride() + valueOffset(h, j)] = c.toBf16(8);
        };
    }

    pub fn run(t: *Gdn) !void {
        const geo: me.Gdn = .{ .rows = t.shape.rows, .kheads = kheads, .vheads = vheads, .mix_stride = mixStride(), .scale = 0 };
        try me.gdn(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.ins[4], t.out, geo);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gdn) usize {
        return t.shape.rows * vheads * kdim;
    }

    fn key(t: *const Gdn, i: usize, h: usize, ch: usize) f64 {
        return t.kn[keyAt(i, h, ch)];
    }

    fn query(t: *const Gdn, i: usize, h: usize, ch: usize) f64 {
        return t.qn[keyAt(i, h, ch)];
    }

    fn value(t: *const Gdn, i: usize, h: usize, j: usize) f64 {
        return c.fromBf16(t.mix[i * mixStride() + valueOffset(h, j)]);
    }

    /// The recurrence in float64 and each output's band: the decay, the key sums' chains, the state's carried error.
    pub fn reference(t: *Gdn, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const a = t.rig.gpu.gpa;
        const rows = t.shape.rows;
        const state = try a.alloc(f64, kdim * kdim);
        defer a.free(state);
        const err = try a.alloc(f64, kdim * kdim);
        defer a.free(err);
        const u = c.unit(.f32);
        const chain = c.gamma(kdim + 1, u);
        for (0..vheads) |h| {
            @memset(state, 0);
            @memset(err, 0);
            const kh = h / per;
            for (0..rows) |i| {
                const decay = @exp(@as(f64, t.g[i * vheads + h]));
                const beta = @as(f64, t.beta[i * vheads + h]);
                const decayed = mi.exp_ulps * 0x1p-23;
                for (0..kdim) |j| {
                    var kv: f64 = 0;
                    var kvMag: f64 = 0;
                    var kvErr: f64 = 0;
                    for (0..kdim) |ci| {
                        const at = ci * kdim + j;
                        const was = state[at];
                        state[at] = was * decay;
                        err[at] = (err[at] + @abs(was) * decayed) * @abs(decay) + u * @abs(state[at]);
                        const term = state[at] * t.key(i, kh, ci);
                        kv += term;
                        kvMag += @abs(term);
                        kvErr += err[at] * @abs(t.key(i, kh, ci)) + u * @abs(term);
                    }
                    kvErr += chain * kvMag;
                    const v = t.value(i, h, j);
                    const delta = (v - kv) * beta;
                    const deltaErr = (u * @abs(v - kv) + kvErr) * @abs(beta) + u * @abs(delta);
                    var o: f64 = 0;
                    var oMag: f64 = 0;
                    var oErr: f64 = 0;
                    for (0..kdim) |ci| {
                        const at = ci * kdim + j;
                        const k = t.key(i, kh, ci);
                        state[at] += k * delta;
                        err[at] += @abs(k) * deltaErr + u * @abs(k * delta) + u * @abs(state[at]);
                        const term = state[at] * t.query(i, kh, ci);
                        o += term;
                        oMag += @abs(term);
                        oErr += err[at] * @abs(t.query(i, kh, ci)) + u * @abs(term);
                    }
                    oErr += chain * oMag;
                    const at = (i * vheads + h) * kdim + j;
                    t.ends.put(want, mag, at, o, .{ .lo = bf(o - oErr), .hi = bf(o + oErr) });
                }
            }
        }
    }
    pub fn bounds(t: *Gdn, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Gdn, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Gdn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const rows = t.shape.rows;
        const state = try a.alloc(f64, kdim * kdim);
        defer a.free(state);
        const terms = try a.alloc(f64, kdim);
        defer a.free(terms);
        for (0..vheads) |h| {
            @memset(state, 0);
            const kh = h / per;
            for (0..rows) |i| {
                const decay = @exp(@as(f64, t.g[i * vheads + h]));
                const beta = @as(f64, t.beta[i * vheads + h]);
                for (0..kdim) |j| {
                    for (0..kdim) |ci| {
                        const at = ci * kdim + j;
                        state[at] *= decay;
                        terms[ci] = state[at] * t.key(i, kh, ci);
                    }
                    const kv = mi.rows.sumIn(acc, order, terms);
                    const delta = (t.value(i, h, j) - kv) * beta;
                    for (0..kdim) |ci| {
                        const at = ci * kdim + j;
                        state[at] += t.key(i, kh, ci) * delta;
                        terms[ci] = state[at] * t.query(i, kh, ci);
                    }
                    out[(i * vheads + h) * kdim + j] = mi.rows.sumIn(acc, order, terms);
                }
            }
        }
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Gdn = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.gdn = try ctl.function("tf_ctl_me_gdn_decay");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_gdn", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_gdn drops its decay", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
