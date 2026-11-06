//! moss_image_encoder.cu's tf_me_attention: the every-fourth-layer mixer's causal softmax over 256-wide heads.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const attn = @import("contract_attn.zig");
const check = @import("check.zig");

const me = moss.encoder;
const c = mi.c;
const g = mi.g;
const Ends = mi.Ends;
const bf = mi.bf;

/// The head width: a block's 256 threads, one channel each.
const D = 256;

/// The encoder's queries a key head: 16 heads over 4 key-value heads.
const group = 4;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The attention's launches: the 16 heads, each head 256 wide, at every served token count.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = 16, .n = D, .k = t };
    return out;
}

/// The score's scale: 1 / sqrt(256), the fp32 constant the launcher passes.
fn scoreScale() f64 {
    return 1.0 / @sqrt(@as(f64, D));
}

fn scoreScaleF32() f32 {
    return 1.0 / @sqrt(@as(f32, D));
}

/// fmaxf's rule: one number when the other is a NaN, and a NaN when both are.
fn fmax(a: f64, b: f64) f64 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

const why = "an output is a weighted mean of bf16 values rounded once: a bf16 weight sum errs under the output's own ulp";

/// A score's dot: 256 products a block, then block_sum_tree's two butterflies; a step rescales both by exp(m - top).
fn depth(s: c.Shape) usize {
    return 4 * (s.k + 1) + 11;
}

pub const Attn = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    /// q [heads, rows, 256], then k and v [kv heads, rows, 256], as bf16 bits
    q: []u16 = &.{},
    kb: []u16 = &.{},
    vb: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: the queries' heads; n: the head width; k: the prompt's rows.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the score's scale, its subtraction from the maximum, expf, the rescaling's product, the division
            .ops = 3 * 0.5 + mi.exp_ulps + mi.div_ulps,
            .out = .bf16,
            // a key head is four query heads: the heads move in fours, the prompt's rows and the width do not
            .tiles = .{ .{ .size = group, .step = group, .min = group }, .{}, .{ .size = 8 } },
            .served = &served,
            .sums = .{ .bf16 = 0, .why = why },
        };
    }

    pub fn deinit(t: *Attn) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Attn) void {
        const a = t.rig.gpu.gpa;
        for ([_][]u16{ t.q, t.kb, t.vb }) |v| a.free(v);
        t.q = &.{};
        t.kb = &.{};
        t.vb = &.{};
    }
    pub fn kernel(t: *Attn) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Attn);

    pub fn kvHeads(t: *const Attn) usize {
        return t.shape.rows / group;
    }

    fn row(buf: []u16, kvh: usize, rows: usize, j: usize) []u16 {
        return buf[(kvh * rows + j) * D ..][0..D];
    }

    fn keyRow(t: *const Attn, kvh: usize, j: usize) []u16 {
        return row(t.kb, kvh, t.shape.k, j);
    }

    fn valueRow(t: *const Attn, kvh: usize, j: usize) []u16 {
        return row(t.vb, kvh, t.shape.k, j);
    }

    fn queryRow(t: *const Attn, h: usize, i: usize) []u16 {
        return row(t.q, h, t.shape.k, i);
    }

    fn key(t: *const Attn, kvh: usize, j: usize, d: usize) f64 {
        return c.fromBf16(t.keyRow(kvh, j)[d]);
    }

    fn value(t: *const Attn, kvh: usize, j: usize, d: usize) f64 {
        return c.fromBf16(t.valueRow(kvh, j)[d]);
    }

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != D or shape.rows == 0 or shape.rows % group != 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const rows = shape.k;
        t.q = try a.alloc(u16, shape.rows * rows * D);
        t.kb = try a.alloc(u16, t.kvHeads() * rows * D);
        t.vb = try a.alloc(u16, t.kvHeads() * rows * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // q and k a normed head's (a score spreads as N(0, 1) after the scale), v a projection's values
        mi.randomBf16(rng, t.q, -1, 1);
        mi.randomBf16(rng, t.kb, -1, 1);
        mi.randomBf16(rng, t.vb, -4, 4);
        switch (fill) {
            .random => {},
            .nonfinite => t.plant(),
            .stress => try t.stress(rng),
        }
        t.ins[0] = try mi.input(&t.rig, t.q);
        t.ins[1] = try mi.input(&t.rig, t.kb);
        t.ins[2] = try mi.input(&t.rig, t.vb);
        t.out = try t.rig.output(rows * shape.rows * D * 2);
        return true;
    }

    /// Gate 1: a NaN query (fmaxf keeps a finite maximum, so the weights' product is NaN), both infinities, a NaN key.
    fn plant(t: *Attn) void {
        const rows = t.shape.k;
        t.queryRow(0, 0)[3] = mi.nan16;
        t.valueRow(0, rows - 1)[7] = mi.inf16;
        t.valueRow(0, rows / 2)[9] = mi.ninf16;
        if (rows > 2) t.valueRow(0, rows - 2)[11] = mi.inf16;
        if (t.kvHeads() > 1 and rows > 2) t.keyRow(1, rows / 3)[5] = mi.nan16;
    }

    /// Gate 4: one key vector a head (every weight exactly 1, the rescaling too), each value column a mirrored pair.
    fn stress(t: *Attn, rng: std.Random) !void {
        const a = t.rig.gpu.gpa;
        const rows = t.shape.k;
        const one = try a.alloc(u16, D);
        defer a.free(one);
        for (one) |*e| e.* = attn.normal(rng);
        const col = try a.alloc(f32, rows);
        defer a.free(col);
        for (0..t.kvHeads()) |kvh| {
            for (0..rows) |j| @memcpy(t.keyRow(kvh, j), one);
            for (0..D) |d| {
                attn.mirrored(rng, col);
                for (col, 0..) |y, j| t.valueRow(kvh, j)[d] = c.toBf16(y);
            }
        }
    }

    pub fn run(t: *Attn) !void {
        const shape = t.shape;
        try me.attention(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.out, shape.k, shape.rows, t.kvHeads(), scoreScaleF32());
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Attn) usize {
        return t.shape.rows * t.shape.k * D;
    }

    /// Query (i, h) against key j: the dot's float64 value, and its terms' magnitude (the roundings' scale).
    fn score(t: *Attn, i: usize, h: usize, j: usize) f64 {
        var dot: f64 = 0;
        for (0..D) |d| dot += c.fromBf16(t.queryRow(h, i)[d]) * t.key(h / group, j, d);
        return dot;
    }

    fn scoreMag(t: *Attn, i: usize, h: usize, j: usize) f64 {
        var mag: f64 = 0;
        for (0..D) |d| mag += @abs(c.fromBf16(t.queryRow(h, i)[d])) * @abs(t.key(h / group, j, d));
        return mag;
    }

    /// The scaled scores, each weight's band (the dot's eleven roundings, the scale, expf), and fmaxf's maximum.
    fn weights(t: *Attn, i: usize, h: usize, s: []f64, rho: []f64) f64 {
        const scale = scoreScale();
        const u = c.unit(.f32);
        var most: f64 = -std.math.inf(f64);
        for (0..i + 1) |j| {
            s[j] = t.score(i, h, j) * scale;
            most = fmax(most, s[j]);
        }
        for (0..i + 1) |j| {
            const delta = 11 * u * scale * t.scoreMag(i, h, j) + 2 * u * (@abs(s[j]) + @abs(most));
            rho[j] = @exp(delta) - 1 + mi.exp_ulps * 0x1p-23;
        }
        return most;
    }

    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const a = t.rig.gpu.gpa;
        const rows = t.shape.k;
        const s = try a.alloc(f64, rows);
        defer a.free(s);
        const p = try a.alloc(f64, rows);
        defer a.free(p);
        const rho = try a.alloc(f64, rows);
        defer a.free(rho);
        const gamma = c.gamma(@floatFromInt(depth(t.shape)), c.unit(.f32));
        for (0..rows) |i| for (0..t.shape.rows) |h| {
            const most = t.weights(i, h, s, rho);
            var l: f64 = 0;
            var llo: f64 = 0;
            var lhi: f64 = 0;
            for (0..i + 1) |j| {
                p[j] = @exp(s[j] - most);
                l += p[j];
                llo += p[j] * (1 - rho[j]);
                lhi += p[j] * (1 + rho[j]);
            }
            for (0..D) |d| {
                var num: f64 = 0;
                var nlo: f64 = 0;
                var nhi: f64 = 0;
                var abs: f64 = 0;
                for (0..i + 1) |j| {
                    const term = p[j] * t.value(h / group, j, d);
                    num += term;
                    nlo += @min(term * (1 - rho[j]), term * (1 + rho[j]));
                    nhi += @max(term * (1 - rho[j]), term * (1 + rho[j]));
                    abs += @abs(term);
                }
                // the sums' own roundings, gamma(depth) of their terms' magnitude, about the ratio
                const band = gamma * abs / l;
                const at = (i * t.shape.rows + h) * D + d;
                t.ends.put(want, mag, at, num / l, .{ .lo = bf(nlo / lhi - band), .hi = bf(nhi / llo + band) });
            }
        };
    }
    pub fn bounds(t: *Attn, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Attn, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const rows = t.shape.k;
        const terms = try a.alloc(f64, rows);
        defer a.free(terms);
        const ones = try a.alloc(f64, rows);
        defer a.free(ones);
        @memset(ones, 1);
        for (0..rows) |i| for (0..t.shape.rows) |h| {
            for (0..i) |j| if (!std.mem.eql(u16, t.keyRow(h / group, j), t.keyRow(h / group, i))) return error.KeysScoreApart;
            const l = mi.rows.sumIn(acc, order, ones[0 .. i + 1]);
            for (0..D) |d| {
                for (terms[0 .. i + 1], 0..) |*x, j| x.* = t.value(h / group, j, d);
                out[(i * t.shape.rows + h) * D + d] = mi.rows.sumIn(acc, order, terms[0 .. i + 1]) / l;
            }
        };
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Attn = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.attention = try ctl.function("tf_ctl_me_attention_scale");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_attention", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_attention drops its score scale", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
