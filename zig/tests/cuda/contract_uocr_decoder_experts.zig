//! tf_uocr_expert_gu and tf_uocr_expert_down under the contract: one token's 6 experts by their address table.
const std = @import("std");
const ud = @import("contract_uocr_decoder.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = ud.c;
const g = ud.g;
const mi = ud.mi;
const H = ud.H;
const E = ud.expert_ffn;
/// Distinct experts' weights the test holds; table entry e points at pool e % 8.
const pools = 8;
const pool_elems = 3 * E * H;

/// A lane's 8 values a step (k / 256 steps), each step's 8 products in two 4-chains, then the warp's 5 shuffles.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 4 + 5;
}

pub const Experts = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    /// Gate and up rows of the token (rows 2 x 896 an expert, k 1280); else the down rows (1280 an expert, k 896).
    gu: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// the input rows: x [1280], or the 6 experts' activations [6, 896]
    x: []u16 = &.{},
    w: []u16 = &.{},
    ids: [ud.topk]u32 = undefined,
    ins: [4]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: one token; n: its 6 experts' outputs; k: the reduction (1280 or 896).
    pub fn spec(name: []const u8, gu: bool) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .served = if (gu) &.{.{ .rows = 1, .n = ud.topk * 2 * E, .k = H }} else &.{.{ .rows = 1, .n = ud.topk * H, .k = E }},
        };
    }

    pub fn deinit(t: *Experts) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Experts) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Experts) g.Kernel {
        return .{ .spec = spec(t.name, t.gu), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Experts);

    /// Pool p's gate (0), up (1) or down (2) weights.
    fn part(t: *Experts, p: usize, which: usize) []u16 {
        return t.w[p * pool_elems + which * E * H ..][0 .. E * H];
    }

    /// Output o's weight row and input row.
    fn rows(t: *Experts, o: usize) struct { w: []const u16, x: []const u16 } {
        if (t.gu) {
            const j = o / (2 * E);
            const r = o % (2 * E);
            const which: usize = if (r < E) 0 else 1;
            return .{ .w = t.part(t.ids[j] % pools, which)[(r % E) * H ..][0..H], .x = t.x };
        }
        const j = o / H;
        return .{ .w = t.part(t.ids[j] % pools, 2)[(o % H) * E ..][0..E], .x = t.x[j * E ..][0..E] };
    }

    pub fn prepare(t: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // one token whatever the rows (the single-shape gates widen them): its n outputs over k
        const want: c.Shape = spec(t.name, t.gu).served[0];
        if (shape.rows == 0 or shape.n != want.n or shape.k != want.k) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try a.alloc(u16, if (t.gu) H else ud.topk * E);
        t.w = try a.alloc(u16, pools * pool_elems);
        var taken: u64 = 0;
        for (&t.ids) |*id| {
            var e = rng.uintLessThan(u32, ud.experts);
            while ((taken >> @intCast(e)) & 1 != 0) e = rng.uintLessThan(u32, ud.experts);
            taken |= @as(u64, 1) << @intCast(e);
            id.* = e;
        }
        if (fill == .stress) {
            // inputs 1, each weight row mirrored over its columns: fp32 sums exact, fp16 and bf16 lose bits
            @memset(t.x, c.toBf16(1));
            const k = shape.k;
            const col = try a.alloc(f32, k);
            defer a.free(col);
            var i: usize = 0;
            while (i < t.w.len) : (i += k) {
                attn.mirrored(rng, col);
                for (col, t.w[i..][0..k]) |v, *o| o.* = c.toBf16(v);
            }
        } else {
            for (t.x) |*v| v.* = attn.normal(rng);
            mi.randomBf16(rng, t.w, -0.25, 0.25);
        }
        if (fill == .nonfinite) {
            if (t.gu) {
                t.part(t.ids[0] % pools, 0)[5] = mi.nan16;
                t.part(t.ids[1] % pools, 1)[3 * H + 9] = mi.inf16;
                t.part(t.ids[2] % pools, 0)[7 * H + 1] = mi.ninf16;
            } else {
                t.x[3] = mi.nan16;
                t.part(t.ids[1] % pools, 2)[2 * E + 9] = mi.inf16;
            }
        }
        var table: [ud.experts * 3]u64 = undefined;
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.w);
        for (0..ud.experts) |e| for (0..3) |which| {
            table[e * 3 + which] = t.ins[1] + ((e % pools) * pool_elems + which * E * H) * 2;
        };
        t.ins[2] = try mi.input(&t.rig, &table);
        t.ins[3] = try mi.input(&t.rig, &t.ids);
        t.out = try t.rig.output(shape.n * 2);
        return true;
    }

    pub fn run(t: *Experts) !void {
        const i = t.ins;
        if (t.gu) {
            try t.k.run("expert_gu", t.rig.s, .{ ud.topk * 2 * E / 32, 1, 1 }, 256, .{ i[0], i[2], i[3], t.out, E });
        } else {
            try t.k.run("expert_down", t.rig.s, .{ ud.topk * H / 32, 1, 1 }, 256, .{ i[0], i[2], i[3], t.out, E });
        }
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Experts) usize {
        return t.shape.n;
    }

    pub fn reference(t: *Experts, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, o| {
            const r = t.rows(o);
            var s: f64 = 0;
            var abs: f64 = 0;
            for (r.w, r.x) |a, b| {
                const p = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                s += p;
                abs += @abs(p);
            }
            w.* = s;
            m.* = abs;
        }
    }

    pub fn simulate(t: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, t.shape.k);
        defer a.free(terms);
        for (out, 0..) |*y, o| {
            const r = t.rows(o);
            for (terms, r.w, r.x) |*v, w, x| v.* = @as(f64, c.fromBf16(w)) * c.fromBf16(x);
            y.* = mi.rows.sumIn(acc, order, terms);
        }
    }
};
