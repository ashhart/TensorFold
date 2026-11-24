//! moss_image_encoder.cu's tf_me_norm: a zero-centered RMSNorm by (1 + w), checked at the publisher's bf16 points.
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

/// The encoder's hidden width: the norm's reduction and its rows' stride.
const hidden = 4096;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The norm's launches: every served token count at the encoder's hidden width.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = hidden, .k = hidden };
    return out;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// A thread's share of the row in turn, then the block tree's two five-shuffle warp sums.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

pub const Norm = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// x [rows, width] and the norm's weight [width], as bf16 bits
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: tokens; k: the hidden width (n unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 2 * (mi.div_ulps + mi.sqrt_ulps) + 1,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 256 } },
            .served = &served,
            .sums = .{ .bf16 = 0, .why = why },
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Norm) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Norm) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const k = shape.k;
        const n = shape.rows * k;
        t.x = try a.alloc(u16, n);
        t.w = try a.alloc(u16, k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the rows about N(0, 1), the weight a norm's own, so 1 + w runs from 1.25 to 3
        if (fill == .stress) mi.integers(rng, t.x, 32) else mi.randomBf16(rng, t.x, -3, 3);
        mi.randomBf16(rng, t.w, 0.25, 2);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| t.x[(i * 37 + 5) % n] = v;
            if (k > 6) {
                t.w[5] = mi.nan16;
                t.w[6] = mi.inf16;
            }
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.w);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Norm) !void {
        const k = t.shape.k;
        try me.norm(&t.k, t.rig.s, t.ins[0], t.ins[1], t.out, t.shape.rows, k, k, k, mi.eps);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * t.shape.k;
    }

    /// Row `r` from its sum of squares `ss` (range `ssr`): (x inv) (1 + w), rounded to bf16 once.
    fn row(t: *Norm, ends: *Ends, r: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const k = t.shape.k;
        const n = mi.invNorm(ss, ssr, @floatFromInt(k));
        for (0..k) |j| {
            const x: f64 = c.fromBf16(t.x[r * k + j]);
            const w: f64 = c.fromBf16(t.w[j]);
            const gain = Range.of(1.0).add(Range.of(w)).fp32(0.5);
            const y = bf(x * n.inv * (1 + w));
            const yr = Range.of(x).mul(n.r).fp32(0.5).mul(gain).fp32(0.5).bf16();
            ends.put(want, mag, r * k + j, y, yr);
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const k = t.shape.k;
        for (0..t.shape.rows) |r| {
            const s = mi.squares(t.x[r * k ..][0..k], depth(t.shape), t.fill == .stress);
            t.row(&t.ends, r, s.ss, s.r, want, mag);
        }
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const k = t.shape.k;
        const terms = try a.alloc(f64, k);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        for (0..t.shape.rows) |r| {
            for (terms, t.x[r * k ..][0..k]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = mi.rows.sumIn(acc, order, terms);
            t.row(&scratch, r, ss, Range.of(ss), out, mag);
        }
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Norm = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.norm = try ctl.function("tf_ctl_me_norm_weight");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_norm", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_norm drops (1 + w)", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
