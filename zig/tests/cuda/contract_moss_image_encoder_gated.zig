//! moss_image_encoder.cu's tf_me_gated_norm: a 128-wide head RMSNorm by w, gated by silu(z), at bf16 points.
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

/// The encoder's hidden width: 32 heads of these 128 channels, and the weight's length.
const hidden = 4096;
const head = 128;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The norm's launches: every served token count at the encoder's hidden width.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = hidden, .k = 1 };
    return out;
}

const why = "a bf16 output follows the head's sum of squares as ss^-1/2: a bf16 sum errs under an output ulp";

/// A lane's four squares in turn, then the warp's five shuffles.
fn depth(_: c.Shape) usize {
    return head / 32 + 5;
}

pub const Gated = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// x [rows, width] and its gate z [rows, width], as bf16 bits, and the norm's weight [128]
    x: []u16 = &.{},
    z: []u16 = &.{},
    w: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: tokens; n: the hidden width (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the mean's four fp32 steps, the two weight multiplies, the silu's three, and the gate's product
            .ops = 2 * (mi.div_ulps + mi.sqrt_ulps) + 5 * 0.5 + mi.exp_ulps,
            .mids = 2,
            .out = .bf16,
            // a head is 128 wide: the width moves in whole heads, the rows and the reduction do not
            .tiles = .{ .{ .size = 8 }, .{ .size = head, .step = head, .min = head }, .{} },
            .served = &served,
            .sums = .{ .bf16 = 0, .why = why },
        };
    }

    pub fn deinit(t: *Gated) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Gated) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.z);
        a.free(t.w);
        t.x = &.{};
        t.z = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Gated) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gated);

    pub fn prepare(t: *Gated, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n % head != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.x = try a.alloc(u16, n);
        t.z = try a.alloc(u16, n);
        t.w = try a.alloc(u16, head);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gate 4 wants whole values: whole squares sum exactly in fp32, and 128 x 64^2 overflows an fp16 sum
        if (fill == .stress) {
            mi.integers(rng, t.x, 64);
            mi.integers(rng, t.z, 8);
        } else {
            // the rows about N(0, 2), the gate a projection's logits, the weight a norm's own
            mi.randomBf16(rng, t.x, -4, 4);
            mi.randomBf16(rng, t.z, -8, 8);
        }
        mi.randomBf16(rng, t.w, 0.25, 2);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.x[(i * 37 + 5) % n] = v;
                t.z[(i * 53 + 7) % n] = v;
            }
            t.w[5] = mi.nan16;
            t.w[6] = mi.inf16;
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.z);
        t.ins[2] = try mi.input(&t.rig, t.w);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Gated) !void {
        try me.gatedNorm(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.out, t.shape.rows, t.shape.n, mi.eps);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gated) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Head `h` of row `row` from its sum of squares `ss` (range `ssr`): normed, weighted twice, gated, rounded once.
    fn headRow(t: *Gated, ends: *Ends, row: usize, h: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const at = row * t.shape.n + h * head;
        const n = mi.invNorm(ss, ssr, @as(f64, head));
        for (0..head) |j| {
            const x: f64 = c.fromBf16(t.x[at + j]);
            const w: f64 = c.fromBf16(t.w[j]);
            const z: f64 = c.fromBf16(t.z[at + j]);
            // the norm's fp32 scale and the bf16 of x inv, then the weight's product rounded to bf16
            const inner = Range.of(x).mul(n.r).fp32(0.5).bf16();
            const p = Range.of(w).mul(inner).fp32(0.5).bf16();
            // the silu's fp32 roundings, then the gate's one rounding of bf16 x fp32 to bf16
            const gate = mi.siluRange(Range.of(z)).fp32(mi.exp_ulps + 0.5 + mi.div_ulps);
            ends.put(want, mag, at + j, bf(bf(w * bf(x * n.inv)) * mi.silu(z)), p.mul(gate).fp32(0.5).bf16());
        }
    }

    pub fn reference(t: *Gated, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.shape.rows) |row| for (0..t.shape.n / head) |h| {
            const at = row * t.shape.n + h * head;
            const s = mi.squares(t.x[at..][0..head], depth(t.shape), t.fill == .stress);
            t.headRow(&t.ends, row, h, s.ss, s.r, want, mag);
        };
    }
    pub fn bounds(t: *Gated, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Gated, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Gated, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, head);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        for (0..t.shape.rows) |row| for (0..t.shape.n / head) |h| {
            const at = row * t.shape.n + h * head;
            for (terms, t.x[at..][0..head]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = mi.rows.sumIn(acc, order, terms);
            t.headRow(&scratch, row, h, ss, Range.of(ss), out, mag);
        };
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Gated = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.gated_norm = try ctl.function("tf_ctl_me_gated_norm_weight");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_gated_norm", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_gated_norm drops its weight", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
