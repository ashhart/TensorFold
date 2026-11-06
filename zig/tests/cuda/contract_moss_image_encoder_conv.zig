//! moss_image_encoder.cu's tf_me_conv: the DeltaNet's 4-tap causal conv and its silu, at the publisher's bf16 points.
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

/// The DeltaNet's mixed row: q and k's 16 heads of 128, then v's 32, and the conv's channel count.
const mix = 8192;

/// The conv's taps, as the checkpoint's linear_conv_kernel_dim.
const taps = 4;

/// The prompt rows the encoder launches: a chat-formatted turn's 33, and the shortest prompt's 10.
const tokens = [_]usize{ 33, 10 };

/// The conv's launches: every served token count at the encoder's mixed width.
const served = servedShapes();

fn servedShapes() [tokens.len]c.Shape {
    var out: [tokens.len]c.Shape = undefined;
    for (&out, tokens) |*o, t| o.* = .{ .rows = t, .n = mix, .k = 1 };
    return out;
}

/// Four taps a thread, each add rounded once.
fn depth(_: c.Shape) usize {
    return taps;
}

const why = "four taps a value: a reduced accumulator's roundings stay under the bf16 store's own";

pub const Conv = struct {
    rig: rig_.Rig,
    k: me.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    /// x [rows, width] and the taps [4, width], as bf16 bits
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: tokens; n: the mixed width (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the silu's expf, its add and its division, each rounded once
            .ops = mi.exp_ulps + 0.5 + mi.div_ulps,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} },
            .served = &served,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = why },
        };
    }

    pub fn deinit(t: *Conv) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Conv) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Conv) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Conv);

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const w = shape.n;
        const n = shape.rows * w;
        t.x = try a.alloc(u16, n);
        t.w = try a.alloc(u16, taps * w);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gate 4 needs whole values: four whole taps sum exactly in fp32, in any order
        if (fill == .stress) {
            mi.integers(rng, t.x, 8);
            mi.integers(rng, t.w, 4);
        } else {
            // the mixed row about N(0, 2), the conv's taps a filter's own
            mi.randomBf16(rng, t.x, -4, 4);
            mi.randomBf16(rng, t.w, -2, 2);
        }
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.x[(i * 37 + 5) % n] = v;
                t.w[(i * 53 + 7) % (taps * w)] = v;
            }
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.w);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Conv) !void {
        try me.conv(&t.k, t.rig.s, t.ins[0], t.ins[1], t.out, t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Conv) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Tap `tap`'s product for output (row, col): a bf16 tap times a bf16 row, skipped before the first row.
    fn product(t: *const Conv, row: usize, col: usize, tap: usize) f64 {
        if (row + tap < taps - 1) return 0;
        const x: f64 = c.fromBf16(t.x[(row + tap + 1 - taps) * t.shape.n + col]);
        return x * @as(f64, c.fromBf16(t.w[col * taps + tap]));
    }

    /// A tap sum `s` (range `sr`) through the bf16 rounding, the silu's fp32 roundings and the store's.
    fn value(s: f64, sr: Range) struct { want: f64, r: Range } {
        return .{ .want = bf(mi.silu(bf(s))), .r = mi.siluRange(sr.bf16()).fp32(mi.exp_ulps + 0.5 + mi.div_ulps).bf16() };
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const w = t.shape.n;
        for (0..t.shape.rows) |row| for (0..w) |col| {
            var s: f64 = 0;
            var sr = Range.of(0);
            var terms: f64 = 0;
            for (0..taps) |tap| {
                const p = t.product(row, col, tap);
                s += p;
                terms += @abs(p);
                sr = sr.add(Range.of(p)).fp32(0.5);
            }
            const v = value(s, sr);
            t.ends.put(want, mag, row * w + col, v.want, v.r);
            // the bound follows the tap sum's terms: the silu's slope carries them to the output
            mag[row * w + col] = terms;
        };
    }
    pub fn bounds(t: *Conv, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Conv, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Conv, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, taps);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        const w = t.shape.n;
        for (0..t.shape.rows) |row| for (0..w) |col| {
            for (terms, 0..) |*p, tap| p.* = t.product(row, col, tap);
            const s = mi.rows.sumIn(acc, order, terms);
            const v = value(s, Range.of(s));
            scratch.put(out, mag, row * w + col, v.want, v.r);
        };
    }
};

/// The encoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: me.Kernels,
    t: [2]Conv = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try me.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.conv = try ctl.function("tf_ctl_me_conv_silu");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image encoder tf_me_conv", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image encoder tf_me_conv drops its silu", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
