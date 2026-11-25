//! moss_image_decoder.cu's tf_md_norm under the contract: a GroupNorm by the statistics' mean and rstd, then SiLU.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const md = moss.decoder;
const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;

/// The decoder's two calls: the norm alone (the mid block's attention), and the norm then SiLU.
pub const plain = 1;
pub const silu = 2;

fn isSilu(shape: c.Shape) ?bool {
    return switch (shape.k) {
        plain => false,
        silu => true,
        else => null,
    };
}

/// The (pixels, channels) pairs its SiLU norms run at: each picture's four levels at 384, 384, then 192 and 96.
const pairs = [_][2]usize{
    .{ 1456, 384 }, .{ 5824, 384 }, .{ 23296, 384 }, .{ 23296, 192 }, .{ 93184, 192 }, .{ 93184, 96 }, // 416 x 224
    .{ 5120, 384 },  .{ 20480, 384 }, .{ 81920, 384 },  .{ 81920, 192 },  .{ 327680, 192 },  .{ 327680, 96 }, // 512 x 640
    .{ 16384, 384 }, .{ 65536, 384 }, .{ 262144, 384 }, .{ 262144, 192 }, .{ 1048576, 192 }, .{ 1048576, 96 },
};

/// The mid block's attention reads the same statistics at each picture's first level, without SiLU.
const attention = [_]usize{ 1456, 5120, 16384 };

/// The norm launches: every pair's, then the attention's plain ones.
const served = servedShapes();

fn servedShapes() [pairs.len + attention.len]c.Shape {
    var out: [pairs.len + attention.len]c.Shape = undefined;
    for (pairs, 0..) |p, i| out[i] = .{ .rows = p[0], .n = p[1], .k = silu };
    for (attention, pairs.len..) |pixels, i| out[i] = .{ .rows = pixels, .n = 384, .k = plain };
    return out;
}

pub const Norm = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    silu: bool = false,
    /// x [pixels, channels] and the statistics' mean | rstd [64], as fp32 bits
    x: []f32 = &.{},
    s: []f32 = &.{},
    /// gamma and beta [channels]
    v: []f32 = &.{},
    ins: [4]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: pixels; n: channels (a multiple of the 32 groups); k: plain or SiLU.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            // the mean's subtraction, rstd's and gamma's products, beta's add, then SiLU's exp, add and division
            .ops = 4 * 0.5 + 2 * mi.exp_ulps + mi.div_ulps,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 32, .step = 32, .min = 32 }, .{} },
            .served = &served,
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
        a.free(t.s);
        a.free(t.v);
        t.x = &.{};
        t.s = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Norm) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const want = isSilu(shape) orelse return false;
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.silu = want;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.x = try a.alloc(f32, n);
        t.s = try a.alloc(f32, 64);
        t.v = try a.alloc(f32, 2 * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the activations about N(0, 1), the statistics a checkpoint's, gamma and beta a norm's
        for (t.x) |*e| e.* = rng.floatNorm(f32);
        for (t.s[0..32]) |*e| e.* = 0.1 * rng.floatNorm(f32);
        for (t.s[32..]) |*e| e.* = 0.5 + rng.float(f32) * 1.5;
        for (t.v[0..shape.n]) |*e| e.* = 0.25 + rng.float(f32) * 1.75;
        for (t.v[shape.n..]) |*e| e.* = rng.float(f32) - 0.5;
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
                t.x[(i * 37 + 5) % n] = v;
            }
            t.s[32 + 3] = -std.math.inf(f32);
            t.s[7] = std.math.inf(f32);
            t.v[5] = std.math.nan(f32);
            t.v[shape.n + 6] = std.math.inf(f32);
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.s);
        t.ins[2] = try mi.input(&t.rig, t.v[0..shape.n]);
        t.ins[3] = try mi.input(&t.rig, t.v[shape.n..]);
        t.out = try t.rig.output(n * 4);
        return true;
    }

    pub fn run(t: *Norm) !void {
        try md.norm(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.out, t.shape.rows, t.shape.n, t.silu);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const channels = t.shape.n;
        const size = channels / 32;
        for (0..t.shape.rows) |px| for (0..channels) |ch| {
            const i = px * channels + ch;
            const grp = ch / size;
            const xv: f64 = t.x[i];
            const mean: f64 = t.s[grp];
            const rstd: f64 = t.s[32 + grp];
            const gc: f64 = t.v[ch];
            const bc: f64 = t.v[channels + ch];
            const r = Range.of(xv).add(Range.of(-mean)).fp32(0.5).mul(Range.of(rstd)).fp32(0.5).mul(Range.of(gc)).fp32(0.5).add(Range.of(bc)).fp32(0.5);
            const v: f64 = ((xv - mean) * rstd) * gc + bc;
            if (t.silu) {
                const neg = Range{ .lo = -r.hi, .hi = -r.lo };
                const down = neg.up(mi.exp).fp32(mi.exp_ulps).add(Range.of(1)).fp32(0.5);
                t.ends.put(want, mag, i, v / (1 + @exp(-v)), r.mul(down.recip()).fp32(mi.div_ulps));
            } else t.ends.put(want, mag, i, v, r);
        };
    }
    /// The allowance's ends about the float64 value: the chain's roundings, SiLU's exp and division included.
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

/// The decoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: md.Kernels,
    t: [2]Norm = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try md.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: Ends = .{ .a = gpu.gpa };
        var with = k;
        with.norm = try ctl.function("tf_ctl_md_norm_beta");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_norm", .ends = ends };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_norm drops beta", .control = .shapes, .ends = ends };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
