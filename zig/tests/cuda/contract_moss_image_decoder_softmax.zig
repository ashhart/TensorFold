//! moss_image_decoder.cu's tf_md_softmax under the contract: the mid block's attention row, scaled and normalized.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const md = moss.decoder;
const c = mi.c;
const g = mi.g;

/// The decoder's score scale, 1 / sqrt(384): the mid block's one head of 384, as its host computes it.
const scale: f32 = 1.0 / @sqrt(@as(f32, 384));

/// Gate 4's far score: scaled and taken from a row's maximum of 0 it underflows, in expf and in float64 alike.
const underflow: f32 = -20000;

/// The mid block's positions: 1456 (52 x 28 of 416 x 224), 5120 (64 x 80) and 16384 (128 x 128), smallest first.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 1456, .k = 1456 },
    .{ .rows = 1, .n = 5120, .k = 5120 },
    .{ .rows = 1, .n = 16384, .k = 16384 },
};

/// A thread's share of the row, block_sum_tree's two warp sums (5 steps each), and the exponentials' roundings.
fn depth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 10 + 4;
}

pub const Softmax = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    /// the scores [rows, rows]: the pristine ones an input, the run's its own output (the kernel is in place)
    scores: []f32 = &.{},
    src: u64 = 0,
    out: u64 = 0,

    /// n: the row's length and the rows there are (a square score matrix); rows unused.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the score's product, the maximum's subtraction, and the row's division by its sum
            .ops = 2 * 0.5 + mi.exp_ulps + mi.div_ulps,
            .out = .f32,
            .in_place = true,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &served,
        };
    }

    pub fn deinit(t: *Softmax) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Softmax) void {
        t.rig.gpu.gpa.free(t.scores);
        t.scores = &.{};
    }
    pub fn kernel(t: *Softmax) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Softmax);

    pub fn prepare(t: *Softmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.n;
        t.scores = try a.alloc(f32, n * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the attention's scores, q k^T of unit vectors: about N(0, 384) before the scale
        for (t.scores) |*e| e.* = switch (fill) {
            // gate 4: a seventh underflow, the rest sit at the row's maximum: exponentials 1 or 0, the sum their count
            .stress => if (rng.uintLessThan(u32, 8) == 0) underflow else 0,
            else => 20 * rng.floatNorm(f32),
        };
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| t.scores[(i * n + i + 3) % (n * n)] = v;
        }
        t.src = try mi.input(&t.rig, t.scores);
        t.out = try t.rig.output(n * n * 4);
        return true;
    }

    pub fn run(t: *Softmax) !void {
        // in place: the row's pristine scores go back before each independent run
        const bytes = t.count() * 4;
        try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.out, .len = bytes }).copyFrom(0, t.src, bytes, t.rig.s.handle);
        try md.softmax(&t.k, t.rig.s, t.out, t.shape.n, scale);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Softmax) usize {
        return t.shape.n * t.shape.n;
    }

    /// The row's scaled scores, its maximum over the ones that are not NaN (fmaxf drops those), and each exponential.
    fn exponentials(t: *Softmax, r: usize, terms: []f64) void {
        const row = t.scores[r * t.shape.n ..][0..t.shape.n];
        var most: f64 = -std.math.inf(f64);
        for (row) |e| {
            const v: f64 = @as(f32, e * scale);
            if (!std.math.isNan(v)) most = @max(most, v);
        }
        const highest: f32 = @floatCast(most);
        for (terms, row) |*v, e| {
            // the kernel's own arithmetic: the score times the scale, the maximum's subtraction, then expf
            const down: f32 = e * scale - highest;
            v.* = @exp(@as(f64, down));
        }
    }

    pub fn reference(t: *Softmax, want: []f64, mag: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, t.shape.n);
        defer a.free(terms);
        for (0..t.shape.n) |r| {
            t.exponentials(r, terms);
            var total: f64 = 0;
            for (terms) |e| total += e;
            for (terms, 0..) |e, j| {
                const i = r * t.shape.n + j;
                want[i] = e / total;
                // the divisor is the row's own sum: the bound is this output's own magnitude, `depth` deep
                mag[i] = if (std.math.isFinite(want[i])) @abs(want[i]) else 0;
            }
        }
    }

    pub fn simulate(t: *Softmax, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, t.shape.n);
        defer a.free(terms);
        for (0..t.shape.n) |r| {
            t.exponentials(r, terms);
            const total = mi.rows.sumIn(acc, order, terms);
            for (terms, 0..) |e, j| out[r * t.shape.n + j] = e / total;
        }
    }
};

/// The decoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: md.Kernels,
    t: [2]Softmax = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try md.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        var with = k;
        with.softmax = try ctl.function("tf_ctl_md_softmax_scale");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_softmax" };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_softmax drops its score scale", .control = .shapes };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
