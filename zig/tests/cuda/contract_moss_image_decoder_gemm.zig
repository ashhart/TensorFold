//! moss_image_decoder.cu's tf_md_gemm under the contract: a 1 x 1 convolution's sums, and the attention's products.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");
const check = @import("check.zig");

const md = moss.decoder;
const c = mi.c;
const g = mi.g;

/// One of the decoder's launches: its rows, outputs a row and reduction, the strides it reads, and its flags.
const Call = struct {
    shape: c.Shape,
    lda: usize,
    ldb: usize,
    /// The mid block's scores read q and k as packed rows of 3 x 384, an output row apiece.
    transposed: bool = false,
    add: bool = false,
};

/// The decoder's launches at 1456 and 16384 pixels; the 16384-squared scores stay at the smaller pictures (4 GiB).
const calls = [_]Call{
    // post_quant_conv's 1 x 1 at 416 x 224: 32 channels to 32
    .{ .shape = .{ .rows = 1456, .n = 32, .k = 32 }, .lda = 32, .ldb = 32 },
    // the mid block's q | k | v at 416 x 224: 384 to 1152
    .{ .shape = .{ .rows = 1456, .n = 1152, .k = 384 }, .lda = 384, .ldb = 1152 },
    // the attention's out projection at 416 x 224, over the block's input
    .{ .shape = .{ .rows = 1456, .n = 384, .k = 384 }, .lda = 384, .ldb = 384, .add = true },
    // the scores at 416 x 224: q k^T of the packed q | k | v rows
    .{ .shape = .{ .rows = 1456, .n = 1456, .k = 384 }, .lda = 1152, .ldb = 1152, .transposed = true },
    // the attention's out projection at 1024 x 1024, over the block's input
    .{ .shape = .{ .rows = 16384, .n = 384, .k = 384 }, .lda = 384, .ldb = 384, .add = true },
};

fn servedShapes() [calls.len]c.Shape {
    var out: [calls.len]c.Shape = undefined;
    for (&out, calls) |*o, call| o.* = call.shape;
    return out;
}

/// The served shapes, in the order the decoder launches them.
const served = servedShapes();

/// The launch of a prepared shape: the decoder's own call for a served one, else the plain form the edges probe.
fn callFor(shape: c.Shape) Call {
    for (calls) |call| if (std.meta.eql(call.shape, shape)) return call;
    return .{ .shape = shape, .lda = shape.k, .ldb = shape.n };
}

/// The products and the sum they feed, rounded once each: two roundings a term, then the bias and the residual.
fn depth(s: c.Shape) usize {
    return s.k + 2;
}

pub const Gemm = struct {
    rig: rig_.Rig,
    k: md.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    call: Call = undefined,
    /// A [rows, lda], and B the rows of ldb: [reduction, ldb] or (transposed) [outputs, ldb]
    x: []f32 = &.{},
    w: []f32 = &.{},
    /// bias [n] and the residual [rows, n] the call adds
    v: []f32 = &.{},
    ins: [4]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: pixels; n: outputs; k: the reduction.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            // the bias's and the residual's adds (a call without the residual takes one fewer)
            .ops = 2,
            .out = .f32,
            .tiles = .{ .{ .size = 64 }, .{ .size = 64 }, .{ .size = 16, .step = 16, .min = 16 } },
            .served = &served,
        };
    }

    pub fn deinit(t: *Gemm) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Gemm) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        a.free(t.v);
        t.x = &.{};
        t.w = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Gemm) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gemm);

    pub fn prepare(t: *Gemm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        const call = callFor(shape);
        if (call.lda < shape.k or call.ldb < (if (call.transposed) shape.k else shape.n)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.call = call;
        const a = t.rig.gpu.gpa;
        const m = shape.rows;
        const w = if (call.transposed) (shape.n - 1) * call.ldb + shape.k else (shape.k - 1) * call.ldb + shape.n;
        t.x = try a.alloc(f32, (m - 1) * call.lda + shape.k);
        t.w = try a.alloc(f32, w);
        t.v = try a.alloc(f32, 2 * shape.n + m * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gate 4 needs whole values: whole products of whole values sum exactly in fp32, in any order
        for (t.x) |*e| e.* = switch (fill) {
            .stress => @floatFromInt(rng.intRangeAtMost(i32, -32, 32)),
            else => 2 * rng.floatNorm(f32),
        };
        for (t.w) |*e| e.* = switch (fill) {
            .stress => @floatFromInt(rng.intRangeAtMost(i32, -32, 32)),
            else => 0.5 * rng.floatNorm(f32),
        };
        for (t.v[0..shape.n]) |*e| e.* = if (fill == .stress) @floatFromInt(rng.intRangeAtMost(i32, -8, 8)) else 0.5 * rng.floatNorm(f32);
        for (t.v[shape.n..]) |*e| e.* = if (fill == .stress) @floatFromInt(rng.intRangeAtMost(i32, -8, 8)) else rng.floatNorm(f32);
        if (fill == .nonfinite) {
            // a row apiece, so no row mixes two infinities into a NaN the kernel could not also make
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
                t.x[(i * t.call.lda + i) % t.x.len] = v;
                t.w[(i * 7 + 3) % t.w.len] = v;
            }
            t.v[1] = std.math.nan(f32);
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.w);
        t.ins[2] = try mi.input(&t.rig, t.v[0..shape.n]);
        t.ins[3] = try mi.input(&t.rig, t.v[shape.n..]);
        t.out = try t.rig.output(m * shape.n * 4);
        return true;
    }

    pub fn run(t: *Gemm) !void {
        try md.gemm(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.out, t.shape.rows, t.shape.n, t.shape.k, t.call.lda, t.call.ldb, t.call.transposed, t.call.add);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gemm) usize {
        return t.shape.rows * t.shape.n;
    }

    fn at(t: *const Gemm, i: usize, kk: usize) f64 {
        return t.x[i * t.call.lda + kk];
    }
    fn bt(t: *const Gemm, kk: usize, j: usize) f64 {
        return if (t.call.transposed) t.w[j * t.call.ldb + kk] else t.w[kk * t.call.ldb + j];
    }

    pub fn reference(t: *Gemm, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (0..t.shape.rows) |i| for (0..n) |j| {
            var sum: f64 = 0;
            var terms: f64 = 0;
            for (0..t.shape.k) |kk| {
                const p = t.at(i, kk) * t.bt(kk, j);
                sum += p;
                terms += @abs(p);
            }
            const o = i * n + j;
            want[o] = sum + t.v[j] + (if (t.call.add) t.v[n + o] else 0);
            mag[o] = terms;
        };
    }

    pub fn simulate(t: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const n = t.shape.n;
        const terms = try a.alloc(f64, t.shape.k);
        defer a.free(terms);
        for (0..t.shape.rows) |i| for (0..n) |j| {
            for (terms, 0..) |*e, kk| e.* = t.at(i, kk) * t.bt(kk, j);
            var v: f32 = @floatCast(mi.rows.sumIn(acc, order, terms));
            v += t.v[j];
            if (t.call.add) v += t.v[n + i * n + j];
            out[i * n + j] = v;
        };
    }
};

/// The decoder's kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: md.Kernels,
    t: [2]Gemm = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), ctl: *cuda.Module) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try md.Kernels.load(gpu.d);
        errdefer k.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        var with = k;
        with.gemm = try ctl.function("tf_ctl_md_gemm_bias");
        set.t[0] = .{ .rig = rig, .k = k, .name = "cuda moss image decoder tf_md_gemm" };
        set.t[1] = .{ .rig = rig, .k = with, .name = "control: moss image decoder tf_md_gemm drops its bias", .control = .shapes };
        for (&set.t) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.t) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};
