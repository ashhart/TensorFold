//! md_wino and md_wino_32 under the contract: Winograd F(2x2, 3x3) against the float64 convolution.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");
const st = @import("metal_moss_image_decoder_steps.zig");

const c = mi.c;
const g = mi.g;
const dec = tf.moss_image.decoder;
const ref = st.ref;
const uniform = st.uniform;
const plant = st.plant;

/// B^T, G and A^T of F(2x2, 3x3).
const BT = [4][4]f64{ .{ 1, 0, -1, 0 }, .{ 0, 1, 1, 0 }, .{ 0, -1, 1, 0 }, .{ 0, 1, 0, -1 } };
const G = [4][3]f64{ .{ 1, 0, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.5, -0.5, 0.5 }, .{ 0, 0, 1 } };
const AT = [2][4]f64{ .{ 1, 1, 1, 0 }, .{ 0, 1, -1, -1 } };

/// A 3x3 form the decoder launches through one instance: its group width, read from half size or not, a residual.
pub const Form = struct { label: []const u8, group: usize, up: bool = false, add: bool = false, served: []const c.Shape };

/// The pictures here are 8 x 18 (the decoder's up to 1024 x 1024): the sums' depth is the input channels alone.
pub const forms = [_]Form{
    .{ .label = "md_wino 3x3", .group = 64, .served = &.{ .{ .rows = 8, .n = 384, .k = 384 }, .{ .rows = 8, .n = 384, .k = 32 }, .{ .rows = 8, .n = 192, .k = 384 }, .{ .rows = 8, .n = 192, .k = 192 } } },
    .{ .label = "md_wino 3x3 upsampled", .group = 64, .up = true, .served = &.{ .{ .rows = 8, .n = 384, .k = 384 }, .{ .rows = 8, .n = 192, .k = 192 } } },
    .{ .label = "md_wino 3x3 + residual", .group = 64, .add = true, .served = &.{ .{ .rows = 8, .n = 384, .k = 384 }, .{ .rows = 8, .n = 192, .k = 192 } } },
    .{ .label = "md_wino_32 3x3", .group = 32, .served = &.{ .{ .rows = 8, .n = 96, .k = 192 }, .{ .rows = 8, .n = 96, .k = 96 }, .{ .rows = 8, .n = 3, .k = 96 } } },
    .{ .label = "md_wino_32 3x3 + residual", .group = 32, .add = true, .served = &.{.{ .rows = 8, .n = 96, .k = 96 }} },
};

const cols = 18;

pub const Wino = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: Form,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    /// the weights as released [cout, cin, 9], and their transform as the decoder packs it [16, cin, ldw]
    w: []f32 = &.{},
    u: []f32 = &.{},
    b: []f32 = &.{},
    r: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// The transforms' roundings (2 in B^T d B, 1 in G g G^T), cin products, A^T M A's 4, the bias and the residual.
    fn depth(s: c.Shape) usize {
        return s.k + 3 + 4 + 2;
    }

    /// Rows: the output picture's rows (18 columns); n: output channels; k: input channels.
    pub fn spec(name: []const u8, f: Form) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{ .size = 4, .step = 2, .min = 2 }, .{ .size = f.group }, .{ .size = 16, .step = 16, .min = 16 } },
            .served = f.served,
        };
    }

    pub fn deinit(t: *Wino) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Wino) void {
        inline for (.{ &t.x, &t.w, &t.u, &t.b, &t.r }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn kernel(t: *Wino, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Wino);

    fn ldw(t: *const Wino) usize {
        return std.mem.alignForward(usize, t.shape.n, 64);
    }
    fn pixels(t: *const Wino) usize {
        return t.shape.rows * cols;
    }
    fn inCols(t: *const Wino) usize {
        return if (t.form.up) cols / 2 else cols;
    }

    pub fn prepare(t: *Wino, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows % 2 != 0 or shape.n == 0 or shape.k == 0 or shape.k % 16 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const K = shape.k;
        const L = t.ldw();
        const in_px = if (t.form.up) t.pixels() / 4 else t.pixels();
        t.x = try t.a.alloc(f32, in_px * K);
        t.w = try t.a.alloc(f32, shape.n * K * 9);
        t.u = try t.a.alloc(f32, 16 * K * L);
        t.b = try t.a.alloc(f32, shape.n);
        t.r = try t.a.alloc(f32, if (t.form.add) t.pixels() * shape.n else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // equal pixels, mirrored channel halves, weights 1 then -1: quarters throughout, sums cancel
            const half = K / 2;
            for (0..in_px) |p| for (0..K) |ci| {
                const pos = if (ci < half) ci else ci - half;
                const v: f32 = if (pos % 2 == 0) 127 else 1021;
                t.x[p * K + ci] = if (ci < half) v else v + @as(f32, if (pos < 4) 1 else 0);
            };
            for (0..shape.n) |o| for (0..K) |ci| @memset(t.w[(o * K + ci) * 9 ..][0..9], if (ci < half) 1 else -1);
            @memset(t.b, 0);
            for (t.r) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
        } else {
            uniform(rng, t.x, -2, 2);
            uniform(rng, t.w, -0.1, 0.1);
            uniform(rng, t.b, -1, 1);
            uniform(rng, t.r, -3, 3);
            if (fill == .nonfinite) plant(t.x, K);
        }
        @memset(t.u, 0);
        for (0..shape.n) |o| for (0..K) |ci| {
            for (dec.transform(t.w[(o * K + ci) * 9 ..][0..9]), 0..) |v, pt| t.u[(pt * K + ci) * L + o] = v;
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.u));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        if (t.form.add) t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.r));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Wino) !void {
        const p = t.rig.begin();
        const s: dec.ConvShape = .{ .rows = t.shape.rows, .cols = cols, .cin = t.shape.k, .cout = t.shape.n, .ldw = t.ldw(), .up = t.form.up };
        dec.winograd(p.e, t.pipe, t.form.group, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), if (t.form.add) ref(t.ins[3]) else null, ref(t.out), s);
        try t.rig.finish(p);
    }

    pub fn count(t: *Wino) usize {
        return t.pixels() * t.shape.n;
    }

    /// The input at picture position (y, x), zero in the padding.
    fn at(t: *const Wino, y: isize, x: isize, ci: usize) f64 {
        if (y < 0 or x < 0 or y >= t.shape.rows or x >= cols) return 0;
        const uy: usize = @intCast(y);
        const ux: usize = @intCast(x);
        const p = if (t.form.up) (uy / 2) * t.inCols() + ux / 2 else uy * cols + ux;
        return t.x[p * t.shape.k + ci];
    }

    /// B^T d B of tile (ty, tx)'s patch for channel ci, and the same of |d| (the magnitude its roundings act on).
    fn tileV(t: *const Wino, ty: usize, tx: usize, ci: usize, v: *[16]f64, va: *[16]f64) void {
        var d: [4][4]f64 = undefined;
        for (0..4) |i| for (0..4) |j| {
            d[i][j] = t.at(@as(isize, @intCast(ty * 2 + i)) - 1, @as(isize, @intCast(tx * 2 + j)) - 1, ci);
        };
        for (0..4) |i| for (0..4) |j| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (0..4) |a| for (0..4) |b| {
                const k = BT[i][a] * BT[j][b];
                if (k == 0) continue;
                s += k * d[a][b];
                m += @abs(k * d[a][b]);
            };
            v[i * 4 + j] = s;
            va[i * 4 + j] = m;
        };
    }

    /// G g G^T exactly, and the same of |g|.
    fn exactU(t: *const Wino, o: usize, ci: usize, u: *[16]f64, ua: *[16]f64) void {
        const w = t.w[(o * t.shape.k + ci) * 9 ..][0..9];
        for (0..4) |i| for (0..4) |j| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (0..3) |a| for (0..3) |b| {
                const k = G[i][a] * G[j][b];
                s += k * w[a * 3 + b];
                m += @abs(k * w[a * 3 + b]);
            };
            u[i * 4 + j] = s;
            ua[i * 4 + j] = m;
        };
    }

    /// Each output as A^T M A, M each point's products V U summed by `sum` (U exact or as packed); `mag` its bound's.
    fn each(t: *const Wino, packed_u: bool, sum: anytype, out: []f64, mag: ?[]f64) !void {
        const K = t.shape.k;
        const vs = try t.a.alloc([16]f64, K);
        defer t.a.free(vs);
        const vas = try t.a.alloc([16]f64, K);
        defer t.a.free(vas);
        const terms = try t.a.alloc(f64, K);
        defer t.a.free(terms);
        const us = try t.a.alloc([16]f64, K);
        defer t.a.free(us);
        const uas = try t.a.alloc([16]f64, K);
        defer t.a.free(uas);
        for (0..t.shape.rows / 2) |ty| for (0..cols / 2) |tx| {
            for (0..K) |ci| t.tileV(ty, tx, ci, &vs[ci], &vas[ci]);
            for (0..t.shape.n) |o| {
                for (0..K) |ci| t.exactU(o, ci, &us[ci], &uas[ci]);
                var m: [16]f64 = undefined;
                var ma: [16]f64 = undefined;
                for (0..16) |pt| {
                    var mm: f64 = 0;
                    for (0..K) |ci| {
                        const uv = if (packed_u) @as(f64, t.u[(pt * K + ci) * t.ldw() + o]) else us[ci][pt];
                        terms[ci] = vs[ci][pt] * uv;
                        mm += vas[ci][pt] * uas[ci][pt];
                    }
                    m[pt] = sum.of(terms);
                    ma[pt] = mm;
                }
                for (0..2) |rr| for (0..2) |cc| {
                    var y: f64 = 0;
                    var ya: f64 = 0;
                    for (0..4) |i| for (0..4) |j| {
                        const k = AT[rr][i] * AT[cc][j];
                        if (k == 0) continue;
                        y += k * m[i * 4 + j];
                        ya += @abs(k) * ma[i * 4 + j];
                    };
                    const px = (ty * 2 + rr) * cols + tx * 2 + cc;
                    const idx = px * t.shape.n + o;
                    const res: f64 = if (t.form.add) t.r[idx] else 0;
                    out[idx] = y + t.b[o] + res;
                    if (mag) |mg| {
                        const total = ya + @abs(@as(f64, t.b[o])) + @abs(res);
                        mg[idx] = if (std.math.isFinite(total)) total else 0;
                    }
                };
            }
        };
    }

    const Exact = struct {
        fn of(_: Exact, terms: []const f64) f64 {
            var s: f64 = 0;
            for (terms) |v| s += v;
            return s;
        }
    };
    const Replay = struct {
        acc: c.Accum,
        order: g.Order,
        fn of(r: Replay, terms: []const f64) f64 {
            return contract.rows.sumIn(r.acc, r.order, terms);
        }
    };

    pub fn reference(t: *Wino, want: []f64, mag: []f64) !void {
        try t.each(false, Exact{}, want, mag);
    }

    pub fn simulate(t: *Wino, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.each(true, Replay{ .acc = acc, .order = order }, out, null);
    }
};
