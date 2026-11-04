//! fl_prep under the contract: q and k rows normed (MLX's rms_norm), rotated in fp32 (sliding layers), k and v cached.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");

const c = contract.spec;
const g = contract.gates;
const bf = fl.bf;
const Range = fl.Range;
const u = fl.u32_ulp;
const D = fl.head_dim;
const H = fl.heads;
const KVH = fl.kv_heads;
/// A row of the qkv projection: q, then k, then v.
const xrow = (H + 2 * KVH) * D;
const kvw = KVH * D;

/// glue.metal's fl_prep_args.
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, dim: u32, pos: u32, ring: u32, rope: u32, stride: u32, eps: f32, theta: f32 };

/// A head's squares: four fma steps a lane, then the simd_sum tree.
fn depth(_: c.Shape) usize {
    return D / 32 + 5;
}

const Case = struct { sliding: bool, fault: bool };

/// An accumulator and order gate 4 replays the sums in.
const Acc = struct { a: c.Accum, o: g.Order };

pub const Prep = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Case,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    pos: usize = 0,
    slots: usize = 0,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    qw: []u16 = &.{},
    kw: []u16 = &.{},
    shown: std.ArrayList(u8) = .empty,
    ins: [3]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    fn base(sliding: bool) []const u8 {
        return if (sliding) "metal flint glue fl_prep sliding (RoPE, ring)" else "metal flint glue fl_prep full (NoPE)";
    }
    const drop: fl.Edit = .{ .kernel = "fl_prep", .from = "float(bfloat(v[j] * inv))", .to = "(v[j] * inv)" };

    /// Rows: the call's rows; n: the head dim; k: the first row's position.
    pub fn spec(name: []const u8, sliding: bool) c.Spec {
        const sl = [_]c.Shape{ .{ .rows = fl.window_rows, .n = D, .k = 262000 }, .{ .rows = fl.prompt_rows, .n = D, .k = 0 }, .{ .rows = 1, .n = D, .k = fl.ring - 1 }, .{ .rows = fl.batch_rows, .n = D, .k = 5000 } };
        const full = [_]c.Shape{ .{ .rows = fl.window_rows, .n = D, .k = 40000 }, .{ .rows = fl.prompt_rows, .n = D, .k = 0 }, .{ .rows = 1, .n = D, .k = 3 } };
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} },
            .served = if (sliding) &sl else &full,
            .sums = .{ .bf16 = 0, .why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Prep {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, "fl_prep", if (v.fault) drop else null), .v = v };
    }
    pub fn deinit(t: *Prep) void {
        t.free();
        t.shown.deinit(t.a);
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Prep) void {
        inline for (.{ "x", "qw", "kw" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Prep) g.Kernel {
        t.name = fl.named(base(t.v.sliding), if (t.v.fault) "the norm's x * inv bf16 rounding dropped" else null, &t.label);
        return .{ .spec = spec(t.name, t.v.sliding), .ctx = t, .vt = &vt, .control = if (t.v.fault) .shapes else null };
    }
    const vt = blk: {
        var h = g.hooks(Prep);
        h.output = output;
        break :blk h;
    };

    fn slotOf(t: *const Prep, row: usize) usize {
        return if (t.v.sliding) (t.pos + row) % fl.ring else t.pos + row;
    }

    pub fn prepare(t: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != D) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.pos = shape.k;
        t.fill = fill;
        t.slots = if (t.v.sliding) fl.ring else t.pos + t.rows;
        t.x = try t.a.alloc(u16, t.rows * xrow);
        t.qw = try t.a.alloc(u16, D);
        t.kw = try t.a.alloc(u16, D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 (128 x 1,600 at most) and overflow fp16 in any order
            for (t.x) |*v| v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            @memset(t.qw, c.toBf16(1));
            @memset(t.kw, c.toBf16(1));
        } else {
            for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
            for (t.qw) |*v| v.* = c.toBf16(rng.float(f32) * 2);
            for (t.kw) |*v| v.* = c.toBf16(rng.float(f32) * 2);
        }
        if (fill == .nonfinite) {
            // NaN in row 0's q head 0; +Inf in row 1's k head 1; -Inf in row 2's v head 2; both in row 3's q head 5
            const at = [_]usize{ 3, xrow + H * D + 1 * D + 7, 2 * xrow + (H + KVH) * D + 2 * D + 9, 3 * xrow + 5 * D + 1, 3 * xrow + 5 * D + 70 };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[2], fl.planted[1], fl.planted[2] };
            for (at, vals) |i, v| if (i < t.x.len) {
                t.x[i] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.qw));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.kw));
        t.outs[0] = try t.rig.output(t.rows * H * D * 2);
        t.outs[1] = try t.rig.output(t.slots * kvw * 2);
        t.outs[2] = try t.rig.output(t.slots * kvw * 2);
        return true;
    }

    pub fn run(t: *Prep) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(PArgs{ .rows = @intCast(t.rows), .heads = H, .kv_heads = KVH, .dim = D, .pos = @intCast(t.pos), .ring = if (t.v.sliding) @intCast(fl.ring) else 0, .rope = @intFromBool(t.v.sliding), .stride = xrow, .eps = fl.eps, .theta = fl.theta }, 3);
        for (t.outs, 4..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchGroups(mtl.Size.of(H + KVH, t.rows, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    /// The written values only: q's rows, then each row's k slot, then each row's v slot.
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const t: *Prep = @ptrCast(@alignCast(ctx));
        t.shown.clearRetainingCapacity();
        try t.shown.appendSlice(t.a, t.rig.bytes(t.outs[0], t.rows * H * D * 2));
        for (1..3) |o| for (0..t.rows) |r| try t.shown.appendSlice(t.a, t.rig.bytes(t.outs[o], t.slots * kvw * 2)[t.slotOf(r) * kvw * 2 ..][0 .. kvw * 2]);
        return t.shown.items;
    }

    pub fn count(t: *Prep) usize {
        return t.rows * (H * D + 2 * kvw);
    }

    /// Output i's head: q or k (normed, maybe rotated), or v (a copy), with its row, its source and its dim.
    const Where = struct { kind: enum { q, k, v }, row: usize, src: usize, d: usize };
    fn where(t: *const Prep, i: usize) Where {
        const qn = t.rows * H * D;
        if (i < qn) return .{ .kind = .q, .row = i / (H * D), .src = (i / (H * D)) * xrow + (i % (H * D)) / D * D, .d = i % D };
        const j = i - qn;
        const per = t.rows * kvw;
        const r = (j % per) / kvw;
        const h = (j % kvw) / D;
        const isv = j >= per;
        return .{ .kind = if (isv) .v else .k, .row = r, .src = r * xrow + @as(usize, if (isv) H + KVH else H) * D + h * D, .d = j % D };
    }

    /// A head's normed values in float64 at MLX's points, from its sum of squares `ss`.
    fn normed(t: *const Prep, w: Where, ss: f64, out: *[D]f64) void {
        const wt = if (w.kind == .q) t.qw else t.kw;
        const inv = 1 / @sqrt(ss / D + @as(f64, fl.eps));
        for (out, 0..) |*o, d| o.* = bf(@as(f64, c.fromBf16(wt[d])) * bf(@as(f64, c.fromBf16(t.x[w.src + d])) * inv));
    }

    fn squares(t: *const Prep, w: Where, acc: ?Acc) f64 {
        var sq: [D]f64 = undefined;
        var ss: f64 = 0;
        for (&sq, 0..) |*q, d| {
            const x: f64 = c.fromBf16(t.x[w.src + d]);
            q.* = x * x;
            ss += q.*;
        }
        return if (acc) |s| contract.rows.sumIn(s.a, s.o, &sq) else ss;
    }

    /// The rotation's angle for dim d at position p: p theta^(-i / half) for i = d mod half.
    fn angle(p: usize, d: usize) f64 {
        const i: f64 = @floatFromInt(d % (D / 2));
        return @as(f64, @floatFromInt(p)) * std.math.pow(f64, fl.theta, -i / (D / 2));
    }

    fn rotated(t: *const Prep, w: Where, nrm: *const [D]f64) f64 {
        if (w.kind == .v) return c.fromBf16(t.x[w.src + w.d]);
        if (!t.v.sliding) return nrm[w.d];
        const half = D / 2;
        const a = angle(t.pos + w.row, w.d);
        const re = nrm[w.d % half];
        const im = nrm[w.d % half + half];
        return bf(if (w.d < half) re * @cos(a) - im * @sin(a) else re * @sin(a) + im * @cos(a));
    }

    /// Every head's outputs, D at a time (a head's outputs are contiguous): its float64 values at MLX's points.
    fn values(t: *const Prep, out: []f64, acc: ?Acc) void {
        var nrm: [D]f64 = undefined;
        var b: usize = 0;
        while (b < out.len) : (b += D) {
            var w = t.where(b);
            if (w.kind != .v) t.normed(w, t.squares(w, acc), &nrm);
            for (out[b..][0..D], 0..) |*o, d| {
                w.d = d;
                o.* = t.rotated(w, &nrm);
            }
        }
    }

    pub fn reference(t: *Prep, want: []f64, mag: []f64) !void {
        t.values(want, null);
        @memset(mag, 0);
    }

    pub fn simulate(t: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.values(out, .{ .a = acc, .o = order });
    }

    /// A normed value's range: squares within gamma(depth) (gate 4: exact), 4 roundings to inv, x * inv, bf16, w, bf16.
    fn normRange(t: *const Prep, w: Where, ss: f64, d: usize) Range {
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(depth(.{ .rows = 0, .n = 0, .k = 0 })), u);
        const iv = Range{ .lo = 1 / @sqrt(ss * (1 + gam) / D + @as(f64, fl.eps)) * (1 - 4 * u), .hi = 1 / @sqrt(ss * (1 - gam) / D + @as(f64, fl.eps)) * (1 + 4 * u) };
        const wt = if (w.kind == .q) t.qw else t.kw;
        return iv.times(c.fromBf16(t.x[w.src + d])).fp32().bf16().times(c.fromBf16(wt[d])).bf16();
    }

    /// cos and sin over the angle's spread: fp32's exponent (3 roundings), exp2 and the product.
    fn trig(p: usize, d: usize) [2]Range {
        const i = d % (D / 2);
        const a = angle(p, d);
        const x = @as(f64, @floatFromInt(i)) / (D / 2) * @log2(@as(f64, fl.theta));
        const delta = if (i == 0) 0 else @abs(a) * (3 * x + 3) * u;
        const cs = @cos(a);
        const sn = @sin(a);
        const sq = delta * delta / 2 + 4 * u;
        return .{ Range.point(cs).widen(@abs(sn) * delta + sq), Range.point(sn).widen(@abs(cs) * delta + sq) };
    }

    pub fn intervals(t: *Prep, low: []f64, high: []f64) !void {
        const half = D / 2;
        var nr: [D]Range = undefined;
        var b: usize = 0;
        while (b < low.len) : (b += D) {
            var w = t.where(b);
            if (w.kind != .v) {
                const ss = t.squares(w, null);
                for (&nr, 0..) |*x, d| x.* = t.normRange(w, ss, d);
            }
            for (low[b..][0..D], high[b..][0..D], 0..) |*l, *h, d| {
                w.d = d;
                var r: Range = undefined;
                if (w.kind == .v) {
                    r = Range.point(c.fromBf16(t.x[w.src + d]));
                } else if (!t.v.sliding) {
                    r = nr[d];
                } else {
                    const re = nr[d % half];
                    const im = nr[d % half + half];
                    const cs = trig(t.pos + w.row, d);
                    const neg = Range{ .lo = -im.hi, .hi = -im.lo };
                    r = if (d < half) re.mul(cs[0]).fp32().plus(neg.mul(cs[1]).fp32()).fp32().bf16() else re.mul(cs[1]).fp32().plus(im.mul(cs[0]).fp32()).fp32().bf16();
                }
                l.* = r.lo;
                h.* = r.hi;
            }
        }
    }

    pub fn bounds(t: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tests: fl.List(Prep) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Case{ .{ .sliding = true, .fault = false }, .{ .sliding = false, .fault = false }, .{ .sliding = true, .fault = true } }) |v| {
            if (fl.wanted(only, Prep.base(v.sliding))) try s.tests.add(s.a, s.kernels, try Prep.init(s.a, s.device, s.queue, v));
        }
    }
    pub fn deinit(s: *Suite) void {
        s.tests.deinit(s.a);
    }
};

/// fl_prep is glue.metal's, which metal_flint_glue.zig compiles for gate 5.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !?fl.Texts {
    return null;
}
