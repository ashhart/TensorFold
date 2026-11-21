//! The decoder's MLP glue under the contract: ud_silu_mul (SwiGLU's bf16 steps) and ud_combine (experts' weighted sum).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const H = k.hidden;
const Mutation = k.Mutation;
const Range = k.Range;
const bf = k.bf;

// --- ud_silu_mul: Y = bf16(bf16(silu(g)) * u), silu in fp32 --------------------------------------------------------

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// silu's fp32 steps: exp (4 ulps), 1 + e, the division (2 ulps).
fn siluRange(x: f64) Range {
    if (!std.math.isFinite(x)) return Range.point(silu(x));
    const e = Range.point(@exp(-x)).fp32(4);
    const den = (Range{ .lo = 1 + e.lo, .hi = 1 + e.hi }).fp32(1);
    const q: Range = if (x >= 0) .{ .lo = x / den.hi, .hi = x / den.lo } else .{ .lo = x / den.lo, .hi = x / den.hi };
    return q.fp32(2);
}

pub const Silu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: tokens or expert slots; n: the width. Gate and up rows sit side by side, 2n apart (the joint product).
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_silu_mul",
            .accum = .none,
            .depth = k.noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            // the shared expert's and the dense layer's prompt rows, a chunk's expert slots, a step's 6 slots and rows
            .served = &.{ .{ .rows = k.prompt_rows, .n = k.shared_ffn, .k = 1 }, .{ .rows = k.prompt_rows, .n = k.dense_ffn, .k = 1 }, .{ .rows = 6 * k.prompt_rows, .n = k.expert_ffn, .k = 1 }, .{ .rows = 6, .n = k.expert_ffn, .k = 1 }, .{ .rows = 1, .n = k.dense_ffn, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Silu {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_silu_mul"), .m = m };
    }
    pub fn deinit(t: *Silu) void {
        t.a.free(t.gu);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Silu) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Silu);

    pub fn prepare(t: *Silu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gu);
        t.shape = shape;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.gu = try t.a.alloc(u16, shape.rows * 2 * n);
        // gates deep into silu's tails on both sides; nonzero, so an infinite up never meets a zero
        for (t.gu) |*v| v.* = c.toBf16(@max(0x1p-6, rng.float(f32) * 24) * (if (rng.boolean()) @as(f32, 1) else -1));
        if (fill == .nonfinite) for (0..@min(shape.rows, 6)) |r| {
            const at = r * 2 * n + (if (r < 3) 0 else n) + (r * 7) % n;
            t.gu[at] = c.toBf16(k.planted[r % 3]);
        };
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.gu));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(t: *Silu) !void {
        const n: u32 = @intCast(t.shape.n);
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.in.buf, t.in.off + n * 2, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(extern struct { width: u32, gstride: u32, ustride: u32 }{ .width = n, .gstride = 2 * n, .ustride = 2 * n }, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Silu) usize {
        return t.shape.rows * t.shape.n;
    }

    fn parts(t: *Silu, i: usize) [2]f64 {
        const n = t.shape.n;
        const at = (i / n) * 2 * n + i % n;
        return .{ c.fromBf16(t.gu[at]), c.fromBf16(t.gu[at + n]) };
    }

    pub fn reference(t: *Silu, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const p = t.parts(i);
            w.* = bf(bf(silu(p[0])) * p[1]);
            m.* = 0;
        }
    }

    pub fn intervals(t: *Silu, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const p = t.parts(i);
            const r = siluRange(p[0]).bf16().mul(Range.point(p[1])).fp32(1).bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Silu, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};

// --- ud_combine: Y = bf16(bf16(sum_j D[slot[r, j]] w[r, j]) + S), the sum in fp32 -----------------------------------

/// The products and adds of 6 terms, one after another.
fn combineDepth(_: c.Shape) usize {
    return 2 * k.topk;
}

pub const Combine = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    d: []u16 = &.{},
    slot: []u32 = &.{},
    wts: []f32 = &.{},
    s: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: tokens; n: the width (1,280); k: the picks a row (6).
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_combine top 6 plus the shared expert",
            .accum = .f32,
            .depth = combineDepth,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = k.prompt_rows, .n = H, .k = k.topk }, .{ .rows = 1, .n = H, .k = k.topk } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Combine {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_combine"), .m = m };
    }
    pub fn deinit(t: *Combine) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Combine) void {
        inline for (.{ "d", "slot", "wts", "s" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Combine);

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k != k.topk) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const slots = shape.rows * k.topk;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.d = try t.a.alloc(u16, slots * H);
        t.slot = try t.a.alloc(u32, slots);
        t.wts = try t.a.alloc(f32, slots);
        t.s = try t.a.alloc(u16, shape.rows * H);
        // the slots in a shuffled order, as experts group a prompt's rows
        for (t.slot, 0..) |*v, i| v.* = @intCast(i);
        rng.shuffle(u32, t.slot);
        if (fill == .stress) {
            // per output: 2^16 or 2^17, an integer below 128, the large term's negative: exact only in fp32
            for (0..shape.rows) |r| for (0..H) |j| {
                const big: f32 = if (rng.boolean()) 65536 else 131072;
                const vals = [_]f32{ big, @floatFromInt(1 + rng.uintLessThan(u32, 127)), -big, 0, 0, 0 };
                for (vals, 0..) |v, q| t.d[@as(usize, t.slot[r * k.topk + q]) * H + j] = c.toBf16(v);
            };
            @memset(t.wts, 1);
            @memset(t.s, 0);
        } else {
            k.fillBf16(rng, t.d, 4);
            for (t.wts) |*v| v.* = 0.02 + rng.float(f32) * 0.2;
            k.fillBf16(rng, t.s, 2);
        }
        if (fill == .nonfinite) {
            t.d[@as(usize, t.slot[0]) * H + 1] = c.toBf16(k.planted[0]);
            if (shape.rows > 2) {
                t.wts[1 * k.topk + 2] = k.planted[1];
                t.s[2 * H + 3] = c.toBf16(k.planted[2]);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.d));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.slot));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.s));
        t.out = try t.rig.output(shape.rows * H * 2);
        return true;
    }

    pub fn run(t: *Combine) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 4);
        p.e.dispatchThreads(mtl.Size.of(H, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Combine) usize {
        return t.shape.rows * H;
    }

    fn terms(t: *Combine, i: usize, out: *[k.topk]f64) void {
        const r = i / H;
        for (out, 0..) |*o, q| o.* = @as(f64, c.fromBf16(t.d[@as(usize, t.slot[r * k.topk + q]) * H + i % H])) * t.wts[r * k.topk + q];
    }

    fn outputs(t: *Combine, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        var tm: [k.topk]f64 = undefined;
        for (out, 0..) |*o, i| {
            t.terms(i, &tm);
            var s: f64 = 0;
            var m: f64 = 0;
            for (tm) |v| {
                s += v;
                m += @abs(v);
            }
            if (acc != .none) s = contract.rows.sumIn(acc, order, &tm);
            o.* = bf(bf(s) + c.fromBf16(t.s[i]));
            if (mag) |ms| ms[i] = if (std.math.isFinite(m)) m else 0;
        }
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, out, null);
    }

    /// The fp32 sum within gamma(12) of its terms (exact at gate 4), its bf16 point, the shared add and its point.
    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        var tm: [k.topk]f64 = undefined;
        for (low, high, 0..) |*l, *h, i| {
            t.terms(i, &tm);
            var s: f64 = 0;
            var m: f64 = 0;
            for (tm) |v| {
                s += v;
                m += @abs(v);
            }
            const e = if (t.fill == .stress) 0 else c.gamma(2 * k.topk, c.unit(.f32)) * m;
            const r = Range.point(s).widen(e).bf16().add(Range.point(c.fromBf16(t.s[i]))).fp32(1).bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    silus: std.ArrayList(*Silu) = .empty,
    combines: std.ArrayList(*Combine) = .empty,

    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), only: ?[]const u8, m: Mutation) !void {
        const x = try s.a.create(T);
        x.* = try T.init(s.a, s.device, s.queue, m);
        const kern = x.kernel();
        if (!k.wanted(only, kern.spec.name)) {
            x.deinit();
            s.a.destroy(x);
            return;
        }
        try list.append(s.a, x);
        try s.kernels.append(s.a, kern);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .silu_swapped }) |m| try s.add(Silu, &s.silus, only, m);
        for ([_]Mutation{ .none, .combine_no_shared }) |m| try s.add(Combine, &s.combines, only, m);
    }

    pub fn deinit(s: *Suite) void {
        inline for (.{ "silus", "combines" }) |f| {
            for (@field(s, f).items) |x| {
                x.deinit();
                s.a.destroy(x);
            }
            @field(s, f).deinit(s.a);
        }
    }
};

/// Gate 5's texts for this group's launches.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?k.Texts {
    return k.limitsText(name, &.{ "ud_silu_mul", "ud_combine" });
}
