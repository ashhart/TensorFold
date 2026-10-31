//! id_norm, id_norm32 and id_norm_split under the contract: LayerNorm32 in fp32, at idol-sdk's fp16 points.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");

const c = id.c;
const g = id.g;
const ks = id.ks;
const cfg = id.cfg;
const H = id.H;
const Range = id.Range;
const Ends = id.Ends;
const hf = id.hf;
const kref = id.kref;

pub const Kind = enum { norm, norm32, split };

const norm_why = "an fp16 output follows the sums as var^-1/2: a bf16 sum's error is a few output ulps";

/// A thread's 4 values of the 1,024 in turn, the simdgroup tree, then the tree over the threadgroup's 8 simdgroups.
const sum_depth = 1024 / 256 + 5 + 5;

fn normDepth(_: c.Shape) usize {
    return sum_depth + 3;
}

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    kind: Kind,
    mode: ks.Norm,
    name: []const u8,
    shape: c.Shape = undefined,
    stress: bool = false,
    x: []f16 = &.{},
    /// w then b, fp32
    wb: []f32 = &.{},
    /// the modulation's 1 + scale, then its shift, fp16
    mod: []f16 = &.{},
    ins: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n and k: the width, 1,024 (the launcher's).
    pub fn spec(name: []const u8, kind: Kind, mode: ks.Norm) c.Spec {
        const all: []const c.Shape = &.{ .{ .rows = 12294, .n = H, .k = H }, .{ .rows = 8192, .n = H, .k = H }, .{ .rows = 4101, .n = H, .k = H } };
        return .{
            .name = name,
            .accum = .f32,
            .depth = normDepth,
            .ops = 2 * (id.div_ulps + id.rsqrt_ulps) + 3,
            .out = if (kind == .norm32) .f32 else .f16,
            .served = switch (kind) {
                .norm => switch (mode) {
                    .modulated => all[0..2],
                    .affine => all[2..],
                    // no block calls the plain form: checked at every stream's length
                    .plain => all,
                },
                // the output head's modulated LayerNorm; the affine form at the same rows
                .norm32 => all[1..2],
                .split => all,
            },
            .sums = if (kind == .norm) .{ .bf16 = 0, .why = norm_why } else .{},
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Norm) void {
        t.a.free(t.x);
        t.a.free(t.wb);
        t.a.free(t.mod);
        t.x = &.{};
        t.wb = &.{};
        t.mod = &.{};
    }
    pub fn kernel(t: *Norm, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.kind, t.mode), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Norm);

    fn eps(t: *const Norm) f32 {
        return if (t.kind == .norm) cfg.block_eps else cfg.default_eps;
    }

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k != H) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.stress = fill == .stress;
        t.x = try t.a.alloc(f16, shape.rows * H);
        t.wb = try t.a.alloc(f32, 2 * H);
        t.mod = try t.a.alloc(f16, 2 * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // integers mirrored in each row's halves: a mean of exactly 0 and squares summing to ~2^22, exact in fp32
            for (0..shape.rows) |r| for (0..H / 2) |i| {
                const v = id.int(rng, -64, 64);
                t.x[r * H + i] = @floatCast(v);
                t.x[r * H + H / 2 + i] = @floatCast(-v);
            };
        } else id.randomF16(rng, t.x, -3, 3);
        id.randomF32(rng, t.wb[0..H], 0.25, 2);
        id.randomF32(rng, t.wb[H..], -1, 1);
        id.randomF16(rng, t.mod[0..H], 0.5, 1.5);
        id.randomF16(rng, t.mod[H..], -1, 1);
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, H / 3 }, .{ 1, H / 2 }, .{ 2, H - 1 }, .{ 3, 0 }, .{ 3, H - 1 } };
            for (at, [_]f16{ id.nan16, id.inf16, -id.inf16, id.inf16, -id.inf16 }) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * H + p[1]] = v;
            };
            t.wb[5] = id.nan32;
            t.wb[H + 6] = id.inf32;
            t.mod[7] = -id.inf16;
            t.mod[H + 8] = id.nan16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wb[0..H]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.wb[H..]));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.mod[0..H]));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.mod[H..]));
        t.out = try t.rig.output(t.count() * @as(usize, if (t.kind == .norm32) 4 else 2));
        return true;
    }

    pub fn run(t: *Norm) !void {
        const k = &t.lib.k;
        try t.lib.need(switch (t.kind) {
            .norm => "id_norm",
            .norm32 => "id_norm32",
            .split => "id_norm_split",
        });
        const p = t.rig.begin();
        const in = t.ins;
        switch (t.kind) {
            .norm => k.norm(p.e, kref(in[0]), kref(in[1]), kref(in[2]), kref(in[3]), kref(in[4]), kref(t.out), t.shape.rows, t.eps(), t.mode),
            .norm32 => k.norm32(p.e, kref(in[0]), kref(in[1]), kref(in[2]), kref(t.out), t.shape.rows, t.eps(), t.mode),
            .split => k.normSplit(p.e, kref(in[0]), kref(in[1]), kref(in[2]), kref(t.out), t.shape.rows, t.eps()),
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * H * @as(usize, if (t.kind == .split) 3 else 1);
    }

    /// Row r's outputs from the kernel's mean (range `mr`) and sum of squared deviations (range `qr`).
    fn row(t: *Norm, ends: *Ends, r: usize, mean: f64, mr: Range, q: f64, qr: Range, want: []f64, mag: []f64) void {
        const n: f64 = H;
        const e: f64 = t.eps();
        const inv = 1 / @sqrt(q / n + e);
        const ir = (Range{ .lo = qr.lo / n, .hi = qr.hi / n }).add(Range.of(e)).fp32(0.5).rsqrt().fp32(id.rsqrt_ulps);
        for (0..H) |i| {
            const x: f64 = t.x[r * H + i];
            const v = (x - mean) * inv;
            const vr = Range.of(x).sub(mr).fp32(0.5).mul(ir).fp32(0.5);
            const w: f64 = t.wb[i];
            const b: f64 = t.wb[H + i];
            const s1: f64 = t.mod[i];
            const sh: f64 = t.mod[H + i];
            const at = r * H + i;
            switch (t.kind) {
                .norm => switch (t.mode) {
                    .plain => ends.put(want, mag, at, hf(v), vr.fp16()),
                    .affine => ends.put(want, mag, at, hf(v * w + b), vr.mul(Range.of(w)).fp32(0.5).add(Range.of(b)).fp32(0.5).fp16()),
                    .modulated => {
                        const y = vr.fp16().mul(Range.of(s1)).fp32(0.5).fp16().add(Range.of(sh)).fp32(0.5).fp16();
                        ends.put(want, mag, at, hf(hf(hf(v) * s1) + sh), y);
                    },
                },
                .norm32 => switch (t.mode) {
                    .modulated => ends.put(want, mag, at, v * (1 + w) + b, vr.mul(Range.of(1 + w).fp32(0.5)).fp32(0.5).add(Range.of(b)).fp32(0.5)),
                    else => ends.put(want, mag, at, v * w + b, vr.mul(Range.of(w)).fp32(0.5).add(Range.of(b)).fp32(0.5)),
                },
                .split => {
                    const y = v * w + b;
                    const yr = vr.mul(Range.of(w)).fp32(0.5).add(Range.of(b)).fp32(0.5);
                    const hi = hf(y);
                    const hr = yr.fp16();
                    // hi + lo: lo = fp16(y - hi), y - hi exact in fp32 for whichever fp16 hi the range rounds to
                    const lr = (Range{ .lo = yr.lo - hr.hi, .hi = yr.hi - hr.lo }).fp16();
                    const base = r * 3 * H + i;
                    ends.put(want, mag, base, hi, hr);
                    ends.put(want, mag, base + H, hi, hr);
                    ends.put(want, mag, base + 2 * H, hf(y - hi), lr);
                },
            }
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const n: f64 = H;
        const gs = c.gamma(sum_depth, c.unit(.f32));
        const gq = c.gamma(sum_depth + 3, c.unit(.f32));
        for (0..t.shape.rows) |r| {
            const xs = t.x[r * H ..][0..H];
            var s: f64 = 0;
            var abs: f64 = 0;
            for (xs) |v| {
                s += v;
                abs += @abs(@as(f64, v));
            }
            const mean = s / n;
            var q: f64 = 0;
            for (xs) |v| q += (v - mean) * (v - mean);
            if (t.stress) {
                // a mean of 0 exactly, the deviations the integers and their squares' sums exact in fp32
                t.row(&t.ends, r, mean, Range.of(mean), q, Range.of(q), want, mag);
                continue;
            }
            const mr = Range{ .lo = (s - gs * abs) / n, .hi = (s + gs * abs) / n };
            // the squares about the kernel's mean exceed float64's by n (m' - mean)^2, then fp32 rounds them
            const dm = @max(@abs(mr.lo - mean), @abs(mr.hi - mean));
            const qr = Range{ .lo = q * (1 - gq), .hi = (q + n * dm * dm) * (1 + gq) };
            t.row(&t.ends, r, mean, mr, q, qr, want, mag);
        }
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [H]f64 = undefined;
        for (0..t.shape.rows) |r| {
            const xs = t.x[r * H ..][0..H];
            for (&terms, xs) |*v, x| v.* = x;
            const mean = contract.rows.sumIn(acc, order, &terms) / H;
            for (&terms, xs) |*v, x| v.* = (x - mean) * (x - mean);
            const q = contract.rows.sumIn(acc, order, &terms);
            t.row(&scratch, r, mean, Range.of(mean), q, Range.of(q), out, mag);
        }
    }
};
