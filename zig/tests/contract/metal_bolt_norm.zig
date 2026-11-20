//! Bolt's norm rows under the contract: the gated group norm over the Mamba inner state, the residual add with RMS.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const bolt = @import("metal_bolt.zig");

const c = contract.spec;
const g = contract.gates;
const Mut = bolt.Mut;
const Range = bolt.Range;
const bf16of = bolt.bf16of;
const build = bolt.build;
const under = bolt.under;
const controls = bolt.controls;

/// A Nemotron kernel's source by its key in kernels/metal/nemotron/kernels.zig `all`, as the engine compiles it.
fn nemotronSource(key: []const u8) []const u8 {
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return k.source;
    unreachable;
}

pub const Norm = enum { group16, group128, add, add_moe };

fn normDims(v: Norm) struct { width: usize, group: usize, threads: usize, mp: usize } {
    return switch (v) {
        .group16 => .{ .width = 8192, .group = 1024, .threads = 256, .mp = 16 },
        .group128 => .{ .width = 8192, .group = 1024, .threads = 256, .mp = 128 },
        .add, .add_moe => .{ .width = 4096, .group = 4096, .threads = 512, .mp = 16 },
    };
}

/// A thread's fma chain over its values, the simd_sum tree, then the threadgroup's simdgroup partials in order.
fn normDepth(v: Norm) usize {
    const d = normDims(v);
    return d.group / d.threads + 5 + d.threads / 32;
}
fn depthGroup(_: c.Shape) usize {
    return normDepth(.group16);
}
fn depthAdd(_: c.Shape) usize {
    return normDepth(.add);
}

const norm_why = "a bf16 row follows the sum of squares as ss^-1/2, so a bf16 sum's error is a few output ulps";

/// rows (a decode step's 1 to a prompt chunk's 16, or up to MP 128), the row width, the reduction (a norm group).
const served_group16 = [_]c.Shape{ .{ .rows = 16, .n = 8192, .k = 1024 }, .{ .rows = 1, .n = 8192, .k = 1024 } };
const served_group128 = [_]c.Shape{.{ .rows = 128, .n = 8192, .k = 1024 }};
const served_add = [_]c.Shape{ .{ .rows = 16, .n = 4096, .k = 4096 }, .{ .rows = 1, .n = 4096, .k = 4096 } };

pub fn normSpec(v: Norm) c.Spec {
    return .{
        .name = switch (v) {
            .group16 => "metal bolt group norm tf_nemotron_group_norm_xs_bolt_16",
            .group128 => "metal bolt group norm tf_nemotron_group_norm_xs_bolt_128",
            .add => "metal bolt add norm tf_nemotron_add_norm_xs_bolt",
            .add_moe => "metal bolt add norm tf_nemotron_add_norm_moe_xs_bolt",
        },
        .accum = .f32,
        .depth = if (v == .group16 or v == .group128) depthGroup else depthAdd,
        .ops = 8, // the mean's division, + eps, rsqrt, the two products
        .mids = if (v == .group16 or v == .group128) 1 else 0, // bf16(v * scale) before the weight
        .out = .bf16,
        .served = switch (v) {
            .group16 => &served_group16,
            .group128 => &served_group128,
            .add, .add_moe => &served_add,
        },
        .sums = .{ .bf16 = 0, .why = norm_why },
    };
}

pub const NormProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Norm,
    mut: ?Mut = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    eps: f32 = 1e-5,
    h: []u16 = &.{}, // group: X (the gated input); add: H
    x: []u16 = &.{}, // add: the delta; add_moe: Y (one pick)
    sh: []u16 = &.{}, // add_moe: the shared expert's output
    w: []u16 = &.{}, // the norm weight, one row
    hv: []f32 = &.{}, // what the kernel normalizes: X itself, or bf16(H + delta) as HN must hold it
    refs: [5]rig_.Ref = undefined,
    hn: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    xs: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Norm, mut: ?Mut) !NormProbe {
        const name = switch (v) {
            .group16 => "tf_nemotron_group_norm_xs_bolt_16",
            .group128 => "tf_nemotron_group_norm_xs_bolt_128",
            .add => "tf_nemotron_add_norm_xs_bolt",
            .add_moe => "tf_nemotron_add_norm_moe_xs_bolt",
        };
        const src = switch (v) {
            .group16, .group128 => nemotronSource("group_norm_xs_bolt_16"),
            .add => nemotronSource("add_norm_xs_bolt"),
            .add_moe => nemotronSource("add_norm_moe_xs_bolt"),
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .mut = mut, .pipe = try build(a, device, src, name, mut) };
    }

    fn free(t: *NormProbe) void {
        for ([_][]u16{ t.h, t.x, t.sh, t.w }) |b| t.a.free(b);
        t.a.free(t.hv);
        t.h = &.{};
        t.x = &.{};
        t.sh = &.{};
        t.w = &.{};
        t.hv = &.{};
    }

    pub fn deinit(t: *NormProbe) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *NormProbe) g.Kernel {
        return .{ .spec = under(normSpec(t.v), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(NormProbe);

    fn group(t: *const NormProbe) bool {
        return t.v == .group16 or t.v == .group128;
    }

    pub fn prepare(t: *NormProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const d = normDims(t.v);
        if (shape.n != d.width or shape.k != d.group or shape.rows == 0 or shape.rows > d.mp) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.rows * d.width;
        t.h = try t.a.alloc(u16, n);
        t.x = try t.a.alloc(u16, n);
        t.sh = try t.a.alloc(u16, n);
        t.w = try t.a.alloc(u16, d.width);
        t.hv = try t.a.alloc(f32, n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6e6f);
        const rng = prng.random();
        for (t.h) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) / 4);
        for (t.sh) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) / 4);
        for (t.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // row 0 a NaN, row 1 +Inf, row 2 -Inf, row 3 both (in the group norm, all in group 0 of the row)
                const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 7 }, .{ 2, 0 }, .{ 3, 1 }, .{ 3, 2 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, val| if (p[0] < shape.rows) {
                    t.h[p[0] * d.width + p[1]] = c.toBf16(val);
                };
            },
            .stress => {
                // integers whose squares sum below 2^24 in any order (exact in fp32); fp16 overflows on the way
                const most: u32 = if (t.group()) 90 else 63;
                for (t.h) |*v| v.* = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, most)));
                @memset(t.x, 0);
                @memset(t.sh, 0);
                for (t.w) |*v| v.* = c.toBf16(1);
            },
        }
        // what the kernel normalizes: the group norm its input; the add norms bf16(H + delta) in f32, as HN holds it
        for (t.hv, 0..) |*o, i| {
            const hval = c.fromBf16(t.h[i]);
            o.* = switch (t.v) {
                .group16, .group128 => hval,
                .add => bf16of(hval + c.fromBf16(t.x[i])),
                .add_moe => bf16of(hval + bf16of(c.fromBf16(t.x[i]) + c.fromBf16(t.sh[i]))),
            };
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.h));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.sh));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(&[2]i32{ @intCast(shape.rows), @intCast(d.mp) }));
        t.out = try t.rig.output(n * 2);
        if (!t.group()) t.hn = try t.rig.scratch(n * 2);
        t.xs = try t.rig.scratch(d.width / 64 * d.mp * 4);
        return true;
    }

    pub fn run(t: *NormProbe) !void {
        const d = normDims(t.v);
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        const eps = t.eps;
        switch (t.v) {
            .group16, .group128 => {
                e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
                e.setBuffer(t.refs[3].buf, t.refs[3].off, 1);
                e.setValue(eps, 2);
                e.setBuffer(t.refs[4].buf, t.refs[4].off, 3);
                e.setBuffer(t.out.buf, t.out.off, 4);
                e.setBuffer(t.xs.buf, t.xs.off, 5);
                // as layers.groupNorm launches it: a threadgroup per group and row, rows padded to MP
                e.dispatchGroups(mtl.Size.of(d.width / d.group, d.mp, 1), mtl.Size.of(d.threads, 1, 1));
            },
            .add => {
                e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
                e.setBuffer(t.refs[1].buf, t.refs[1].off, 1);
                e.setBuffer(t.refs[3].buf, t.refs[3].off, 2);
                e.setValue(eps, 3);
                e.setBuffer(t.refs[4].buf, t.refs[4].off, 4);
                e.setBuffer(t.hn.buf, t.hn.off, 5);
                e.setBuffer(t.out.buf, t.out.off, 6);
                e.setBuffer(t.xs.buf, t.xs.off, 7);
                e.dispatchGroups(mtl.Size.of(d.mp, 1, 1), mtl.Size.of(d.threads, 1, 1));
            },
            .add_moe => {
                const ones: [16]f32 = @splat(1);
                e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
                e.setBuffer(t.refs[1].buf, t.refs[1].off, 1);
                e.setBytes(std.mem.sliceAsBytes(&ones), 2); // one pick of weight 1 a row (bolt.moe's s.ones)
                e.setBuffer(t.refs[2].buf, t.refs[2].off, 3);
                e.setBuffer(t.refs[3].buf, t.refs[3].off, 4);
                e.setValue(eps, 5);
                e.setBuffer(t.refs[4].buf, t.refs[4].off, 6);
                e.setBuffer(t.hn.buf, t.hn.off, 7);
                e.setBuffer(t.out.buf, t.out.off, 8);
                e.setBuffer(t.xs.buf, t.xs.off, 9);
                e.dispatchGroups(mtl.Size.of(d.mp, 1, 1), mtl.Size.of(d.threads, 1, 1));
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *NormProbe) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Group (or row) g of row r: its first column and its float64 sum of squares and their magnitude.
    fn squares(t: *const NormProbe, r: usize, col: usize) struct { at: usize, ss: f64 } {
        const d = normDims(t.v);
        const first = col / d.group * d.group;
        var ss: f64 = 0;
        for (t.hv[r * d.width + first ..][0..d.group]) |v| ss += @as(f64, v) * v;
        return .{ .at = first, .ss = ss };
    }

    pub fn reference(t: *NormProbe, want: []f64, mag: []f64) !void {
        const d = normDims(t.v);
        for (0..t.shape.rows) |r| {
            var col: usize = 0;
            while (col < d.width) : (col += d.group) {
                const q = t.squares(r, col);
                const scale = 1 / @sqrt(q.ss / @as(f64, @floatFromInt(d.group)) + t.eps);
                for (col..col + d.group) |j| {
                    const i = r * d.width + j;
                    want[i] = c.fromBf16(t.w[j]) * (@as(f64, t.hv[i]) * scale);
                    mag[i] = @abs(want[i]);
                }
            }
        }
    }

    pub fn intervals(t: *NormProbe, low: []f64, high: []f64) !void {
        const d = normDims(t.v);
        const u = c.unit(.f32);
        for (0..t.shape.rows) |r| {
            var col: usize = 0;
            while (col < d.width) : (col += d.group) {
                const q = t.squares(r, col);
                // the fp32 sum of squares within gamma(depth) of float64, then / group, + eps, rsqrt (8 ulps)
                const err = c.gamma(@floatFromInt(normDepth(t.v)), u) * q.ss;
                const gw: f64 = @floatFromInt(d.group);
                const lo_ss = @max(0, q.ss - err);
                const hi_ss = q.ss + err;
                const s = Range{ .lo = 1 / @sqrt(hi_ss / gw + t.eps), .hi = 1 / @sqrt(lo_ss / gw + t.eps) };
                const scale = s.widen(c.gamma(10, u));
                for (col..col + d.group) |j| {
                    const i = r * d.width + j;
                    const wv: f64 = c.fromBf16(t.w[j]);
                    const v = Range{ .lo = t.hv[i], .hi = t.hv[i] };
                    const out = if (t.group())
                        // bf16(W * bf16(v * scale)): the product, its bf16 point, the weight's product, the output
                        v.times(scale).widen(u).bf16().scale(wv).widen(u).bf16()
                    else
                        // bf16(W * (v * scale)): two fp32 products, then the output
                        v.times(scale).widen(u).scale(wv).widen(u).bf16();
                    low[i] = out.lo;
                    high[i] = out.hi;
                }
            }
        }
    }

    pub fn bounds(t: *NormProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        const ranges = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(ranges);
        const low = ranges[0..t.count()];
        const high = ranges[t.count()..];
        try t.intervals(low, high);
        for (allow, want, low, high) |*v, w, lo, hi| v.* = @max(@abs(w - lo), @abs(hi - w));
    }

    pub fn simulate(t: *NormProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const d = normDims(t.v);
        const sq = try t.a.alloc(f64, d.group);
        defer t.a.free(sq);
        for (0..t.shape.rows) |r| {
            var col: usize = 0;
            while (col < d.width) : (col += d.group) {
                for (sq, 0..) |*o, j| {
                    const v: f64 = t.hv[r * d.width + col + j];
                    o.* = v * v;
                }
                const ss = contract.rows.sumIn(acc, order, sq);
                const scale = 1 / @sqrt(ss / @as(f64, @floatFromInt(d.group)) + t.eps);
                for (col..col + d.group) |j| {
                    const i = r * d.width + j;
                    out[i] = c.fromBf16(t.w[j]) * (@as(f64, t.hv[i]) * scale);
                }
            }
        }
    }

    /// OUT against float64; HN bit for bit; XS as the in-order f32 sum of the kernel's own bf16 outputs; padding zero.
    pub fn check(t: *NormProbe, at: g.Gate) !?[]const u8 {
        const d = normDims(t.v);
        const out = t.rig.bytes(t.out, t.count() * 2);
        {
            const n = t.count();
            const buf = try t.a.alloc(f64, 4 * n);
            defer t.a.free(buf);
            const want = buf[0..n];
            const mag = buf[n .. 2 * n];
            const low = buf[2 * n .. 3 * n];
            const high = buf[3 * n ..];
            try t.reference(want, mag);
            if (at == .nonfinite) {
                for (want, 0..) |w, i| {
                    const got = c.read(.bf16, out, i);
                    if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&t.msg, "OUT[{d}] = {d} is not of float64's class ({d})", .{ i, got, w });
                }
            } else {
                try t.intervals(low, high);
                for (low, high, 0..) |lo, hi, i| {
                    const got = c.read(.bf16, out, i);
                    if (!(got >= lo and got <= hi)) return try std.fmt.bufPrint(&t.msg, "OUT[{d}] = {d} outside float64's [{d}, {d}] through its bf16 points", .{ i, got, lo, hi });
                }
            }
        }
        if (!t.group()) {
            const hn = t.rig.bytes(t.hn, t.count() * 2);
            for (t.hv, 0..) |want, i| {
                const got = c.fromBf16(std.mem.readInt(u16, hn[i * 2 ..][0..2], .little));
                const same = got == want or (std.math.isNan(got) and std.math.isNan(want));
                if (!same) return try std.fmt.bufPrint(&t.msg, "HN[{d}] = {d}, bf16(H + delta) = {d}", .{ i, got, want });
            }
        }
        const xs = t.rig.bytes(t.xs, d.width / 64 * d.mp * 4);
        for (0..d.width / 64) |gi| for (0..d.mp) |r| {
            const got: f32 = @bitCast(std.mem.readInt(u32, xs[(gi * d.mp + r) * 4 ..][0..4], .little));
            var want: f32 = 0;
            if (r < t.shape.rows) for (0..64) |j| {
                want += c.fromBf16(std.mem.readInt(u16, out[(r * d.width + gi * 64 + j) * 2 ..][0..2], .little));
            };
            const same = @as(u32, @bitCast(got)) == @as(u32, @bitCast(want)) or (std.math.isNan(got) and std.math.isNan(want));
            if (!same) return try std.fmt.bufPrint(&t.msg, "XS group {d} row {d} = {d}, the f32 sum of its 64 outputs = {d}", .{ gi, r, got, want });
        };
        return null;
    }
};
