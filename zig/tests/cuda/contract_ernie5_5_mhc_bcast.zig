//! ERNIE 5.5's first-layer mHC pre (ernie5_5/mhc.cu's broadcast) under the contract: its 4 outputs each alone.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_mhc_cuda");
const rig_ = @import("contract_rig.zig");
const mc = @import("contract_ernie5_5_mhc.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const mh = ernie.mhc;

const d = mh.width;
const kw = mh.wide;

/// y, post and comb as the pre side's; streams: e copied into the 4 streams (bf16, exact).
pub const Which = enum { y, post, comb, streams };

/// A planted fault (control entries): the mix RMS's mean over pre's 4 streams, or e written into stream 0 alone.
pub const Fault = enum { none, wide_mean, one_stream };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, which: Which, fault: Fault = .none };

pub const bcast_entries = [_]Entry{
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre_broadcast (normed rows)", .symbol = "tf_ernie55_mhc_pre_broadcast", .which = .y },
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre_broadcast (post)", .symbol = "tf_ernie55_mhc_pre_broadcast", .which = .post },
    .{ .name = "control: ernie5_5 mhc broadcast with the mix RMS's mean over 4 x 4096 values", .symbol = "tf_ctl_ernie55_mhc_pre_broadcast_wide_mean", .which = .post, .fault = .wide_mean },
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre_broadcast (comb, Sinkhorn)", .symbol = "tf_ernie55_mhc_pre_broadcast", .which = .comb },
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre_broadcast (e into the 4 streams)", .symbol = "tf_ernie55_mhc_pre_broadcast", .which = .streams },
    .{ .name = "control: ernie5_5 mhc broadcast writing stream 0 alone", .symbol = "tf_ctl_ernie55_mhc_pre_broadcast_one_stream", .which = .streams, .fault = .one_stream },
};

/// A thread's 16 terms in one fma chain, the butterfly (5), the 8 warps (7); the normed rows' squares are shorter.
fn depth(_: c.Shape) usize {
    return 16 + 5 + 7;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

const mix_why = "terms are multiples of 2^-9: fp16 may hold a partial, so bf16's miss is the one shown";

fn width(w: Which) usize {
    return switch (w) {
        .y => d,
        .post => mh.streams,
        .comb => mh.streams * mh.streams,
        .streams => kw,
    };
}

/// rows (gates 1 and 4 and the controls take the first: its rows 2 and 5 are tiny), a decode row, a prompt chunk.
pub fn served(w: Which) []const c.Shape {
    return switch (w) {
        .y => &[_]c.Shape{ .{ .rows = 6, .n = d, .k = d }, .{ .rows = 1, .n = d, .k = d }, .{ .rows = 64, .n = d, .k = d } },
        .post => &[_]c.Shape{ .{ .rows = 6, .n = 4, .k = d }, .{ .rows = 1, .n = 4, .k = d }, .{ .rows = 64, .n = 4, .k = d } },
        .comb => &[_]c.Shape{ .{ .rows = 6, .n = 16, .k = d }, .{ .rows = 1, .n = 16, .k = d }, .{ .rows = 64, .n = 16, .k = d } },
        .streams => &[_]c.Shape{ .{ .rows = 6, .n = kw, .k = d }, .{ .rows = 1, .n = kw, .k = d }, .{ .rows = 64, .n = kw, .k = d } },
    };
}

pub fn spec(e: Entry) c.Spec {
    const all = served(e.which);
    const sh = if (e.fault == .none) all else all[0..1];
    return switch (e.which) {
        .y => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 16, .mids = 1, .out = .bf16, .served = sh, .sums = .{ .f16 = 0, .bf16 = 0, .why = mc.y_why } },
        .post => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 16, .out = .f32, .served = sh, .sums = .{ .f16 = 0, .why = mix_why } },
        .comb => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 256, .out = .f32, .served = sh, .sums = .{ .f16 = 0, .why = mix_why } },
        .streams => .{ .name = e.name, .accum = .none, .depth = noSums, .out = .bf16, .served = sh },
    };
}

pub const Bcast = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    p: mh.Params = mc.params,
    e: []u16 = &.{}, // bf16 rows
    fnb: []f32 = &.{}, // [24][4096]
    ef: []f64 = &.{}, // a row as f64
    t: []f64 = &.{}, // scratch, then the row's 4 copies
    base: [mh.mixes]f32 = undefined,
    w: [d]f64 = undefined, // the norm's weight, bf16 values
    dev: [5]u64 = @splat(0), // e, fnb, scale, base, w
    outs: [4]u64 = @splat(0), // streams, post, comb, y

    const vt = g.hooks(Bcast);

    pub fn kernel(self: *Bcast) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Bcast) void {
        const a = self.rig.gpu.gpa;
        a.free(self.e);
        a.free(self.fnb);
        a.free(self.ef);
        a.free(self.t);
        self.e = &.{};
        self.fnb = &.{};
        self.ef = &.{};
        self.t = &.{};
    }

    pub fn deinit(self: *Bcast) void {
        self.free();
        self.rig.deinit();
    }

    /// Rows near unit RMS, offset by 8, and below 1e-6; stress: pre 1/2, e half-integers, every sum exact in fp32.
    pub fn prepare(self: *Bcast, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != d or shape.n != width(self.entry.which)) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        self.p = mc.params;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.e = try a.alloc(u16, shape.rows * d);
        self.fnb = try a.alloc(f32, mh.mixes * d);
        self.ef = try a.alloc(f64, d);
        self.t = try a.alloc(f64, 2 * kw);
        var w16: [d]u16 = undefined;
        for (&w16, &self.w) |*v, *fw| {
            v.* = bfr.bits(0.5 + rng.float(f32));
            fw.* = mc.bf(v.*);
        }
        if (fill == .stress) {
            self.p.hc_eps = 0;
            for (self.fnb, 0..) |*f, i| f.* = if (i < mh.streams * d) 0 else @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 5))) - 2)) * 0x1p-8;
            for (&self.base, 0..) |*b, i| b.* = if (i < mh.streams) 0 else (rng.float(f32) * 2 - 1) * 0.5;
            for (self.e) |*v| v.* = bfr.bits((@as(f32, @floatFromInt(rng.uintLessThan(u32, 63))) - 31) / 2);
        } else {
            for (self.fnb) |*f| f.* = (rng.float(f32) * 2 - 1) * 0.04;
            for (&self.base) |*b| b.* = rng.float(f32) * 2 - 1;
            for (0..shape.rows) |r| {
                const m: f32 = if (r % 3 == 2) 1e-6 else 1;
                const off: f32 = if (r % 3 == 1) 8 else 0;
                for (self.e[r * d ..][0..d], 0..) |*v, i| v.* = bfr.bits(m * (off + (rng.float(f32) * 2 - 1) * (if (i % 512 == 77) @as(f32, 20) else 1)));
            }
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, 42, std.math.nan(f32) }, .{ 1, 7, std.math.inf(f32) }, .{ 2, d - 1, -std.math.inf(f32) } };
            for (planted) |pl| if (pl[0] < shape.rows) {
                self.e[pl[0] * d + pl[1]] = bfr.bits(pl[2]);
            };
        }
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.e)), try self.rig.input(std.mem.sliceAsBytes(self.fnb)), try self.rig.input(std.mem.sliceAsBytes(&mc.scale)), try self.rig.input(std.mem.sliceAsBytes(&self.base)), try self.rig.input(std.mem.sliceAsBytes(&w16)) };
        const sizes = [4]usize{ shape.rows * kw * 2, shape.rows * 4 * 4, shape.rows * 16 * 4, shape.rows * d * 2 };
        for (&self.outs, sizes, [4]Which{ .streams, .post, .comb, .y }) |*o, n, w| o.* = if (w == self.entry.which) try self.rig.output(n) else try self.rig.scratch(n);
        return true;
    }

    pub fn run(self: *Bcast) !void {
        try ernie.launchBcast(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.outs[0], self.outs[1], self.outs[2], self.outs[3], .{ .rows = self.shape.rows }, self.p);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Bcast) usize {
        return self.shape.rows * width(self.entry.which);
    }

    /// Row r as f64.
    fn row(self: *Bcast, r: usize) *[d]f64 {
        const e = self.ef[0..d];
        for (e, self.e[r * d ..][0..d]) |*v, h| v.* = mc.bf(h);
        return e;
    }

    /// mhc.preBroadcast in float64 (streams: e's copies); mag: as the pre side's, from fnb's |terms| over e.
    pub fn reference(self: *Bcast, want: []f64, mag: []f64) !void {
        const n = width(self.entry.which);
        for (0..self.shape.rows) |r| {
            const e = self.row(r);
            const wo = want[r * n ..][0..n];
            const mo = mag[r * n ..][0..n];
            if (self.entry.which == .streams) {
                for (0..mh.streams) |s| for (wo[s * d ..][0..d], mo[s * d ..][0..d], e) |*o, *mg, v| {
                    o.* = v;
                    mg.* = @abs(v);
                };
                continue;
            }
            const x = self.t[kw..][0..kw];
            var out: mh.Row = undefined;
            mh.preBroadcast(e, self.fnb, &mc.scale, &self.base, &self.w, self.p, mh.exact, x, self.t[0..kw], &out);
            var ss: f64 = 0;
            for (e) |v| ss += v * v;
            const rms = 1 / (@sqrt(ss / d) + self.p.rms_eps);
            var mass: [mh.mixes]f64 = undefined;
            var mix: [mh.mixes]f64 = undefined;
            for (&mass, &mix, 0..) |*ms, *mx, i| {
                var abs: f64 = 0;
                var sum: f64 = 0;
                for (e, self.fnb[i * d ..][0..d]) |v, f| {
                    abs += @abs(v * f);
                    sum += v * f;
                }
                ms.* = abs * rms;
                mx.* = sum * rms;
            }
            const which: mc.Which = switch (self.entry.which) {
                .y => .y,
                .post => .post,
                .comb => .comb,
                .streams => unreachable,
            };
            mc.rowMags(which, x, &out, &mix, &mass, &self.base, &self.w, self.p, wo, mo);
        }
    }

    /// mhc.preBroadcast with every sum in `acc`, in `order` (the streams have none: their copies).
    pub fn simulate(self: *Bcast, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = width(self.entry.which);
        const sim: mc.Sim = .{ .acc = acc, .order = order };
        for (0..self.shape.rows) |r| {
            const e = self.row(r);
            const dst = out[r * n ..][0..n];
            if (self.entry.which == .streams) {
                for (0..mh.streams) |s| @memcpy(dst[s * d ..][0..d], e);
                continue;
            }
            var o: mh.Row = undefined;
            mh.preBroadcast(e, self.fnb, &mc.scale, &self.base, &self.w, self.p, sim, self.t[kw..][0..kw], self.t[0..kw], &o);
            switch (self.entry.which) {
                .post => @memcpy(dst, &o.post),
                .comb => @memcpy(dst, &o.comb),
                .y => @memcpy(dst, &o.out),
                .streams => unreachable,
            }
        }
    }
};
