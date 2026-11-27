//! ERNIE 5.5's mHC pre side (ernie5_5/mhc.cu) under the contract: its normed rows, post and comb, each output alone.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_mhc_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const bfr = core.bf16_round;
const mh = ernie.mhc;

const d = mh.width;
const kw = mh.wide;

/// The released config's rms_norm_eps, hc_eps and hc_sinkhorn_iters; scales of the order the checkpoint's are.
const params: mh.Params = .{ .rms_eps = 1e-6, .hc_eps = 1e-6, .norm_eps = 1e-6, .iters = 20 };
const scale = [3]f32{ 0.7, 1.3, 2.1 };

/// y: the normed rows (bf16); post and comb: the mixes' fp32 outputs.
pub const Which = enum { y, post, comb };

/// A planted fault (control entries): the plain mix RMS, or comb's softmax over a column.
pub const Fault = enum { none, plain_rms, column_softmax };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, which: Which, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre (normed rows)", .symbol = "tf_ernie55_mhc_pre", .which = .y },
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre (post)", .symbol = "tf_ernie55_mhc_pre", .which = .post },
    .{ .name = "control: ernie5_5 mhc pre with the plain mix RMS, 1 / sqrt(mean + eps)", .symbol = "tf_ctl_ernie55_mhc_pre_plain_rms", .which = .post, .fault = .plain_rms },
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_mhc_pre (comb, Sinkhorn)", .symbol = "tf_ernie55_mhc_pre", .which = .comb },
    .{ .name = "control: ernie5_5 mhc pre with comb's softmax over a column", .symbol = "tf_ctl_ernie55_mhc_pre_column_softmax", .which = .comb, .fault = .column_softmax },
};

/// A thread's 64 terms in one fma chain, the butterfly (5), the 8 warps (7); the normed rows' squares are shorter.
fn depth(_: c.Shape) usize {
    return 64 + 5 + 7;
}

const y_why = "pre is 1/2 and the collapse exact under stress, but fp16 squares overflow and bf16's move the output ulps";
const mix_why = "terms are integers at 2^-8: fp16 may hold a partial, so bf16's miss is the one shown";

fn width(w: Which) usize {
    return switch (w) {
        .y => d,
        .post => mh.streams,
        .comb => mh.streams * mh.streams,
    };
}

/// rows (gates 1 and 4 and the controls take the first: its rows 2 and 5 are tiny), a decode row, a prompt chunk.
fn served(w: Which) []const c.Shape {
    return switch (w) {
        .y => &[_]c.Shape{ .{ .rows = 6, .n = d, .k = kw }, .{ .rows = 1, .n = d, .k = kw }, .{ .rows = 64, .n = d, .k = kw } },
        .post => &[_]c.Shape{ .{ .rows = 6, .n = 4, .k = kw }, .{ .rows = 1, .n = 4, .k = kw }, .{ .rows = 64, .n = 4, .k = kw } },
        .comb => &[_]c.Shape{ .{ .rows = 6, .n = 16, .k = kw }, .{ .rows = 1, .n = 16, .k = kw }, .{ .rows = 64, .n = 16, .k = kw } },
    };
}

pub fn spec(e: Entry) c.Spec {
    const all = served(e.which);
    const sh = if (e.fault == .none) all else all[0..1];
    return switch (e.which) {
        .y => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 16, .mids = 1, .out = .bf16, .served = sh, .sums = .{ .f16 = 0, .bf16 = 0, .why = y_why } },
        .post => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 16, .out = .f32, .served = sh, .sums = .{ .f16 = 0, .why = mix_why } },
        .comb => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 256, .out = .f32, .served = sh, .sums = .{ .f16 = 0, .why = mix_why } },
    };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// Sums in an accumulator and order: gate 4's simulation.
const Sim = struct {
    acc: c.Accum,
    order: g.Order,
    pub fn of(s: Sim, t: []const f64) f64 {
        return rows_.sumIn(s.acc, s.order, t);
    }
};

pub const Pre = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    p: mh.Params = params,
    x: []u16 = &.{}, // rows of 4 bf16 streams
    fnw: []f32 = &.{},
    t: []f64 = &.{}, // scratch: a row's terms, then the row as f64
    base: [mh.mixes]f32 = undefined,
    w: [d]f64 = undefined, // the norm's weight, bf16 values
    dev: [5]u64 = @splat(0), // x, fn, scale, base, w
    outs: [3]u64 = @splat(0), // post, comb, y

    const vt = g.hooks(Pre);

    pub fn kernel(self: *Pre) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Pre) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.fnw);
        a.free(self.t);
        self.x = &.{};
        self.fnw = &.{};
        self.t = &.{};
    }

    pub fn deinit(self: *Pre) void {
        self.free();
        self.rig.deinit();
    }

    /// Rows near unit RMS, offset by 8, and below 1e-6; stress: pre 1/2 on stream 0 alone, every sum exact in fp32.
    pub fn prepare(self: *Pre, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != kw or shape.n != width(self.entry.which)) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        self.p = params;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, shape.rows * kw);
        self.fnw = try a.alloc(f32, mh.mixes * kw);
        self.t = try a.alloc(f64, 2 * kw);
        var w16: [d]u16 = undefined;
        for (&w16, &self.w) |*v, *fw| {
            v.* = bfr.bits(0.5 + rng.float(f32));
            fw.* = bf(v.*);
        }
        if (fill == .stress) {
            self.p.hc_eps = 0;
            for (self.fnw, 0..) |*f, i| f.* = if (i < mh.streams * kw) 0 else @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 5))) - 2)) * 0x1p-8;
            for (&self.base, 0..) |*b, i| b.* = if (i < mh.streams) 0 else (rng.float(f32) * 2 - 1) * 0.5;
            for (0..shape.rows) |r| for (self.x[r * kw ..][0..kw], 0..) |*v, i| {
                v.* = if (i < d) bfr.bits(2 * (@as(f32, @floatFromInt(rng.uintLessThan(u32, 63))) - 31)) else 0;
            };
        } else {
            for (self.fnw) |*f| f.* = (rng.float(f32) * 2 - 1) * 0.02;
            for (&self.base) |*b| b.* = rng.float(f32) * 2 - 1;
            for (0..shape.rows) |r| {
                const m: f32 = if (r % 3 == 2) 1e-6 else 1;
                const off: f32 = if (r % 3 == 1) 8 else 0;
                for (self.x[r * kw ..][0..kw], 0..) |*v, i| v.* = bfr.bits(m * (off + (rng.float(f32) * 2 - 1) * (if (i % 512 == 77) @as(f32, 20) else 1)));
            }
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, 42, std.math.nan(f32) }, .{ 1, d + 7, std.math.inf(f32) }, .{ 2, kw - 1, -std.math.inf(f32) } };
            for (planted) |pl| if (pl[0] < shape.rows) {
                self.x[pl[0] * kw + pl[1]] = bfr.bits(pl[2]);
            };
        }
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), try self.rig.input(std.mem.sliceAsBytes(self.fnw)), try self.rig.input(std.mem.sliceAsBytes(&scale)), try self.rig.input(std.mem.sliceAsBytes(&self.base)), try self.rig.input(std.mem.sliceAsBytes(&w16)) };
        const sizes = [3]usize{ shape.rows * 4 * 4, shape.rows * 16 * 4, shape.rows * d * 2 };
        for (&self.outs, sizes, [3]Which{ .post, .comb, .y }) |*o, n, w| o.* = if (w == self.entry.which) try self.rig.output(n) else try self.rig.scratch(n);
        return true;
    }

    pub fn run(self: *Pre) !void {
        try ernie.launch(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.outs[0], self.outs[1], self.outs[2], .{ .rows = self.shape.rows }, self.p);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Pre) usize {
        return self.shape.rows * width(self.entry.which);
    }

    /// Row r's 4 streams as f64, in the upper half of the scratch.
    fn row(self: *Pre, r: usize) *[kw]f64 {
        const x = self.t[kw..][0..kw];
        for (x, self.x[r * kw ..][0..kw]) |*v, h| v.* = bf(h);
        return x;
    }

    /// mhc.pre in float64; mag: the mixes' |terms| through sigmoid', softmax and the scale, and the collapse's terms.
    pub fn reference(self: *Pre, want: []f64, mag: []f64) !void {
        const n = width(self.entry.which);
        for (0..self.shape.rows) |r| {
            const x = self.row(r);
            var out: mh.Row = undefined;
            mh.pre(x, self.fnw, &scale, &self.base, &self.w, self.p, mh.exact, self.t[0..kw], &out);
            var ss: f64 = 0;
            for (x) |v| ss += v * v;
            const rms = mh.mixRms(ss, self.p.rms_eps);
            var mass: [mh.mixes]f64 = undefined;
            var mix: [mh.mixes]f64 = undefined;
            for (&mass, &mix, 0..) |*ms, *mx, i| {
                var abs: f64 = 0;
                var sum: f64 = 0;
                for (x, self.fnw[i * kw ..][0..kw]) |v, f| {
                    abs += @abs(v * f);
                    sum += v * f;
                }
                ms.* = abs * rms;
                mx.* = sum * rms;
            }
            const wo = want[r * n ..][0..n];
            const mo = mag[r * n ..][0..n];
            switch (self.entry.which) {
                .post => for (0..4) |k| {
                    wo[k] = out.post[k];
                    mo[k] = 0.5 * scale[1] * 1.5 * mass[4 + k] + @abs(wo[k]);
                },
                .comb => {
                    var most: f64 = 0;
                    for (mass[8..]) |v| most = @max(most, v);
                    for (0..16) |i| {
                        wo[i] = out.comb[i];
                        mo[i] = @abs(wo[i]) * 2 * scale[2] * 1.5 * most;
                    }
                },
                .y => {
                    var pw: [4]f64 = undefined;
                    var dp: [4]f64 = undefined;
                    for (0..4) |k| {
                        pw[k] = mh.sigmoid(mix[k] * scale[0] + self.base[k]) + self.p.hc_eps;
                        dp[k] = 0.25 * scale[0] * 1.5 * mass[k];
                    }
                    var s2: f64 = 0;
                    for (0..d) |h| {
                        var v: f64 = 0;
                        for (0..4) |s| v += pw[s] * x[s * d + h];
                        s2 += v * v;
                    }
                    const rn = 1 / @sqrt(s2 / d + self.p.norm_eps);
                    for (wo, mo, out.out, self.w, 0..) |*o, *mg, y, gw, h| {
                        var om: f64 = 0;
                        for (0..4) |s| om += @abs(pw[s] * x[s * d + h]) + @abs(x[s * d + h]) * dp[s];
                        o.* = y;
                        mg.* = gw * rn * om + @abs(y) / 2;
                    }
                },
            }
        }
    }

    /// mhc.pre with every sum in `acc`, in `order`.
    pub fn simulate(self: *Pre, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = width(self.entry.which);
        const sim: Sim = .{ .acc = acc, .order = order };
        for (0..self.shape.rows) |r| {
            const x = self.row(r);
            var o: mh.Row = undefined;
            mh.pre(x, self.fnw, &scale, &self.base, &self.w, self.p, sim, self.t[0..kw], &o);
            const dst = out[r * n ..][0..n];
            switch (self.entry.which) {
                .post => @memcpy(dst, &o.post),
                .comb => @memcpy(dst, &o.comb),
                .y => @memcpy(dst, &o.out),
            }
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mhc: ernie.Mhc,
    items: [entries.len]Pre = undefined,

    /// The set, its kernels appended to `list` (each control after the output it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m = try ernie.Mhc.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .mhc = m };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.mhc.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.mhc.unload();
        set.gpa.destroy(set);
    }
};

/// The mHC image's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6d686370, &t);
    std.debug.print("SUMMARY ernie5_5 mhc: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-mhc contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the kernel, and the first has tiny rows for the mix RMS's control" {
    for ([_]Which{ .y, .post, .comb }) |w| for (served(w)) |x| try mh.check(.{ .rows = x.rows }, params);
    try std.testing.expect(served(.post)[0].rows >= 3);
}
