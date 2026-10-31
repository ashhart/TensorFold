//! Flash Next's HC mix under the contract on HIP: its float64 model, the dense tile's HC mix epilogue, its gate table.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const Kernels = @FieldType(hip.Launcher, "affine");
/// launches/affine.zig's Arg, as denseGo and denseHcMixGo take it.
const Arg = @typeInfo(@TypeOf(Kernels.denseGo)).@"fn".param_types[2].?;
const C = ?*const anyopaque;
const P = ?*anyopaque;

/// sigmoid and silu as the kernels compute them, within this of float64's: expf within 3 ulps, then two roundings.
pub const fn_err = 0x1p-20;
const nan = std.math.nan(f64);
const inf = std.math.inf(f64);

pub fn bf(v: f64) f64 {
    return kit.round(.bf16, v);
}

/// 1 / (1 + e^-v) in float64: NaN for NaN, 1 for +Inf, 0 for -Inf.
pub fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

/// tf_fn_hc_mix's gate of a g: g rounded to bf16, its sigmoid rounded to bf16.
pub fn gate(v: f64) f64 {
    return bf(sigmoid(bf(v)));
}

/// The gates the kernels can give for a g in [lo, hi]: the sigmoid within fn_err, or 2^-126 below fp32's normals.
pub fn gates(lo: f64, hi: f64) [2]f64 {
    const a = sigmoid(bf(lo)) * (1 - fn_err) - 0x1p-126;
    const b = sigmoid(bf(hi)) * (1 + fn_err) + 0x1p-126;
    return .{ bf(@max(a, 0)), bf(b) };
}

/// The mix's output before its last rounding, its streams' terms summed in `acc` in `order` (float64 for none).
pub fn mixed(gs: []const f64, xs: []const f64, acc: c.Accum, order: rows.Order) f64 {
    var t: [8]f64 = undefined;
    for (gs, xs, t[0..gs.len]) |v, x, *o| o.* = bf(gate(v) * x);
    return rows.sumIn(acc, order, t[0..gs.len]) / @as(f64, @floatFromInt(gs.len));
}

/// One output: float64's value at the kernel's rounding points, the range the device's can take, its allowance.
pub const Range = struct { want: f64, lo: f64, hi: f64, allow: f64 };

/// The largest power of two a nonzero bf16 value is a whole multiple of.
fn unitOf(v: f64) f64 {
    var p = std.math.ldexp(@as(f64, 1), std.math.frexp(v).exponent - 8);
    while (@mod(v, 2 * p) == 0) p *= 2;
    return p;
}

/// fp32 adds of these bf16 terms are exact in any order: every partial a whole count of their least unit below 2^24.
fn exactAdds(t: []const f64) bool {
    var unit: f64 = inf;
    var abs: f64 = 0;
    for (t) |v| if (v != 0) {
        unit = @min(unit, unitOf(v));
        abs += @abs(v);
    };
    return abs == 0 or abs / unit < 0x1p24;
}

/// Output (r, d) of the mix from each stream's g (the device's within `err` of it) and x, the terms' adds in fp32.
pub fn range(gs: []const f64, err: []const f64, xs: []const f64) Range {
    var want: f64 = 0;
    var lo: f64 = 0;
    var hi: f64 = 0;
    var abs: f64 = 0;
    var t: [8]f64 = undefined;
    var fixed = true;
    for (gs, err, xs, t[0..gs.len]) |v, e, x, *o| {
        o.* = bf(gate(v) * x);
        want += o.*;
        const span = gates(v - e, v + e);
        const ends = [2]f64{ bf(span[0] * x), bf(span[1] * x) };
        fixed = fixed and ends[0] == ends[1];
        lo += @min(ends[0], ends[1]);
        hi += @max(ends[0], ends[1]);
        abs += @max(@abs(ends[0]), @abs(ends[1]));
    }
    const n: f64 = @floatFromInt(gs.len);
    // n - 1 fp32 adds and the division by n, each within 2^-24 of the terms' magnitude over n
    const slack = if (fixed and exactAdds(t[0..gs.len])) 0 else 0x1p-24 * abs;
    const pre = [2]f64{ lo / n - slack, hi / n + slack };
    const out = [2]f64{ bf(pre[0]), bf(pre[1]) };
    want /= n;
    const allow = @max(@max(pre[1], out[1]) - want, want - @min(pre[0], out[0]));
    return .{ .want = want, .lo = out[0], .hi = out[1], .allow = allow + 0x1p-133 };
}

pub fn rangeBounds(ranges: []const Range, allow: []f64) void {
    for (allow, ranges) |*l, r| l.* = r.allow;
}

pub fn rangeIntervals(ranges: []const Range, lo: []f64, hi: []f64) void {
    for (lo, hi, ranges) |*a, *b, r| {
        a.* = r.lo;
        b.* = r.hi;
    }
}

/// x's plants in row `r` of [rows][ways][width]: NaN, +Inf, -Inf, and both infinities in one output's streams.
pub fn plantX(x: []f64, ways: usize, width: usize, r: usize) void {
    const at = [_][2]usize{ .{ 1, 5 }, .{ 2, 9 }, .{ 3, 11 }, .{ 0, 13 }, .{ 1, 13 } };
    const put = [_]f64{ nan, inf, -inf, inf, -inf };
    for (at, put) |p, v| x[(r * ways + p[0] % ways) * width + p[1] % width] = v;
}

/// float64 values as bf16 bytes.
pub fn bf16Bytes(a: std.mem.Allocator, v: []const f64) ![]u8 {
    const b = try a.alloc(u8, v.len * 2);
    for (v, 0..) |x, i| kit.put(.bf16, b, i, x);
    return b;
}

/// Gate 1 of a replacement: its bytes are the replaced launches', and every output's class is float64's.
pub fn classes(e: c.Elem, out: []const u8, base: []const u8, want: []const f64) ?[]const u8 {
    if (!std.mem.eql(u8, out, base)) return "non-finite output differs from the replaced launches' bytes";
    var planted: usize = 0;
    for (want, 0..) |w, i| {
        if (c.classOf(w) != .finite) planted += 1;
        if (c.classOf(c.read(e, out, i)) != c.classOf(w)) return "non-finite output class differs from float64";
    }
    if (planted == 0 or planted == want.len) return "the plants make no output, or every output, non-finite";
    return null;
}

const streams = 4;
const low = 320; // hc up's K
const bits = 4;
const group = 32;

/// A group's two WMMAs as 16 dot2 steps, two roundings each (gfx11's fp32 accumulate drops bits), then its two folds.
fn denseDepth(s: c.Shape) usize {
    return 2 * 16 + 2 * (s.k / group);
}

const both = "the separate launches' bytes and every float64 output class";

pub fn denseSpec() c.Spec {
    return .{
        .name = "hip tf_affine_dense_hcmix (hc up, 4-bit, K 320, the HC mix as its epilogue)",
        .accum = .f32,
        .depth = denseDepth,
        .out = .bf16,
        .replaces = "the dense K 320 tile's bf16 product then tf_fn_hc_mix",
        .nan = .{ .stated = both },
        .tiles = .{ .{ .size = 128 }, .{ .size = 128, .step = 128, .min = 128 }, .{} },
        // prompt rows the registry gives the dense tile, four streams of 2,560
        .served = &.{
            .{ .rows = 129, .n = streams * 2560, .k = low },
            .{ .rows = 300, .n = streams * 2560, .k = low },
            .{ .rows = 1000, .n = streams * 2560, .k = low },
        },
        .sums = .{ .exact = false },
    };
}

/// act [rows][320] times hc up's MLX 4-bit matrix [n][320], then mixed with x [rows][4][n / 4]: [rows][n / 4] bf16.
pub const Dense = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    f: ?contract.affine.Affine = null,
    x: []f64 = &.{},
    ranges: []Range = &.{},
    arg: Arg = undefined,
    xs: u64 = 0,
    out: u64 = 0,
    g16: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Dense {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a) };
    }

    pub fn deinit(self: *Dense) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Dense) g.Kernel {
        return .{ .spec = denseSpec(), .ctx = self, .vt = &dense_vt };
    }

    const dense_vt = g.hooks(Dense);

    fn dims(self: *const Dense) usize {
        return self.shape.n / streams;
    }

    /// The affine family's act and weights (gate 4: its integer sums, g in [-56, -8]); x in [-2, 2), 1 at gate 4.
    pub fn prepare(self: *Dense, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != low or shape.n == 0 or shape.n % 128 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        const f = try contract.affine.Affine.init(a, shape, bits, group);
        try f.fill(fill, seed);
        self.f = f;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6863);
        self.x = (try kit.values(a, prng.random(), .bf16, shape.rows * shape.n, -2, 2))[0];
        // the terms are then the gates: positive, so their fp32 adds stay within 2^-24 of the sum
        if (fill == .stress) @memset(self.x, 1);
        if (fill == .nonfinite and shape.rows > 4) plantX(self.x, streams, self.dims(), 4);
        const w = try a.alloc(u32, shape.n * shape.k * bits / 32);
        f.words(w);
        self.arg = .{
            .x = try self.rig.input(std.mem.sliceAsBytes(f.x)),
            .words = try self.rig.input(std.mem.sliceAsBytes(w)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 }, // bf16 tables
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = 0,
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = bits,
            .group = group,
            .fp16 = 0,
        };
        self.xs = try self.rig.input(try bf16Bytes(a, self.x));
        self.out = try self.rig.output(self.count() * 2);
        self.g16 = try self.rig.scratch(shape.rows * shape.n * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    pub fn run(self: *Dense) !void {
        try self.l.affine.denseHcMixGo(self.l.r, self.arg, self.xs, self.out, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The dense K 320 tile's product rounded to bf16, then tf_fn_hc_mix over it.
    pub fn replaced(self: *Dense) ![]const u8 {
        var plain = self.arg;
        plain.out16 = self.g16;
        const st = self.rig.s.handle;
        try self.l.affine.denseGo(self.l.r, plain, st);
        const m: c_int = @intCast(self.shape.rows);
        const d: c_int = @intCast(self.dims());
        try self.l.fn_hc_mix_scalar(kit.ptr(C, self.g16), 2, kit.ptr(C, self.xs), kit.ptr(P, self.base), 2, m, streams, d, st);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn check(self: *Dense, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        return classes(.bf16, out, try self.replaced(), want);
    }

    pub fn count(self: *Dense) usize {
        return self.shape.rows * self.dims();
    }

    fn mixRows(self: *Dense, gv: []const f64, want: []f64, mag: []f64, err: []const f64) void {
        const d = self.dims();
        for (0..self.shape.rows) |r| for (0..d) |j| {
            var gs: [streams]f64 = undefined;
            var es: [streams]f64 = undefined;
            var xs: [streams]f64 = undefined;
            for (0..streams) |s| {
                gs[s] = gv[r * self.shape.n + s * d + j];
                es[s] = err[r * self.shape.n + s * d + j];
                xs[s] = self.x[(r * streams + s) * d + j];
            }
            const i = r * d + j;
            want[i] = mixed(&gs, &xs, .none, .sequential);
            mag[i] = @abs(want[i]);
            self.ranges[i] = range(&gs, &es, &xs);
        };
    }

    /// The float64 product at MLX's factored formula, then the mix at its bf16 points; g within gamma(depth) of it.
    pub fn reference(self: *Dense, want: []f64, mag: []f64) !void {
        const f = self.f.?;
        const a = self.arena.allocator();
        const gv = try a.alloc(f64, f.outputs());
        const err = try a.alloc(f64, f.outputs());
        const gam = c.gamma(@floatFromInt(denseDepth(self.shape)), c.unit(.f32));
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            const o = f.one(r, j);
            gv[r * self.shape.n + j] = o[0];
            err[r * self.shape.n + j] = gam * o[1];
        };
        self.ranges = try a.alloc(Range, self.count());
        self.mixRows(gv, want, mag, err);
    }

    pub fn bounds(self: *Dense, _: []const f64, _: []const f64, allow: []f64) !void {
        rangeBounds(self.ranges, allow);
    }

    pub fn intervals(self: *Dense, lo: []f64, hi: []f64) !void {
        rangeIntervals(self.ranges, lo, hi);
    }

    pub fn simulate(self: *Dense, acc: c.Accum, order: g.Order, out: []f64) !void {
        const f = self.f.?;
        const a = self.arena.allocator();
        const gv = try a.alloc(f64, f.outputs());
        try f.simulate(acc, order, gv);
        const d = self.dims();
        for (0..self.shape.rows) |r| for (0..d) |j| {
            var gs: [streams]f64 = undefined;
            var xs: [streams]f64 = undefined;
            for (0..streams) |s| {
                gs[s] = gv[r * self.shape.n + s * d + j];
                xs[s] = self.x[(r * streams + s) * d + j];
            }
            out[r * d + j] = mixed(&gs, &xs, acc, order);
        };
    }
};

/// tf_hc_gate_fill's table in the tiles' code object: rnd(sigmoid(g)) for every bf16 g, as tf_affine_dense_hcmix reads.
const table_symbol = "_ZN2tf4rocm10tf_hc_gateE";
const entries = 1 << 16;
const table_rule = "every bf16 g: NaN for NaN, else float64's 1 / (1 + e^-g) rounded to bf16 within expf's 2^-20 " ++
    "(2^-126 below fp32's normals); and tf_fn_hc_mix's gate of g, bit for bit";

fn none(_: c.Shape) usize {
    return 0;
}

pub fn tableSpec() c.Spec {
    return .{
        .name = "hip tf_hc_gate_fill (tf_affine_dense_hcmix's 65,536 gates)",
        .accum = .none,
        .depth = none,
        .out = .raw,
        .nan = .{ .stated = table_rule },
        .served = &.{.{ .rows = 1, .n = entries, .k = 0 }},
    };
}

/// The table filled as the launcher fills it, each entry NaN before, copied out of the code object's global.
pub const Table = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    global: u64 = 0,
    gs: u64 = 0,
    ones: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Table {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a) };
    }

    pub fn deinit(self: *Table) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Table) g.Kernel {
        return .{ .spec = tableSpec(), .ctx = self, .vt = &table_vt };
    }

    const table_vt = g.hooks(Table);

    /// Every gate's input is its own index; no fill changes them. Also every bf16 g and x = 1 for tf_fn_hc_mix.
    pub fn prepare(self: *Table, _: c.Shape, _: g.Fill, _: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const t = try self.l.mods[@backingInt(hip.kernels.Group.affine_tiles)].global(table_symbol);
        if (t.len != entries * 2) return error.Invalid;
        self.global = t.address;
        const ids = try a.alloc(u16, entries);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        const ones = try a.alloc(u16, entries);
        @memset(ones, 0x3f80);
        self.gs = try self.rig.input(std.mem.sliceAsBytes(ids));
        self.ones = try self.rig.input(std.mem.sliceAsBytes(ones));
        self.out = try self.rig.output(entries * 2);
        self.base = try self.rig.scratch(entries * 2);
        return true;
    }

    pub fn run(self: *Table) !void {
        const api = self.rig.r.api;
        try self.rig.s.synchronize();
        // every entry NaN first: an entry the fill skips cannot pass as the value the load left there
        try hip.runtime.check(api.hipMemset(@ptrFromInt(self.global), 0xff, entries * 2));
        try self.l.affine.fillHcGate(self.l.r);
        const st = self.rig.s.handle;
        try hip.runtime.check(api.hipMemcpyDtoDAsync(@ptrFromInt(self.out), @ptrFromInt(self.global), entries * 2, st));
        try self.rig.s.synchronize();
    }

    pub fn count(_: *Table) usize {
        return entries * 2;
    }

    pub fn reference(_: *Table, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each entry by the stated rule, and against tf_fn_hc_mix's gate (one stream, x = 1: its output is the gate).
    pub fn check(self: *Table, _: c.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const got = try a.dupe(u8, try self.rig.bytes(self.out, entries * 2));
        const st = self.rig.s.handle;
        try self.l.fn_hc_mix_scalar(kit.ptr(C, self.gs), 2, kit.ptr(C, self.ones), kit.ptr(P, self.base), 2, 1, 1, entries, st);
        const mix = try self.rig.bytes(self.base, entries * 2);
        for (0..entries) |h| {
            const v: f64 = c.fromBf16(@intCast(h));
            const out = c.read(.bf16, got, h);
            const span = gates(v, v);
            const ok = if (std.math.isNan(v)) std.math.isNan(out) else out >= span[0] and out <= span[1];
            if (!ok) return try std.fmt.bufPrint(&self.msg, "g {e} holds {e}, the rule gives [{e}, {e}]", .{ v, out, span[0], span[1] });
            if (!std.mem.eql(u8, got[2 * h ..][0..2], mix[2 * h ..][0..2])) {
                return try std.fmt.bufPrint(&self.msg, "g {e} holds {e}, tf_fn_hc_mix's gate is {e}", .{ v, out, c.read(.bf16, mix, h) });
            }
        }
        return null;
    }
};
