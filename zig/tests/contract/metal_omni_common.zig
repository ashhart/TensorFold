//! LFM3-Omni's Metal kernel tests' shared parts: the sources, float64 values with their fp32 spread, and the probe.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;
pub const Ref = rig_.Ref;

/// The metallib's four sources; a pipeline belongs to one.
pub const File = enum { omni, gemm, rows, nax };

/// A source's text as the metallib compiles it, its includes inlined.
pub fn text(a: std.mem.Allocator, file: File) ![]u8 {
    const s = sources.lfm3_omni;
    const common = try std.mem.replaceOwned(u8, a, s.common, "#pragma once\n", "");
    defer a.free(common);
    return switch (file) {
        .omni, .gemm, .rows => std.mem.replaceOwned(u8, a, switch (file) {
            .omni => s.omni,
            .gemm => s.gemm,
            else => s.rows,
        }, "#include \"om_common.h\"", common),
        .nax => blk: {
            const gemm = try std.mem.replaceOwned(u8, a, s.gemm_nax, "#include \"../nax.h\"", sources.nax);
            defer a.free(gemm);
            break :blk std.mem.replaceOwned(u8, a, s.nax, "#include \"../prefill/gemm_nax.metal\"", gemm);
        },
    };
}

/// The metallib's options: safe math and precise fp32 functions, the OS's Metal language.
pub fn options() mtl.CompileOptions {
    var o = mtl.CompileOptions.mlx();
    o.math_mode = .safe;
    o.math_functions = .precise;
    return o;
}

/// A source compiled with `from` replaced once by `to` (a planted control), or as served.
pub fn compile(a: std.mem.Allocator, device: mtl.Device, file: File, edit: ?[2][]const u8) !mtl.Library {
    const served = try text(a, file);
    defer a.free(served);
    const t = if (edit) |e| blk: {
        if (std.mem.count(u8, served, e[0]) != 1) {
            std.debug.print("control text not found once in {t}: {s}\n", .{ file, e[0] });
            return error.ControlTextNotFoundOnce;
        }
        break :blk try std.mem.replaceOwned(u8, a, served, e[0], e[1]);
    } else try a.dupe(u8, served);
    defer a.free(t);
    return mtl.Library.fromSource(device, t, options());
}

// Metal's error bounds for the precise functions the kernels call, in fp32 ulps.
pub const div_ulps = 2.5;
pub const rsqrt_ulps = 2;
pub const sqrt_ulps = 1;
pub const fn_ulps = 4;

fn bfv(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A float64 value at the kernel's bf16 points, and the range its fp32 steps may land in.
pub const R = struct {
    v: f64,
    lo: f64,
    hi: f64,

    pub fn of(x: f64) R {
        return .{ .v = x, .lo = x, .hi = x };
    }
    pub fn bf16(h: u16) R {
        return of(c.fromBf16(h));
    }
    /// Rounded to bf16 (nearest even), both ends with it.
    pub fn bf(x: R) R {
        return .{ .v = bfv(x.v), .lo = bfv(x.lo), .hi = bfv(x.hi) };
    }
    /// An fp32 step good to `ulps`.
    pub fn fp(x: R, ulps: f64) R {
        const u = ulps * 0x1p-23;
        const lo = if (std.math.isFinite(x.lo)) x.lo - u * @abs(x.lo) - 0x1p-149 else x.lo;
        const hi = if (std.math.isFinite(x.hi)) x.hi + u * @abs(x.hi) + 0x1p-149 else x.hi;
        return .{ .v = x.v, .lo = lo, .hi = hi };
    }
    pub fn add(x: R, y: R) R {
        return (R{ .v = x.v + y.v, .lo = x.lo + y.lo, .hi = x.hi + y.hi }).fp(0.5);
    }
    pub fn sub(x: R, y: R) R {
        return x.add(y.neg());
    }
    pub fn neg(x: R) R {
        return .{ .v = -x.v, .lo = -x.hi, .hi = -x.lo };
    }
    fn span(x: R, y: R, v: f64, p: [4]f64) R {
        _ = x;
        _ = y;
        return .{ .v = v, .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    pub fn mulExact(x: R, y: R) R {
        return span(x, y, x.v * y.v, .{ x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi });
    }
    pub fn mul(x: R, y: R) R {
        return x.mulExact(y).fp(0.5);
    }
    /// x / y with y's range on one side of zero.
    pub fn div(x: R, y: R) R {
        return span(x, y, x.v / y.v, .{ x.lo / y.lo, x.lo / y.hi, x.hi / y.lo, x.hi / y.hi }).fp(div_ulps);
    }
    /// f of an increasing function, good to `ulps`.
    pub fn up(x: R, comptime f: fn (f64) f64, ulps: f64) R {
        return (R{ .v = f(x.v), .lo = f(x.lo), .hi = f(x.hi) }).fp(ulps);
    }
    /// f with |f'| <= l over the range, good to `ulps`.
    pub fn lip(x: R, comptime f: fn (f64) f64, l: f64, ulps: f64) R {
        const v = f(x.v);
        const d = l * @max(x.v - x.lo, x.hi - x.v);
        return (R{ .v = v, .lo = v - d, .hi = v + d }).fp(ulps);
    }
    pub fn rsqrt(x: R) R {
        return (R{ .v = 1 / @sqrt(x.v), .lo = 1 / @sqrt(x.hi), .hi = 1 / @sqrt(x.lo) }).fp(rsqrt_ulps);
    }
};

pub fn exp(x: f64) f64 {
    return @exp(x);
}
pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}
pub fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// exp(-x) as fp32 gives it: past fp32's largest value it is +Inf.
fn expNeg(x: R) R {
    var e = x.neg().up(exp, fn_ulps);
    const top: f64 = std.math.floatMax(f32);
    if (e.v > top) e.v = std.math.inf(f64);
    if (e.lo > top) e.lo = std.math.inf(f64);
    if (e.hi > top) e.hi = std.math.inf(f64);
    return e;
}

/// SiLU as the kernels round it: x / (1 + exp(-x)), the exp and the division fp32 steps.
pub fn siluR(x: R) R {
    return x.div(R.of(1).add(expNeg(x)));
}

/// sigmoid as om_common.h's sigm.
pub fn sigmR(x: R) R {
    return R.of(1).div(R.of(1).add(expNeg(x)));
}

/// erf by its Taylor series (|x| < 4), else +-1 less erfc's asymptote: an independent float64 erf.
pub fn erf(x: f64) f64 {
    const ax = @abs(x);
    if (std.math.isNan(x)) return x;
    var r: f64 = 1;
    if (ax < 4) {
        var term = ax;
        var s: f64 = ax;
        var n: f64 = 0;
        while (@abs(term) > 1e-18 * @abs(s)) {
            n += 1;
            term *= -ax * ax / n;
            s += term / (2 * n + 1);
        }
        r = s * 2 / @sqrt(std.math.pi);
    } else r = 1 - @exp(-ax * ax) / (ax * @sqrt(std.math.pi));
    return if (x < 0) -r else r;
}

/// The exact GELU as om_bias_act rounds it: 0.5 v (1 + erf(v / sqrt 2)), erf_f good to 2e-7.
pub fn geluR(v: R) R {
    const z = v.mul(R.of(0.70710678118654752));
    var e = z.lip(erf, 1.1283791670955126, fn_ulps);
    e.lo -= 2e-7;
    e.hi += 2e-7;
    return R.of(0.5).mul(v).mul(R.of(1).add(e));
}

/// How a replay runs its sums: as float64 with fp32's spread, exactly (gate 4's inputs), or in an accumulator.
pub const Mode = union(enum) { ref, exact, sim: struct { acc: c.Accum, order: g.Order } };

/// A sum of `terms` (`depth` fp32 roundings deep at most) under `mode`.
pub fn sum(mode: Mode, buf: []f64, terms: []const R, depth: usize) R {
    switch (mode) {
        .sim => |s| {
            for (buf[0..terms.len], terms) |*b, t| b.* = t.v;
            return R.of(contract.rows.sumIn(s.acc, s.order, buf[0..terms.len]));
        },
        else => {},
    }
    var r = R.of(0);
    var m: f64 = 0;
    for (terms) |t| {
        r.v += t.v;
        r.lo += t.lo;
        r.hi += t.hi;
        m += @max(@abs(t.lo), @abs(t.hi));
    }
    if (mode == .ref) {
        const e = c.gamma(@floatFromInt(depth), c.unit(.f32)) * m;
        r.lo -= e;
        r.hi += e;
    }
    return r;
}

pub const nan16: u16 = 0x7fc0;
pub const inf16: u16 = 0x7f80;
pub const ninf16: u16 = 0xff80;

pub fn randomBf16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// Integers in [-bound, bound] as bf16 (exact while bound <= 256).
pub fn intsBf16(rng: std.Random, out: []u16, bound: u32) void {
    for (out) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 2 * bound + 1))) - @as(f32, @floatFromInt(bound)));
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// One kernel's test: its pipeline, host inputs and device buffers; `Def` says how to fill, launch and replay it.
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    def: *const Def,
    spec_: c.Spec,
    /// The variant a def serves (a mode, an epilogue, rows a launch).
    variant: u32 = 0,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    host: [8][]align(8) u8 = @splat(&.{}),
    refs: [8]Ref = undefined,
    /// In-place outputs: their bytes before every run.
    keep: [3]?struct { at: usize, ref: Ref } = @splat(null),
    dummy: Ref = undefined,
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    n_out: usize = 0,
    scratch: []f64 = &.{},

    pub fn kernel(p: *Probe, control: ?g.Gate) g.Kernel {
        return .{ .spec = p.spec_, .ctx = p, .vt = if (p.def.check != null) &vt_checked else &vt, .control = control };
    }
    const vt = g.hooks(Probe);
    const vt_checked = blk: {
        var v = g.hooks(Probe);
        v.check = checkHook;
        break :blk v;
    };
    fn checkHook(ctx: *anyopaque, gate: g.Gate) anyerror!?[]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return p.def.check.?(p, gate, try p.rig.outputBytes());
    }

    pub fn deinit(p: *Probe) void {
        p.release();
        p.a.free(p.lo);
        p.a.free(p.hi);
        p.a.free(p.scratch);
        p.rig.deinit();
    }

    fn release(p: *Probe) void {
        for (&p.host) |*h| {
            p.a.free(h.*);
            h.* = &.{};
        }
        p.keep = @splat(null);
    }

    /// Host bytes for input slot `i`, typed.
    pub fn alloc(p: *Probe, comptime T: type, i: usize, n: usize) ![]T {
        const bytes = try p.a.alignedAlloc(u8, .@"8", n * @sizeOf(T));
        @memset(bytes, 0);
        p.host[i] = bytes;
        return @alignCast(std.mem.bytesAsSlice(T, bytes));
    }
    pub fn get(p: *const Probe, comptime T: type, i: usize) []T {
        return @alignCast(std.mem.bytesAsSlice(T, p.host[i]));
    }
    /// Slot `i` bound as an input.
    pub fn in(p: *Probe, i: usize) !void {
        p.refs[i] = try p.rig.input(p.host[i]);
    }
    /// Slot `i` bound as an output (poisoned), or in place (its host bytes restored before each run).
    pub fn out(p: *Probe, i: usize, in_place: bool) !void {
        p.refs[i] = try p.rig.output(p.host[i].len);
        if (!in_place) return;
        for (&p.keep) |*k| if (k.* == null) {
            k.* = .{ .at = i, .ref = p.refs[i] };
            return;
        };
        return error.TooManyInPlace;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        p.release();
        p.shape = shape;
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        if (!try p.def.prep(p, prng.random())) return false;
        p.dummy = try p.rig.scratch(64);
        return true;
    }

    pub fn run(p: *Probe) !void {
        for (p.keep) |k| if (k) |x| @memcpy(p.rig.bytes(x.ref, p.host[x.at].len), p.host[x.at]);
        const pass = p.rig.begin();
        p.def.launch(p, pass.e);
        try p.rig.finish(pass);
    }

    /// Binds slot refs (null: the dummy) at 0.., then the argument struct.
    pub fn bind(p: *Probe, e: mtl.ComputeEncoder, slots: []const ?usize, args: anytype) void {
        e.setPipeline(p.pipe);
        for (slots, 0..) |s, i| {
            const r = if (s) |x| p.refs[x] else p.dummy;
            e.setBuffer(r.buf, r.off, i);
        }
        e.setBytes(std.mem.asBytes(&args), slots.len);
    }

    /// A ref `bytes` further into its buffer.
    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }

    pub fn go(e: mtl.ComputeEncoder, groups: [3]usize, threads: usize) void {
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
    }

    pub fn count(p: *Probe) usize {
        return p.def.count(p);
    }

    pub fn replay(p: *Probe, mode: Mode, out_: []R) !void {
        if (p.scratch.len < 1 << 16) {
            p.a.free(p.scratch);
            p.scratch = try p.a.alloc(f64, 1 << 16);
        }
        try p.def.eval(p, mode, out_);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const n = p.count();
        const rs = try p.a.alloc(R, n);
        defer p.a.free(rs);
        try p.replay(if (p.fill == .stress) .exact else .ref, rs);
        p.a.free(p.lo);
        p.a.free(p.hi);
        p.lo = try p.a.alloc(f64, n);
        p.hi = try p.a.alloc(f64, n);
        for (rs, want, mag, p.lo, p.hi) |r, *w, *m, *l, *h| {
            w.* = r.v;
            m.* = if (std.math.isFinite(r.v)) @abs(r.v) else 0;
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    /// The allowance about the float64 value: its range's farther end (a non-finite value is checked by class).
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.lo, p.hi) |*l, w, lo, hi| {
            const d = @max(@abs(w - lo), @abs(hi - w));
            l.* = if (std.math.isFinite(w) and std.math.isFinite(d)) d + 0x1p-140 else 0;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out_: []f64) !void {
        const rs = try p.a.alloc(R, out_.len);
        defer p.a.free(rs);
        try p.replay(.{ .sim = .{ .acc = acc, .order = order } }, rs);
        for (out_, rs) |*o, r| o.* = r.v;
    }
};

/// A kernel's test: inputs for `p.shape` (false: refused), its launch, its outputs and their float64 replay.
pub const Def = struct {
    prep: *const fn (*Probe, std.Random) anyerror!bool,
    launch: *const fn (*Probe, mtl.ComputeEncoder) void,
    count: *const fn (*Probe) usize,
    eval: *const fn (*Probe, Mode, []R) anyerror!void,
    check: ?*const fn (*Probe, g.Gate, []const u8) anyerror!?[]const u8 = null,
};

/// An entry: the def, its spec, the pipeline and source it runs, its variant, and the control's text change if any.
pub const Entry = struct {
    def: *const Def,
    spec: c.Spec,
    file: File,
    pipe: []const u8,
    variant: u32 = 0,
    control: ?[2][]const u8 = null,
    what: []const u8 = "",
};

pub fn blocks(n: usize, per: usize) usize {
    return (n + per - 1) / per;
}

pub fn i32_(x: usize) i32 {
    return @intCast(x);
}

/// A stated-rule kernel's check: every output inside its replay's range, a non-finite one of the replay's class.
pub fn checkReplay(p: *Probe, _: g.Gate, out: []const u8) !?[]const u8 {
    const n = p.count();
    const rs = try p.a.alloc(R, n);
    defer p.a.free(rs);
    try p.replay(.ref, rs);
    for (rs, 0..) |r, i| {
        const got = c.read(p.spec_.out, out, i);
        const ok = if (std.math.isFinite(r.v)) got >= r.lo and got <= r.hi else c.classOf(got) == c.classOf(r.v);
        if (!ok) {
            std.debug.print("  output {d}: {e} where the replay gives {e} in [{e}, {e}]\n", .{ i, got, r.v, r.lo, r.hi });
            return "an output leaves its replay's range";
        }
    }
    return null;
}

/// A raw kernel's check: its output bytes equal `want` (the replay's bytes) exactly.
pub fn sameBytes(got: []const u8, want: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, got, want)) return null;
    const at = std.mem.indexOfDiff(u8, got, want) orelse got.len;
    std.debug.print("  byte {d} differs\n", .{at});
    return "its bytes differ from the replay's";
}
