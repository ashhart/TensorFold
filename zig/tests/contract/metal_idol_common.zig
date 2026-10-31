//! Idol's kernel tests' shared parts: float64 values at fp16 points, their ranges, inputs and the family's library.
const std = @import("std");
const mtl = @import("metal");
pub const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;
pub const rows = contract.rows;
pub const ks = tf.idol.kernels;
pub const cfg = tf.idol.config;
pub const H = cfg.hidden;
pub const D = cfg.head_dim;

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value rounded once to fp16 (the kernels' half(fp32) of a value float64 holds exactly, or float64's own).
pub fn hf(v: f64) f64 {
    return @as(f16, @floatCast(v));
}

pub fn kref(r: rig_.Ref) ks.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

/// Every value the kernel may hold at a step: its fp32 steps widen both ends, its fp16 points round them.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5).
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    pub fn fp16(a: Range) Range {
        return .{ .lo = hf(a.lo), .hi = hf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn neg(a: Range) Range {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    pub fn sub(a: Range, b: Range) Range {
        return a.add(b.neg());
    }
    pub fn widen(a: Range, e: f64) Range {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    /// An increasing function at both ends.
    pub fn up(a: Range, comptime f: fn (f64) f64) Range {
        return .{ .lo = f(a.lo), .hi = f(a.hi) };
    }
    /// 1 / sqrt(a), a positive.
    pub fn rsqrt(a: Range) Range {
        return .{ .lo = 1 / @sqrt(a.hi), .hi = 1 / @sqrt(a.lo) };
    }
    /// 1 / a, a positive.
    pub fn recip(a: Range) Range {
        return .{ .lo = 1 / a.hi, .hi = 1 / a.lo };
    }
    /// a^2 over the range.
    pub fn square(a: Range) Range {
        const m = @max(a.lo * a.lo, a.hi * a.hi);
        return .{ .lo = if (a.lo <= 0 and a.hi >= 0) 0 else @min(a.lo * a.lo, a.hi * a.hi), .hi = m };
    }
};

pub fn tanh(x: f64) f64 {
    return std.math.tanh(x);
}

pub fn sqrt(x: f64) f64 {
    return @sqrt(x);
}

// Metal's error bounds with the engine's options (fast math on, the kernels' own `precise::` calls).
pub const div_ulps = 2.5;
pub const rsqrt_ulps = 2;
pub const sqrt_ulps = 3;
pub const exp_ulps = 4;
pub const tanh_ulps = 5;
pub const trig_ulps = 4;

/// f over [lo, hi] (cos or sin): its ends, and 1 or -1 where an extreme (a multiple of pi past `at`) falls inside.
fn trigOver(lo: f64, hi: f64, comptime f: fn (f64) f64, at: f64) Range {
    const pi = std.math.pi;
    if (!(hi - lo < 2 * pi)) return .{ .lo = -1, .hi = 1 };
    var out = Range{ .lo = @min(f(lo), f(hi)), .hi = @max(f(lo), f(hi)) };
    var k = @ceil((lo - at) / pi);
    while (k * pi + at <= hi) : (k += 1) {
        if (@mod(k, 2) == 0) out.hi = 1 else out.lo = -1;
    }
    return out;
}

fn cos(x: f64) f64 {
    return @cos(x);
}

fn sin(x: f64) f64 {
    return @sin(x);
}

pub fn cosOver(lo: f64, hi: f64) Range {
    return trigOver(lo, hi, cos, 0);
}

pub fn sinOver(lo: f64, hi: f64) Range {
    return trigOver(lo, hi, sin, std.math.pi / 2.0);
}

/// Each output's float64 value at the kernel's rounding points, and its range, kept for the bounds and intervals.
pub const Ends = struct {
    a: std.mem.Allocator,
    lo: []f64 = &.{},
    hi: []f64 = &.{},

    pub fn deinit(e: *Ends) void {
        e.a.free(e.lo);
        e.a.free(e.hi);
        e.lo = &.{};
        e.hi = &.{};
    }

    pub fn size(e: *Ends, n: usize) !void {
        e.deinit();
        e.lo = try e.a.alloc(f64, n);
        e.hi = try e.a.alloc(f64, n);
    }

    pub fn put(e: *Ends, want: []f64, mag: []f64, i: usize, w: f64, r: Range) void {
        want[i] = w;
        mag[i] = if (std.math.isFinite(w)) @abs(w) else 0;
        e.lo[i] = r.lo;
        e.hi[i] = r.hi;
    }

    /// The allowance about the float64 value: its range's farther end (a non-finite value is checked by class).
    pub fn bounds(e: *const Ends, want: []const f64, allow: []f64) void {
        for (allow, want, e.lo, e.hi) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(@abs(w - lo), @abs(hi - w)) else 0;
    }

    pub fn intervals(e: *const Ends, low: []f64, high: []f64) void {
        @memcpy(low, e.lo);
        @memcpy(high, e.hi);
    }
};

pub fn randomF16(rng: std.Random, out: []f16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = @floatCast(lo + rng.float(f32) * (hi - lo));
}

pub fn randomF32(rng: std.Random, out: []f32, lo: f32, hi: f32) void {
    for (out) |*v| v.* = lo + rng.float(f32) * (hi - lo);
}

/// An integer in [lo, hi].
pub fn int(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(lo + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo + 1)))));
}

/// +1 or -1.
pub fn sign(rng: std.Random) f32 {
    return if (rng.boolean()) 1 else -1;
}

pub const nan16: f16 = std.math.nan(f16);
pub const inf16: f16 = std.math.inf(f16);
pub const nan32: f32 = std.math.nan(f32);
pub const inf32: f32 = std.math.inf(f32);

/// [count, width] rows of which row r is `distinct`'s row r % (its rows): a period prime to every tile.
pub fn periodic(comptime T: type, a: std.mem.Allocator, distinct: []const T, width: usize, count: usize) ![]T {
    const p = distinct.len / width;
    const out = try a.alloc(T, count * width);
    for (0..count) |r| @memcpy(out[r * width ..][0..width], distinct[(r % p) * width ..][0..width]);
    return out;
}

/// One text change a control makes to the family's source.
pub const Edit = struct { from: []const u8, to: []const u8 };

/// A control: the kernel it breaks, the gate that must go red, its source edits and any text it appends.
pub const Control = struct { kernel: []const u8, what: []const u8, gate: g.Gate = .shapes, edits: []const Edit = &.{}, append: []const u8 = "" };

/// The denoiser's library: idol.metal after gemm_nax.metal, through kernels.zig's launchers.
pub const Lib = LibOf(ks.Kernels, &ks.names, sources.idol);

/// A source as the engine compiles it (or a control's copy), its pipelines made as tests ask for them; K's launchers.
pub fn LibOf(comptime K: type, comptime names: []const []const u8, comptime source: []const u8) type {
    return struct {
        a: std.mem.Allocator,
        device: mtl.Device,
        ctl: ?Control,
        k: K = undefined,
        built: bool = false,
        made: [names.len]bool = @splat(false),

        const This = @This();

        fn text(l: *This) ![]u8 {
            var out = try tf.frags.source(l.device, l.a, source);
            const x = l.ctl orelse return out;
            for (x.edits) |e| {
                errdefer l.a.free(out);
                if (std.mem.count(u8, out, e.from) != 1) {
                    std.debug.print("control {s}: text not found once: {s}\n", .{ x.what, e.from });
                    return error.ControlTextNotFoundOnce;
                }
                const next = try std.mem.replaceOwned(u8, l.a, out, e.from, e.to);
                l.a.free(out);
                out = next;
            }
            if (x.append.len == 0) return out;
            defer l.a.free(out);
            return std.mem.concat(l.a, u8, &.{ out, "\n", x.append });
        }

        /// The pipeline `name` made on this library (compiled on first use).
        pub fn need(l: *This, name: []const u8) !void {
            if (!l.built) {
                const src = try l.text();
                defer l.a.free(src);
                l.k.own = try mtl.Library.fromSource(l.device, src, mtl.CompileOptions.mlx());
                l.built = true;
            }
            for (names, 0..) |n, i| if (std.mem.eql(u8, n, name)) {
                if (!l.made[i]) l.k.p[i] = try mtl.Pipeline.init(l.device, l.k.own, n, false);
                l.made[i] = true;
                return;
            };
            return error.NoSuchPipeline;
        }

        pub fn deinit(l: *This) void {
            for (l.k.p, l.made) |p, m| if (m) p.deinit();
            if (l.built) l.k.own.deinit();
        }
    };
}
