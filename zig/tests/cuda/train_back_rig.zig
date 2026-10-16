//! The float64 rig of the input-gradient tests (q4-back, moe-back, attn-back, fn-hc-back): numbers, 4-bit matrices.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// Flash Next's hidden width, the shape every case runs at.
pub const D = 2560;
pub const eps: f64 = 1e-6;
pub const threads = 16;
pub const row_counts = [_]usize{ 64, 256 };

/// The bf16 products' tolerance: the worst column within 3% of its largest value, 1% in L2.
pub const tol_col = 3e-2;
pub const tol_l2 = 1e-2;
/// A float64 reference against central differences.
pub const tol_fd = 1e-5;

pub fn bf16(x: f64) u16 {
    const f: f32 = @floatCast(x);
    const v: u32 = @bitCast(f);
    return @truncate((v + 0x7fff + ((v >> 16) & 1)) >> 16);
}

pub fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// x rounded to bf16, or to fp32.
pub fn rb(x: f64) f64 {
    return bf(bf16(x));
}

pub fn rf(x: f64) f64 {
    return @as(f32, @floatCast(x));
}

pub fn sig(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

pub fn silu(x: f64) f64 {
    return x * sig(x);
}

pub fn siluGrad(x: f64) f64 {
    const s = sig(x);
    return s * (1 + x * (1 - s));
}

pub fn dot(a: []const f64, b: []const f64) f64 {
    const Vec = @Vector(8, f64);
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) acc += @as(Vec, a[i..][0..8].*) * @as(Vec, b[i..][0..8].*);
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

pub fn axpy(y: []f64, c: f64, x: []const f64) void {
    const Vec = @Vector(8, f64);
    const cv: Vec = @splat(c);
    var i: usize = 0;
    while (i + 8 <= y.len) : (i += 8) y[i..][0..8].* = @as(Vec, y[i..][0..8].*) + cv * @as(Vec, x[i..][0..8].*);
    while (i < y.len) : (i += 1) y[i] += c * x[i];
}

pub fn dotF(a: []const f32, b: []const f64) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

/// A 4-bit [n, k] matrix in MLX's layout on the host: 8 codes a word, a bf16 scale and bias every 32 of a row.
pub const HQ4 = struct {
    n: usize,
    k: usize,
    words: []u32,
    sc: []u16,
    bi: []u16,

    /// Row i . x.
    pub fn rowDot(m: *const HQ4, i: usize, x: []const f64) f64 {
        const g = m.k / 32;
        const words = m.words[i * (m.k / 8) ..][0 .. m.k / 8];
        var total: f64 = 0;
        for (0..g) |gi| {
            var sq: f64 = 0;
            var sx: f64 = 0;
            for (0..4) |wi| {
                const w = words[gi * 4 + wi];
                const xs = x[gi * 32 + wi * 8 ..][0..8];
                inline for (0..8) |j| {
                    const q: f64 = @floatFromInt((w >> (4 * j)) & 0xf);
                    sq += q * xs[j];
                    sx += xs[j];
                }
            }
            total += bf(m.sc[i * g + gi]) * sq + bf(m.bi[i * g + gi]) * sx;
        }
        return total;
    }

    /// y += c row i.
    pub fn rowAxpy(m: *const HQ4, i: usize, c: f64, y: []f64) void {
        const g = m.k / 32;
        const words = m.words[i * (m.k / 8) ..][0 .. m.k / 8];
        for (0..g) |gi| {
            const cs = c * bf(m.sc[i * g + gi]);
            const cb = c * bf(m.bi[i * g + gi]);
            for (0..4) |wi| {
                const w = words[gi * 4 + wi];
                const ys = y[gi * 32 + wi * 8 ..][0..8];
                inline for (0..8) |j| ys[j] += cs * @as(f64, @floatFromInt((w >> (4 * j)) & 0xf)) + cb;
            }
        }
    }

    pub fn row(m: *const HQ4, i: usize, out: []f64) void {
        @memset(out, 0);
        m.rowAxpy(i, 1, out);
    }

    pub fn dev(m: *const HQ4, r: *Rig) !cuda.train_q4.Q4 {
        return .{ .w = try r.dev(u32, m.words), .s = try r.dev(u16, m.sc), .b = try r.dev(u16, m.bi), .n = m.n, .k = m.k };
    }

    pub fn bytes(m: *const HQ4) usize {
        return m.words.len * 4 + m.sc.len * 4;
    }
};

pub const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0, // device bytes allocated, for the memory lines
    bad: usize = 0,
    checks: usize = 0,

    pub fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        r.bytes += b.len;
        return b.ptr;
    }

    pub fn zeros(r: *Rig, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(n, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0, r.s.handle);
        r.bytes += b.len;
        return b.ptr;
    }

    pub fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        try r.s.synchronize();
        const out = try r.a.alloc(T, n);
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    pub fn release(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
        r.bytes = 0;
    }

    pub fn uni(r: *Rig, lo: f64, hi: f64) f64 {
        return lo + r.rng.float(f64) * (hi - lo);
    }

    /// A random 4-bit matrix with values in about [-amp, amp]: codes uniform, each group's scale and bias jittered.
    pub fn q4(r: *Rig, n: usize, k: usize, amp: f64) !HQ4 {
        const m: HQ4 = .{ .n = n, .k = k, .words = try r.a.alloc(u32, n * k / 8), .sc = try r.a.alloc(u16, n * k / 32), .bi = try r.a.alloc(u16, n * k / 32) };
        r.rng.bytes(std.mem.sliceAsBytes(m.words));
        for (m.sc, m.bi) |*s, *b| {
            const sv = 2 * amp / 15 * r.uni(0.6, 1.4);
            s.* = bf16(sv);
            b.* = bf16(-7.5 * sv * r.uni(0.85, 1.15));
        }
        return m;
    }

    pub fn f64s(r: *Rig, n: usize, lo: f64, hi: f64, round: enum { none, bf, f }) ![]f64 {
        const out = try r.a.alloc(f64, n);
        for (out) |*v| {
            const x = r.uni(lo, hi);
            v.* = switch (round) {
                .none => x,
                .bf => rb(x),
                .f => rf(x),
            };
        }
        return out;
    }

    pub fn f32s(r: *Rig, n: usize, lo: f64, hi: f64) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = @floatCast(r.uni(lo, hi));
        return out;
    }

    /// `got` against `want` [rows, cols]: the worst column's largest error over its largest |ref|; the L2 error.
    pub fn compare(r: *Rig, name: []const u8, got: []const f32, want: []const f64, cols: usize, tc: f64, tl: f64) void {
        const rows = want.len / cols;
        var top: f64 = 0;
        for (want) |w| top = @max(top, @abs(w));
        var worst: f64 = 0;
        var at: usize = 0;
        var dd: f64 = 0;
        var ww: f64 = 0;
        for (0..cols) |c| {
            var num: f64 = 0;
            var den: f64 = 0;
            for (0..rows) |i| {
                const w = want[i * cols + c];
                const diff = @as(f64, got[i * cols + c]) - w;
                num = if (std.math.isNan(diff)) std.math.inf(f64) else @max(num, @abs(diff));
                den = @max(den, @abs(w));
                dd += diff * diff;
                ww += w * w;
            }
            const e = num / @max(den, 1e-3 * top, 1e-300);
            if (e > worst) {
                worst = e;
                at = c;
            }
        }
        const l2 = @sqrt(dd / @max(ww, 1e-300));
        const ok = worst <= tc and l2 <= tl;
        r.checks += 1;
        if (!ok) r.bad += 1;
        std.debug.print("RESULT {s}: worst column {e:.2} (column {d} of {d}), relative L2 {e:.2}, largest |ref| {e:.2}{s}\n", .{ name, worst, at, cols, l2, top, if (ok) "" else "  <- FAIL" });
    }

    /// Central differences against a reference gradient: each difference over max(|ref|, 5% of the RMS `scale`).
    pub fn fd(r: *Rig, name: []const u8, num: []const f64, ref: []const f64, scale: f64) void {
        var worst: f64 = 0;
        for (num, ref) |n, g| {
            const e = @abs(n - g) / @max(@abs(g), 0.05 * scale, 1e-300);
            worst = if (std.math.isNan(e)) std.math.inf(f64) else @max(worst, e);
        }
        const ok = worst <= tol_fd;
        r.checks += 1;
        if (!ok) r.bad += 1;
        std.debug.print("RESULT fd {s}: float64 reference against central differences at {d} coordinates, worst relative {e:.2}{s}\n", .{ name, num.len, worst, if (ok) "" else "  <- FAIL" });
    }
};

pub fn rms(xs: []const f64) f64 {
    return @sqrt(dot(xs, xs) / @as(f64, @floatFromInt(@max(xs.len, 1))));
}

pub fn toF32(a: std.mem.Allocator, xs: []const f64) ![]f32 {
    const out = try a.alloc(f32, xs.len);
    for (out, xs) |*o, x| o.* = @floatCast(x);
    return out;
}

pub fn toBf16(a: std.mem.Allocator, xs: []const f64) ![]u16 {
    const out = try a.alloc(u16, xs.len);
    for (out, xs) |*o, x| o.* = bf16(x);
    return out;
}

/// f(ctx, i, thread) for i below n, on `threads` threads (inline if one cannot start).
pub fn parallel(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
    const C = @TypeOf(ctx);
    const Run = struct {
        fn run(c: C, first: usize, count: usize) void {
            var i = first;
            while (i < count) : (i += threads) f(c, i, first);
        }
    };
    var ts: [threads]?std.Thread = @splat(null);
    for (&ts, 0..) |*t, i| t.* = std.Thread.spawn(.{}, Run.run, .{ ctx, i, n }) catch null;
    for (ts, 0..) |t, i| if (t) |th| th.join() else Run.run(ctx, i, n);
}

/// GPU milliseconds between two events on the rig's stream.
pub const Timer = struct {
    a: cuda.Event,
    b: cuda.Event,

    pub fn init(r: *Rig) !Timer {
        return .{ .a = try cuda.Event.init(r.gpu.d, true), .b = try cuda.Event.init(r.gpu.d, true) };
    }
    pub fn deinit(t: *Timer) void {
        t.a.deinit();
        t.b.deinit();
    }
    pub fn start(t: *Timer, r: *Rig) !void {
        try t.a.record(r.s);
    }
    pub fn stop(t: *Timer, r: *Rig) !f32 {
        try t.b.record(r.s);
        try t.b.synchronize();
        return cuda.Event.elapsedMs(t.a, t.b);
    }
};

pub fn mib(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 20);
}
