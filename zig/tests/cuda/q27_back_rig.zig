//! Shared pieces of the 27B's backward checks: bf16 numbers, a host 4-bit matrix, the rig and float64 helpers.
const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen35");
const check = @import("check.zig");
const Gpu = check.Gpu;
const to = qwen27.train_ops;

pub const D = 5120;
pub const I = 17408;
pub const G = 64;
pub const NP = 16; // pocket neurons
pub const eps: f64 = 1e-6;
pub const threads = 16;
pub const row_counts = [_]usize{ 64, 256 };

pub const tol_col = 3e-2;
pub const tol_l2 = 1e-2;
pub const tol_fd = 1e-5;

pub fn bf16(x: f64) u16 {
    const f: f32 = @floatCast(x);
    const v: u32 = @bitCast(f);
    return @truncate((v + 0x7fff + ((v >> 16) & 1)) >> 16);
}

pub fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

pub fn rb(x: f64) f64 {
    return bf(bf16(x));
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

/// A 4-bit [n, k] matrix in MLX's layout on the host, a bf16 scale and bias every G of a row; rows kept dequantized.
pub const HQ4 = struct {
    n: usize,
    k: usize,
    words: []u32,
    sc: []u16,
    bi: []u16,
    vals: []f64, // [n, k] the dequantized values

    pub fn row(m: *const HQ4, i: usize) []const f64 {
        return m.vals[i * m.k ..][0..m.k];
    }

    pub fn dev(m: *const HQ4, r: *Rig) !to.Q4 {
        return .{ .w = try r.dev(u32, m.words), .s = try r.dev(u16, m.sc), .b = try r.dev(u16, m.bi), .n = m.n, .k = m.k, .group = G };
    }
};

pub const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,
    checks: usize = 0,

    pub fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    pub fn zeros(r: *Rig, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(n, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0, r.s.handle);
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
    }

    pub fn uni(r: *Rig, lo: f64, hi: f64) f64 {
        return lo + r.rng.float(f64) * (hi - lo);
    }

    /// A random 4-bit matrix with values in about [-amp, amp]: codes uniform, scales and biases jittered.
    pub fn q4(r: *Rig, n: usize, k: usize, amp: f64) !HQ4 {
        const m: HQ4 = .{ .n = n, .k = k, .words = try r.a.alloc(u32, n * k / 8), .sc = try r.a.alloc(u16, n * k / G), .bi = try r.a.alloc(u16, n * k / G), .vals = try r.a.alloc(f64, n * k) };
        r.rng.bytes(std.mem.sliceAsBytes(m.words));
        for (m.sc, m.bi) |*s, *b| {
            const sv = 2 * amp / 15 * r.uni(0.6, 1.4);
            s.* = bf16(sv);
            b.* = bf16(-7.5 * sv * r.uni(0.85, 1.15));
        }
        for (0..n * k) |i| {
            const q: f64 = @floatFromInt((m.words[i / 8] >> @intCast(4 * (i % 8))) & 0xf);
            m.vals[i] = bf(m.sc[i / G]) * q + bf(m.bi[i / G]);
        }
        return m;
    }

    pub fn f64s(r: *Rig, n: usize, lo: f64, hi: f64, round: bool) ![]f64 {
        const out = try r.a.alloc(f64, n);
        for (out) |*v| v.* = if (round) rb(r.uni(lo, hi)) else r.uni(lo, hi);
        return out;
    }

    /// `got` against `want` [rows, cols]: the worst column's relative difference and the relative L2 error.
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

    /// Central differences against a reference gradient's coordinates.
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

/// f(ctx, i) for i below n, on `threads` threads.
pub fn parallel(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void) void {
    const C = @TypeOf(ctx);
    const Run = struct {
        fn run(c: C, first: usize, count: usize) void {
            var i = first;
            while (i < count) : (i += threads) f(c, i);
        }
    };
    var ts: [threads]?std.Thread = @splat(null);
    for (&ts, 0..) |*t, i| t.* = std.Thread.spawn(.{}, Run.run, .{ ctx, i, n }) catch null;
    for (ts, 0..) |t, i| if (t) |th| th.join() else Run.run(ctx, i, n);
}

/// y = h rinv w, the RMS norm (rounded to bf16 when `round`).
pub fn normRow(h: []const f64, w: []const f64, round: bool, y: []f64) void {
    const rinv = 1 / @sqrt(dot(h, h) / @as(f64, @floatFromInt(h.len)) + eps);
    for (y, h, w) |*o, x, s| o.* = if (round) rb(x * rinv * s) else x * rinv * s;
}

/// The RMS norm undone: dh = rinv w dy - rinv^3 h (w dy . h) / n.
pub fn normBack(h: []const f64, w: []const f64, dy: []const f64, dh: []f64) void {
    const n: f64 = @floatFromInt(h.len);
    const rinv = 1 / @sqrt(dot(h, h) / n + eps);
    var t: f64 = 0;
    for (h, w, dy) |x, s, d| t += s * d * x;
    for (dh, h, w, dy) |*o, x, s, d| o.* = rinv * s * d - rinv * rinv * rinv * x * t / n;
}
