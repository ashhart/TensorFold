//! Host references for the DeepSeek release kernels: the FP8 round trip bit for bit, FP64 projections and their sums.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const rows = core.contract.rows;
const threads = core.contract.reference_threads;

/// Every E4M3 code's value (0x7f and 0xff NaN).
pub const e4m3: [256]f64 = blk: {
    @setEvalBranchQuota(100000);
    var v: [256]f64 = undefined;
    for (&v, 0..) |*x, i| x.* = core.quant.fp8.e4m3(i);
    break :blk v;
};

/// A UE8M0 scale byte's value (255 NaN).
pub fn e8m0(b: u8) f64 {
    return if (b == 255) std.math.nan(f64) else std.math.ldexp(@as(f64, 1), @as(i32, b) - 127);
}

/// The magnitude code nearest to q in [0, 448], a tie to the even code (round to nearest even with subnormals).
pub fn nearest(q: f64) u8 {
    var lo: u8 = 0;
    var hi: u8 = 126;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (e4m3[mid] <= q) lo = mid else hi = mid - 1;
    }
    if (lo == 126) return lo;
    const below = q - e4m3[lo];
    const above = e4m3[lo + 1] - q;
    if (below < above) return lo;
    if (above < below) return lo + 1;
    return if (lo % 2 == 0) lo else lo + 1;
}

/// ceil(log2(x)) of a positive finite x.
fn ceilLog2(x: f64) i32 {
    const f = std.math.frexp(x);
    return if (f.significand == 0.5) f.exponent - 1 else f.exponent;
}

/// One 32-value block of bf16 bits as the reference round-trips it: codes, the UE8M0 byte and the BF16 values.
pub fn actq(x: []const u16, codes: []u8, scale: *u8, rt: []u16) void {
    var amax: f32 = 0;
    for (x) |b| {
        const v = c.fromBf16(b);
        if (!std.math.isFinite(v)) {
            @memset(codes, 0x7f);
            scale.* = 0xff;
            @memset(rt, 0x7fc0);
            return;
        }
        amax = @max(amax, @abs(v));
    }
    const p: f32 = @max(amax, 1e-4) * @as(f32, 1.0 / 448.0);
    const e = ceilLog2(p);
    const s = std.math.ldexp(@as(f64, 1), e);
    scale.* = @intCast(e + 127);
    for (x, codes, rt) |b, *code, *o| {
        const v: f64 = c.fromBf16(b);
        const q = nearest(@min(@abs(v) / s, 448));
        const neg = std.math.signbit(v);
        code.* = q | if (neg) @as(u8, 0x80) else 0;
        const f: f32 = @floatCast(e4m3[q] * s);
        o.* = c.toBf16(if (neg) -f else f);
    }
}

/// The weight of a projection as the kernel multiplies it: FP8 codes with 32x32 block scales, or BF16 bits.
pub const Weight = union(enum) {
    fp8: struct { codes: []const u8, scales: []const u8 },
    bf16: []const u16,

    /// Column j's K values in float64.
    pub fn column(w: Weight, j: usize, k: usize, out: []f64) void {
        switch (w) {
            .fp8 => |f| for (out, 0..) |*o, i| {
                o.* = e4m3[f.codes[j * k + i]] * e8m0(f.scales[(j / 32) * (k / 32) + i / 32]);
            },
            .bf16 => |b| for (out, 0..) |*o, i| {
                o.* = c.fromBf16(b[j * k + i]);
            },
        }
    }
};

/// x [uniq][k] in float64 times W [n][k]: outputs [m][n], row r taking row r % uniq's values.
pub const Projection = struct {
    gpa: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    x: []const f64,
    w: Weight,

    pub fn uniq(p: Projection) usize {
        return p.x.len / p.k;
    }

    const Job = struct {
        p: *const Projection,
        cols: [2]usize,
        want: []f64,
        mag: []f64,
        acc: ?c.Accum = null,
        order: rows.Order = .sequential,
        failed: bool = false,

        pub fn run(job: *Job) void {
            job.work() catch {
                job.failed = true;
            };
        }

        fn work(job: *Job) !void {
            const p = job.p;
            const w = try p.gpa.alloc(f64, 2 * p.k);
            defer p.gpa.free(w);
            const t = w[p.k..];
            for (job.cols[0]..job.cols[1]) |j| {
                p.w.column(j, p.k, w[0..p.k]);
                for (0..p.uniq()) |u| {
                    const x = p.x[u * p.k ..][0..p.k];
                    var sum: f64 = 0;
                    var abs: f64 = 0;
                    if (job.acc) |a| {
                        for (t, x, w[0..p.k]) |*o, xv, wv| o.* = xv * wv;
                        sum = rows.sumIn(a, job.order, t);
                    } else for (x, w[0..p.k]) |xv, wv| {
                        sum += xv * wv;
                        abs += @abs(xv * wv);
                    }
                    var r = u;
                    while (r < p.m) : (r += p.uniq()) {
                        job.want[r * p.n + j] = sum;
                        job.mag[r * p.n + j] = abs;
                    }
                }
            }
        }
    };

    fn spread(p: *const Projection, want: []f64, mag: []f64, acc: ?c.Accum, order: rows.Order) !void {
        var jobs: [threads.maximum]Job = undefined;
        const count = threads.count(threads.maximum, p.n);
        for (jobs[0..count], 0..) |*job, i| job.* = .{ .p = p, .cols = .{ p.n * i / count, p.n * (i + 1) / count }, .want = want, .mag = mag, .acc = acc, .order = order };
        threads.run(Job, jobs[0..count], .normal);
        for (jobs[0..count]) |job| if (job.failed) return error.OutOfMemory;
    }

    /// Float64 outputs and the magnitude of each one's terms.
    pub fn reference(p: *const Projection, want: []f64, mag: []f64) !void {
        return p.spread(want, mag, null, .sequential);
    }

    /// The outputs with every sum in `acc` and `order` (gate 4's replay); mag is scratch.
    pub fn simulate(p: *const Projection, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const mag = try p.gpa.alloc(f64, out.len);
        defer p.gpa.free(mag);
        return p.spread(out, mag, acc, order);
    }
};

test "round trip oracle: ties go to the even code, the scale is the product's ceil log2, NaN blocks" {
    try std.testing.expectEqual(@as(u8, 126), nearest(448));
    try std.testing.expectEqual(@as(u8, 0), nearest(0x1p-10));
    try std.testing.expectEqual(@as(u8, 2), nearest(0x1.4p-8));
    try std.testing.expectEqual(@as(u8, 4), nearest(0x1.cp-8));
    try std.testing.expectEqual(@as(u8, 8), nearest(0x1.ep-7));
    var x: [32]u16 = @splat(0);
    x[0] = c.toBf16(448);
    x[1] = c.toBf16(-1);
    var codes: [32]u8 = undefined;
    var rt: [32]u16 = undefined;
    var s: u8 = 0;
    actq(&x, &codes, &s, &rt);
    try std.testing.expectEqual(@as(u8, 127), s);
    try std.testing.expectEqual(@as(u8, 126), codes[0]);
    try std.testing.expectEqual(@as(u8, 0xb8), codes[1]);
    try std.testing.expectEqual(@as(u8, 0), codes[2]);
    x[5] = c.toBf16(std.math.inf(f32));
    actq(&x, &codes, &s, &rt);
    try std.testing.expectEqual(@as(u8, 255), s);
    try std.testing.expectEqual(@as(u16, 0x7fc0), rt[3]);
}
