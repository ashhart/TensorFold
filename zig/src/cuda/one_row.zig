//! A cuBLASLt plan's 1-row algorithm, taken only when each probe row run alone gives the plan's bits (shared).
const std = @import("std");
const cublaslt = @import("cublaslt.zig");
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Driver = @import("driver.zig").Driver;

const Allocator = std.mem.Allocator;

/// The probe's kinds of input row, `per_class` rows each.
pub const Class = enum { normal, wide, subnormal, cancel };

pub const per_class = 16;

/// Probe rows: each run alone on a candidate, and all together on the plan's algorithm.
pub const rows = per_class * @typeInfo(Class).@"enum".field_names.len;

/// Candidates the heuristic is asked for at 1 row.
pub const candidates = 8;

/// bf16 values in the probe weights' block, repeated over [n, k].
pub const block = 1 << 20;

/// The first of `cands` that runs in `budget`, keeps one K order, then gives the plan's bits (`ctx.same`); null: none.
pub fn pick(cands: []const cublaslt.Heuristic, budget: usize, ctx: anytype) !?usize {
    for (cands, 0..) |h, i| {
        if (h.state != 0 or h.workspace_size > budget) continue;
        if (!ctx.keepsOrder(h.algo)) continue;
        if (try ctx.same(h)) return i;
    }
    return null;
}

fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

/// A bf16 of sign, unbiased exponent (-126..127 normal; -127 for a subnormal) and 7 mantissa bits.
fn bits(neg: bool, exp: i32, mant: u7) u16 {
    const e: u16 = @intCast(exp + 127);
    return (@as(u16, @intFromBool(neg)) << 15) | (e << 7) | mant;
}

/// Input rows [rows, k], `per_class` of each Class in order; each class's rows are described at its branch.
pub fn inputs(out: []u16, k: usize, seed: u64) void {
    std.debug.assert(out.len == rows * k);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (0..rows) |row| {
        const class: Class = @fromBackingInt(@intCast(row / per_class));
        const i = row % per_class;
        const line = out[row * k ..][0..k];
        switch (class) {
            // N(0, 1) scaled by 2^-4 .. 2^3 a row
            .normal => {
                const scale = std.math.ldexp(@as(f32, 1), @as(i32, @intCast(i % 8)) - 4);
                for (line) |*v| v.* = bf16(r.floatNorm(f32) * scale);
            },
            // exponents across 2^-100 .. 2^100: products span the fp32 range short of overflow
            .wide => for (line) |*v| {
                v.* = bits(r.boolean(), r.intRangeAtMost(i32, -100, 100), r.int(u7));
            },
            // half bf16 subnormals, half the smallest normals
            .subnormal => for (line, 0..) |*v, j| {
                v.* = if (j % 2 == 0) bits(r.boolean(), -127, r.intRangeAtMost(u7, 1, 127)) else bits(r.boolean(), r.intRangeAtMost(i32, -126, -110), r.int(u7));
            },
            // +v, -v pairs over 2^0 .. 2^20 with zeros between; the class's first row all zero, its second all one
            .cancel => {
                if (i < 2) {
                    @memset(line, if (i == 0) 0 else bf16(1));
                    continue;
                }
                var j: usize = 0;
                while (j < k) : (j += 1) {
                    if (j % 5 == 4) {
                        line[j] = 0;
                        continue;
                    }
                    const v = bits(r.boolean(), r.intRangeAtMost(i32, 0, 20), r.int(u7));
                    line[j] = v;
                    if (j + 1 < k and (j + 1) % 5 != 4) {
                        j += 1;
                        line[j] = v ^ 0x8000;
                    }
                }
            },
        }
    }
}

/// The weights' block: N(0, 0.03), one value in 512 an outlier of magnitude 1 to 8.
pub fn weights(out: []u16, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (out) |*v| v.* = bf16(if (r.uintLessThan(u32, 512) == 0) (r.float(f32) * 7 + 1) * (if (r.boolean()) @as(f32, -1) else 1) else r.floatNorm(f32) * 0.03);
}

/// The probe on the device: weights [n, k], input rows, the plan's outputs and a candidate's, and their host copies.
pub const Probe = struct {
    w: DeviceBuffer,
    x: DeviceBuffer,
    ref: DeviceBuffer,
    y: DeviceBuffer,
    want: []u8,
    got: []u8,

    pub fn init(gpa: Allocator, d: *const Driver, n: usize, k: usize) !Probe {
        const host = try gpa.alloc(u16, @max(block, rows * k));
        defer gpa.free(host);
        const seed = 0x9e3779b97f4a7c15 ^ (n *% 31 +% k);
        var pr: Probe = undefined;
        pr.w = try DeviceBuffer.alloc(d, 2 * n * k);
        errdefer pr.w.free();
        weights(host[0..block], seed);
        var off: usize = 0;
        while (off < n * k) : (off += block) try pr.w.upload(2 * off, std.mem.sliceAsBytes(host[0..@min(block, n * k - off)]));
        pr.x = try DeviceBuffer.alloc(d, 2 * rows * k);
        errdefer pr.x.free();
        inputs(host[0 .. rows * k], k, seed +% 1);
        try pr.x.upload(0, std.mem.sliceAsBytes(host[0 .. rows * k]));
        pr.ref = try DeviceBuffer.alloc(d, 2 * rows * n);
        errdefer pr.ref.free();
        pr.y = try DeviceBuffer.alloc(d, 2 * rows * n);
        errdefer pr.y.free();
        pr.want = try gpa.alloc(u8, 2 * rows * n);
        errdefer gpa.free(pr.want);
        pr.got = try gpa.alloc(u8, 2 * rows * n);
        return pr;
    }

    pub fn deinit(pr: *Probe, gpa: Allocator) void {
        gpa.free(pr.got);
        gpa.free(pr.want);
        pr.y.free();
        pr.ref.free();
        pr.x.free();
        pr.w.free();
    }
};

/// Rows a check mode compared, a 1-row call's output against the plan's algorithm's, and how many differed.
pub const Tally = struct { rows: usize = 0, differ: usize = 0 };

const H = cublaslt.Heuristic;

fn cand(state: c_int, ws: usize, tag: u64) H {
    var h = std.mem.zeroes(H);
    h.state = state;
    h.workspace_size = ws;
    h.algo.data[0] = tag;
    return h;
}

/// A stand-in for the device: algo tag 1 splits K, tag 2 gives other bits, any other tag the plan's.
const Fake = struct {
    probed: *usize,
    fn keepsOrder(f: Fake, a: cublaslt.Algo) bool {
        _ = f;
        return a.data[0] != 1;
    }
    fn same(f: Fake, h: H) !bool {
        f.probed.* += 1;
        return h.algo.data[0] != 2;
    }
};

test "pick: an unrunnable, an over-budget, a split-K and an other-bits candidate are passed over, in order" {
    var probed: usize = 0;
    const f: Fake = .{ .probed = &probed };
    const cands = [_]H{ cand(1, 0, 0), cand(0, 1 << 30, 0), cand(0, 0, 1), cand(0, 0, 2), cand(0, 0, 3), cand(0, 0, 4) };
    try std.testing.expectEqual(@as(?usize, 4), try pick(&cands, 1 << 20, f));
    try std.testing.expectEqual(@as(usize, 2), probed); // the split-K one never reached the probe
    probed = 0;
    try std.testing.expectEqual(@as(?usize, null), try pick(cands[0..4], 1 << 20, f)); // none: the plan's own runs
    try std.testing.expectEqual(@as(usize, 1), probed);
    try std.testing.expectEqual(@as(?usize, null), try pick(&.{}, 1 << 20, f));
}

test "the probe rows: 16 of each class, the wide class across 2^-100..2^100, subnormals present, the cancel rows paired" {
    const k = 64;
    var x: [rows * k]u16 = undefined;
    inputs(&x, k, 7);
    var subs: usize = 0;
    var lo: i32 = 200;
    var hi: i32 = -200;
    for (0..per_class) |i| for (0..k) |j| {
        const w = x[(per_class + i) * k + j];
        const e: i32 = @as(i32, (w >> 7) & 0xff) - 127;
        lo = @min(lo, e);
        hi = @max(hi, e);
        const s = x[(2 * per_class + i) * k + j];
        if ((s >> 7) & 0xff == 0 and s & 0x7f != 0) subs += 1;
    };
    try std.testing.expect(lo < -90 and hi > 90 and hi <= 100);
    try std.testing.expect(subs >= per_class * k / 2);
    const c = x[3 * per_class * k ..];
    for (c[0..k]) |v| try std.testing.expectEqual(@as(u16, 0), v);
    for (c[k .. 2 * k]) |v| try std.testing.expectEqual(bf16(1), v);
    try std.testing.expectEqual(c[2 * k] ^ 0x8000, c[2 * k + 1]); // +v, -v
    var y: [rows * k]u16 = undefined;
    inputs(&y, k, 7);
    try std.testing.expectEqualSlices(u16, &x, &y); // seeded: the same probe every load
}
