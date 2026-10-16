//! Shared parts of tf-flashnext-widths: synthetic MLX matrices, float64 references and the row checks.
const std = @import("std");
const mtl = @import("metal");

pub const opts = mtl.ResourceOptions.shared;
pub const a = std.heap.page_allocator;
pub const unit: f64 = 0x1p-24; // fp32's unit roundoff

pub fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// One synthetic MLX affine matrix [n, k]: codes kept beside their packed words, bf16 tables.
pub const Mat = struct {
    n: usize,
    k: usize,
    bits: u8,
    group: usize,
    codes: []u8,
    words: []u8,
    scales: []u16,
    biases: []u16,
    skew: f64 = 1, // the references' scale factor: 1.01 makes the negative controls' wrong references

    /// The same matrix with its scales `f` times their values in every reference (the kernels' bytes unchanged).
    pub fn skewed(m: Mat, f: f64) Mat {
        var out = m;
        out.skew = f;
        return out;
    }

    pub fn init(n: usize, k: usize, bits: u8, group: usize, seed: u64) !Mat {
        return initBelow(n, k, bits, group, seed, @as(u32, 1) << @intCast(bits));
    }

    /// Codes below `top` only (an 8-bit matrix whose codes fit 6 bits, so it also exists at 6 bits).
    pub fn initBelow(n: usize, k: usize, bits: u8, group: usize, seed: u64, top: u32) !Mat {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const tables = n * (k / group);
        const codes = try a.alloc(u8, n * k);
        for (codes) |*c| c.* = @intCast(r.uintLessThan(u32, top));
        const m = Mat{ .n = n, .k = k, .bits = bits, .group = group, .codes = codes, .words = try packCodes(codes, bits), .scales = try a.alloc(u16, tables), .biases = try a.alloc(u16, tables) };
        for (m.scales, m.biases) |*s, *b| {
            s.* = toBf((r.float(f32) + 0.5) / 64);
            b.* = toBf(-(r.float(f32) + 0.5) * 0.15);
        }
        return m;
    }

    /// The same codes stored at `bits` in groups of `group` (which divides this group): the same values exactly.
    pub fn recode(m: Mat, bits: u8, group: usize) !Mat {
        if (m.group % group != 0) return error.GroupDoesNotDivide;
        for (m.codes) |c| if (c >> @intCast(bits) != 0) return error.CodeTooWide;
        const reps = m.group / group;
        const out = Mat{ .n = m.n, .k = m.k, .bits = bits, .group = group, .codes = m.codes, .words = try packCodes(m.codes, bits), .scales = try a.alloc(u16, m.scales.len * reps), .biases = try a.alloc(u16, m.biases.len * reps) };
        for (m.scales, m.biases, 0..) |sv, bv, i| for (0..reps) |j| {
            out.scales[i * reps + j] = sv;
            out.biases[i * reps + j] = bv;
        };
        return out;
    }

    pub fn deinit(m: Mat) void {
        a.free(m.words);
        a.free(m.scales);
        a.free(m.biases);
    }

    pub fn code(m: Mat, j: usize, i: usize) u32 {
        return m.codes[j * m.k + i];
    }

    pub fn deq(m: Mat, j: usize, i: usize) f64 {
        const g = j * (m.k / m.group) + i / m.group;
        return m.skew * fromBf(m.scales[g]) * @as(f64, @floatFromInt(m.code(j, i))) + fromBf(m.biases[g]);
    }

    /// s q + b and s q + |b| for every weight, row-major [n, k]: the reference's terms and their magnitudes.
    pub fn dense(m: Mat) ![2][]f64 {
        const out = [2][]f64{ try a.alloc(f64, m.n * m.k), try a.alloc(f64, m.n * m.k) };
        for (0..m.n) |j| for (0..m.k) |i| {
            const g = j * (m.k / m.group) + i / m.group;
            const sq = m.skew * fromBf(m.scales[g]) * @as(f64, @floatFromInt(m.code(j, i)));
            out[0][j * m.k + i] = sq + fromBf(m.biases[g]);
            out[1][j * m.k + i] = sq + @abs(fromBf(m.biases[g]));
        };
        return out;
    }

    /// Weights as qmmw's dequantw rounds them, bf16(s q + b), their magnitudes, and a bf16 step where fp32 nears a tie.
    pub fn denseBf(m: Mat) ![3][]f64 {
        const out = [3][]f64{ try a.alloc(f64, m.n * m.k), try a.alloc(f64, m.n * m.k), try a.alloc(f64, m.n * m.k) };
        for (0..m.n) |j| for (0..m.k) |i| {
            const v = m.deq(j, i);
            const w = bfRound(v);
            const tie = @abs(@abs(v - w) - bfStep(v) / 2) <= @abs(v) * 0x1p-23; // fp32's error, twice over
            out[0][j * m.k + i] = w;
            out[1][j * m.k + i] = @abs(w);
            out[2][j * m.k + i] = if (tie) bfStep(v) else 0;
        };
        return out;
    }
};

/// v rounded to bf16 (nearest, ties to even), as float64.
pub fn bfRound(v: f64) f64 {
    return fromBf(toBf(@floatCast(v)));
}

/// The bf16 step at v's binade.
pub fn bfStep(v: f64) f64 {
    if (v == 0) return 0x1p-133;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(v).exponent - 8);
}

/// The float64 product of one row with one weight row, and its magnitude: sum of (s q + |b|) |x|.
pub fn dot(x: []const f64, terms: [2][]f64, j: usize) [2]f64 {
    const k = x.len;
    var want: f64 = 0;
    var mag: f64 = 0;
    for (x, terms[0][j * k ..][0..k], terms[1][j * k ..][0..k]) |xv, d, m| {
        want += xv * d;
        mag += @abs(xv) * m;
    }
    return .{ want, mag };
}

/// dot over denseBf's weights: the product, its magnitude sum |x| |w|, and the tie slack sum |x| step.
pub fn dot3(x: []const f64, terms: [3][]f64, j: usize) [3]f64 {
    const k = x.len;
    var out: [3]f64 = .{ 0, 0, 0 };
    for (x, terms[0][j * k ..][0..k], terms[1][j * k ..][0..k], terms[2][j * k ..][0..k]) |xv, w, m, e| {
        out[0] += xv * w;
        out[1] += @abs(xv) * m;
        out[2] += @abs(xv) * e;
    }
    return out;
}

/// ratio for a dot3 reference: fp32 sums of `terms` steps over the magnitude, the tie slack, one bf16 rounding.
pub fn ratio3(got: f64, ref: [3]f64, terms: usize, bf16_out: bool) f64 {
    const tol = @as(f64, @floatFromInt(terms)) * unit * ref[1] + ref[2] + (if (bf16_out) @abs(ref[0]) * 0x1p-8 else 0) + 1e-30;
    return @abs(got - ref[0]) / tol;
}

/// Codes as an MLX little-endian bit stream of `bits`-wide values.
pub fn packCodes(codes: []const u8, bits: u8) ![]u8 {
    const out = try a.alloc(u8, (codes.len * bits + 31) / 32 * 4);
    @memset(out, 0);
    for (codes, 0..) |c, i| {
        const bit = i * bits;
        const v = @as(u32, c) << @intCast(bit % 8);
        for (0..3) |b| {
            if (bit / 8 + b < out.len) out[bit / 8 + b] |= @truncate(v >> @intCast(8 * b));
        }
    }
    return out;
}

/// Every check's outcome: failures, the largest difference as a fraction of its bound, and the negative controls.
pub const Tally = struct { failed: usize = 0, worst: f64 = 0, checks: usize = 0, controls: usize = 0, least: f64 = std.math.inf(f64) };

/// `rows` rows of x [rows, k] as bf16 (in a device buffer) and float64.
pub const Inputs = struct {
    buf: mtl.Buffer,
    x: []f64,
    rows: usize,
    k: usize,

    pub fn init(device: mtl.Device, rows: usize, k: usize, seed: u64) !Inputs {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const in = Inputs{ .buf = try device.buffer(rows * k * 2, opts), .x = try a.alloc(f64, rows * k), .rows = rows, .k = k };
        for (in.buf.slice(u16, rows * k), in.x) |*v, *d| {
            v.* = toBf(r.float(f32) * 2 - 1);
            d.* = fromBf(v.*);
        }
        return in;
    }

    pub fn row(in: Inputs, r: usize) []const f64 {
        return in.x[r * in.k ..][0..in.k];
    }
};

/// A column's result against float64: the error over its bound (fp32 sums of `terms` steps, one bf16 rounding).
pub fn ratio(got: f64, want: f64, mag: f64, terms: usize, bf16_out: bool) f64 {
    const tol = @as(f64, @floatFromInt(terms)) * unit * mag + (if (bf16_out) @abs(want) * 0x1p-8 else 0) + 1e-30;
    return @abs(got - want) / tol;
}

/// The worst column of one product, printed with its max abs and relative error and its share of the bound.
pub const Worst = struct {
    frac: f64 = 0,
    col: usize = 0,
    abs: f64 = 0,
    rel: f64 = 0,
    nan: bool = false,

    pub fn add(w: *Worst, col: usize, got: f64, want: f64, frac: f64) void {
        if (std.math.isNan(got)) w.nan = true;
        const d = @abs(got - want);
        if (frac > w.frac or std.math.isNan(got)) {
            w.frac = if (std.math.isNan(got)) std.math.inf(f64) else frac;
            w.col = col;
            w.abs = d;
            w.rel = d / (@abs(want) + 1e-30);
        }
    }

    pub fn report(w: Worst, t: *Tally, label: []const u8) void {
        t.checks += 1;
        t.worst = @max(t.worst, w.frac);
        const ok = w.frac <= 1 and !w.nan;
        if (!ok) t.failed += 1;
        std.debug.print("RESULT {s}: worst column {d}: max abs {e:.3}, relative {e:.3}; {d:.4} of its bound{s}\n", .{ label, w.col, w.abs, w.rel, w.frac, if (ok) "" else " FAIL" });
    }

    /// A negative control: the same results against a reference with scales 1% off must exceed the bound somewhere.
    pub fn control(w: Worst, t: *Tally, label: []const u8) void {
        t.checks += 1;
        t.controls += 1;
        t.least = @min(t.least, w.frac);
        const ok = w.frac > 1;
        if (!ok) t.failed += 1;
        std.debug.print("RESULT {s}: negative control, scales 1% off: {d:.3} of its bound{s}\n", .{ label, w.frac, if (ok) "" else " FAIL (the bound would pass a wrong scale)" });
    }
};

/// Rows [0, rows) of two outputs (row stride `ld` bytes) equal byte for byte; the first difference printed.
pub fn sameRows(t: *Tally, label: []const u8, window: usize, want: []const u8, got: []const u8, rows: usize, ld: usize) bool {
    t.checks += 1;
    if (std.mem.eql(u8, want[0 .. rows * ld], got[0 .. rows * ld])) return true;
    const i = std.mem.indexOfDiff(u8, want, got).?;
    var differ: usize = 0;
    for (0..rows) |r| {
        if (!std.mem.eql(u8, want[r * ld ..][0..ld], got[r * ld ..][0..ld])) differ += 1;
    }
    std.debug.print("RESULT {s}: {d}-row windows DIFFER in {d} of {d} rows; first at row {d} byte {d} FAIL\n", .{ label, window, differ, rows, i / ld, i % ld });
    t.failed += 1;
    return false;
}

pub fn commit(cb: anytype, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }
}

/// A device buffer of `n` bytes filled with 0xff (NaN in bf16 and fp32): a launch that writes nothing fails.
pub fn poisoned(device: mtl.Device, n: usize) !mtl.Buffer {
    const b = try device.buffer(@max(n, 16), opts);
    @memset(b.slice(u8, @max(n, 16)), 0xff);
    return b;
}

/// A device buffer holding `bytes`.
pub fn upload(device: mtl.Device, bytes: []const u8) !mtl.Buffer {
    const b = try device.buffer(@max(bytes.len, 16), opts);
    @memcpy(b.slice(u8, bytes.len), bytes);
    return b;
}
