//! CSA64 fixtures compare Metal, FP32 contract and FP64 ideal on identical synthetic rows; docs/reference-precision.md.
pub const std = @import("std");
pub const mtl = @import("metal");
pub const tf = @import("tensorfold");
pub const math = @import("math.zig");
pub const attention_math = @import("attention_math.zig");

pub const kv = tf.deepseek_v41.kv;
pub const csa2 = tf.deepseek_v41.csa2;
pub const View = tf.deepseek_v41.attention.View;

pub const width: usize = 512;
pub const ratio: usize = 2;
pub const eps: f32 = 0.000001;

/// The local FP32 spacing at the FP64 reference value: the ULP unit for kernel deltas.
pub fn nextUpF32(x: f32) f32 {
    if (std.math.isNan(x) or std.math.isInf(x)) return x;
    const bits: u32 = @bitCast(x);
    if (bits == 0 or bits == 0x80000000) return @bitCast(@as(u32, 1));
    return @bitCast(if (bits >> 31 == 0) bits + 1 else bits - 1);
}

pub fn ulp32At(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f) or std.math.isInf(f)) return @abs(v);
    const a = @abs(f);
    if (a == 0) return 0x1p-149;
    const up = nextUpF32(a);
    if (std.math.isInf(up)) return @as(f64, a) - @as(f64, -nextUpF32(-a));
    return @as(f64, up) - @as(f64, a);
}

/// A BF16-cell comparison: NRMS against the ideal, and the count of cells the two sides disagree on.
pub const BfDelta = struct {
    count: usize = 0,
    nrms: f64 = 0,
    unequal: usize = 0,
    worst: usize = 0,
    worst_relative: f64 = 0,
};

pub fn deltaBf16(actual: []const u16, ideal: []const f64) BfDelta {
    var d: BfDelta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |cell, r, i| {
        const a: f64 = math.f(cell);
        const diff = a - r;
        squared += diff * diff;
        magnitude += r * r;
        const rel = @abs(diff) / @max(@abs(r), 0x1p-126);
        if (cell != math.b64(r)) {
            d.unequal += 1;
            if (rel > d.worst_relative) {
                d.worst_relative = rel;
                d.worst = i;
            }
        }
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

/// An FP32-cell comparison, used for pre-rounding stage boundaries that stay in FP32.
pub const Delta = struct {
    count: usize = 0,
    nrms: f64 = 0,
    max_abs: f64 = 0,
    max_ulp: f64 = 0,
    worst: usize = 0,
};

pub fn deltaF32VsF64(actual: []const f32, ideal: []const f64) Delta {
    var d: Delta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |a, r, i| {
        const diff = @as(f64, a) - r;
        squared += diff * diff;
        magnitude += r * r;
        const ad = @abs(diff);
        if (ad > d.max_abs) {
            d.max_abs = ad;
            d.worst = i;
        }
        const u = ad / ulp32At(r);
        if (u > d.max_ulp) d.max_ulp = u;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

pub fn printBf(case_name: []const u8, stage: []const u8, label: []const u8, d: BfDelta) void {
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"{s}","side":"{s}","cells":{d},"nrms_vs_ideal":{d},"bf16_unequal_vs_ideal":{d},"worst_index":{d},"worst_relative":{d}}}
    , .{ case_name, stage, label, d.count, d.nrms, d.unequal, d.worst, d.worst_relative });
}

pub fn printF32(case_name: []const u8, stage: []const u8, label: []const u8, d: Delta) void {
    std.debug.print("CSA64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"side\":\"{s}\",\"cells\":{d}," ++
        "\"nrms_vs_ideal\":{d},\"max_abs_vs_ideal\":{d},\"max_ulp_vs_ideal\":{d:.3}," ++
        "\"worst_index\":{d}}}\n", .{ case_name, stage, label, d.count, d.nrms, d.max_abs, d.max_ulp, d.worst });
}

pub const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    pub fn next(s: *Lcg) u64 {
        s.state = s.state *% 6364136223846793005 +% 1442695040888963407;
        return s.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    pub fn uniform(s: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(s.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
    pub fn scale(s: *Lcg, amplitude: f32) f32 {
        return s.uniform() * amplitude;
    }
};

pub const Buffers = struct {
    device: mtl.Device,
    items: [16]mtl.Buffer = undefined,
    count: usize = 0,
    pub fn get(b: *Buffers, comptime T: type, n: usize) !mtl.Buffer {
        const count = @max(n, 1);
        const x = try b.device.buffer(try std.math.mul(usize, @sizeOf(T), count), mtl.ResourceOptions.shared);
        b.items[b.count] = x;
        b.count += 1;
        @memset(x.slice(T, n), 0);
        return x;
    }
    pub fn deinit(b: *Buffers) void {
        for (b.items[0..b.count]) |x| x.deinit();
    }
};

pub fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

/// Directly generated cache codes give the kernel and FP64 ideal bit-identical BF16 keys.
pub fn e4(code: u8) f32 {
    const magnitude = code & 127;
    if (magnitude == 127) return std.math.nan(f32);
    const exponent: i32 = @intCast(magnitude >> 3);
    const mantissa = magnitude & 7;
    const base: f32 = if (exponent == 0) @as(f32, @floatFromInt(mantissa)) / 512 else std.math.ldexp(1 + @as(f32, @floatFromInt(mantissa)) / 8, exponent - 7);
    return if ((code & 128) != 0) -base else base;
}

pub fn e2(code: u8) f32 {
    const levels = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = levels[code & 7];
    return if ((code & 8) != 0) -v else v;
}

/// One ring row: 512 FP8 codes plus the FP32 scale of each 32-column block, exactly as dsv41_unpack reads it.
pub fn decodeWindowRow(bytes: []const u8) [width]u16 {
    var row: [width]u16 = undefined;
    for (0..width) |d| {
        const scale: f32 = @bitCast(@as(u32, bytes[width + d / 32]) << 23);
        row[d] = math.b(e4(bytes[d]) * scale);
    }
    return row;
}

/// One compressed row: 256 FP4 nibble bytes plus the FP8 scale of each 16-column block.
pub fn decodeMainRow(bytes: []const u8) [width]u16 {
    var row: [width]u16 = undefined;
    for (0..width) |d| {
        const nibble: u8 = if ((d & 1) != 0) bytes[d / 2] >> 4 else bytes[d / 2] & 15;
        row[d] = math.b(e2(nibble) * e4(bytes[width / 2 + d / 16]));
    }
    return row;
}

/// FP8 encoding under the block scale uses Metal rint semantics: nearest integer, ties to even.
pub fn roundHalfEven(x: f32) f32 {
    const toward: f32 = @trunc(x);
    const magnitude: f32 = if (@abs(x - toward) > 0.5) toward + std.math.sign(x) else toward;
    if (@abs(x - magnitude) != 0.5) return magnitude;
    const as_int: i64 = @intFromFloat(magnitude);
    if (@mod(as_int, 2) == 0) return magnitude;
    return if (x < 0) magnitude + 1 else magnitude - 1;
}

pub fn e4Code(x: f32) u8 {
    const a = @min(@abs(x), 448.0);
    if (a == 0) return if (std.math.signbit(x)) 128 else 0;
    const bits: u32 = @bitCast(a);
    const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127;
    const e = @max(unbiased, -6);
    const q: i32 = @intFromFloat(roundHalfEven(std.math.ldexp(a, 3 - @as(i32, @intCast(e)))));
    const code: u32 = if (e == -6 and q < 8) @intCast(q) else @intCast((e + 7) * 8 + q - 8);
    return @as(u8, @intCast(@min(code, 126))) | @as(u8, if (std.math.signbit(x)) 128 else 0);
}

pub fn e4Scale(x: f32) u8 {
    const bits: u32 = @bitCast(x);
    const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127 + @intFromBool((bits & 0x7fffff) != 0);
    return @intCast(unbiased + 127);
}

pub fn e2Code(x: f32) u8 {
    const a = @min(@abs(x), 6.0);
    const pick: u8 = if (a <= 0.25) 0 else if (a < 0.75) 1 else if (a <= 1.25) 2 else if (a < 1.75) 3 else if (a <= 2.5) 4 else if (a < 3.5) 5 else if (a <= 5.0) 6 else 7;
    return pick | @as(u8, if (std.math.signbit(x)) 8 else 0);
}

/// dsv41_pack kind 0: power-of-two FP8 scale and FP8 codes per 32 columns; both sides read decoded cells.
pub fn packWindowRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. width + width / 32], 0);
    for (0..width / 32) |block| {
        var a: f32 = 1e-4;
        for (values[block * 32 ..][0..32]) |v| a = @max(a, @abs(v));
        const z = a / 448;
        const zbits: u32 = @bitCast(z);
        const scale = std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast((zbits >> 23) & 255)) - 127 + @intFromBool((zbits & 0x7fffff) != 0));
        out[width + block] = e4Scale(scale);
        for (0..32) |j| out[block * 32 + j] = e4Code(values[block * 32 + j] / scale);
    }
}

/// dsv41_pack kind 1: FP8 scale and FP4 codes per 16 columns; both sides read decoded cells.
pub fn packMainRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. width / 2 + width / 16], 0);
    for (0..width / 16) |block| {
        var a: f32 = 6.0 * 0x1p-9;
        for (values[block * 16 ..][0..16]) |v| a = @max(a, @abs(v));
        const scale = e4(e4Code(@max(a, 6.0 * 0x1p-9) / 6.0));
        out[width / 2 + block] = e4Code(@max(a, 6.0 * 0x1p-9) / 6.0);
        for (0..16) |j| {
            const code = e2Code(values[block * 16 + j] / scale);
            const at = block * 8 + j / 2;
            if ((j & 1) != 0) out[at] |= code << 4 else out[at] |= code;
        }
    }
}

pub fn weightCells(gpa: std.mem.Allocator, rng: *Lcg) ![]u16 {
    const out = try gpa.alloc(u16, width);
    for (out) |*cell| {
        const amplitude: f32 = 0.5 + 0.25 * @as(f32, @floatFromInt(rng.next() % 1000)) / 1000;
        cell.* = math.b(amplitude);
    }
    return out;
}
