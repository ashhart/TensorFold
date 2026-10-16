//! EXLR8 tiles: the mcg trellis codebook rounded to E4M3, the tile placement, and W = diag(suh) Hb T Hb diag(svh).

const std = @import("std");
const exl3 = @import("exl3_format.zig");

/// Bytes of one 16x16 tile at width k: 256 values of k bits.
pub fn tileBytes(k: u8) usize {
    return 32 * @as(usize, k);
}

/// OCP E4M3 (float8_e4m3fn): nearest even, saturating at 448; NaN keeps its sign as 0x7F.
pub fn e4m3FromF32(x: f32) u8 {
    const bits: u32 = @bitCast(x);
    const sign: u8 = @intCast((bits >> 24) & 0x80);
    const a = bits & 0x7FFFFFFF;
    if (a > 0x7F800000) return sign | 0x7F;
    if (a >= 0x43E00000) return sign | 0x7E;
    if (a >= 0x3C800000) {
        // At or above 2^-6: keep three mantissa bits; a carry moves the exponent up exactly.
        const r = (a + 0x7FFFF + ((a >> 20) & 1)) & 0xFFF00000;
        return sign | @as(u8, @intCast(((r >> 23) - 120) << 3 | ((r >> 20) & 7)));
    }
    const e = a >> 23;
    if (e == 0 or e < 141 - 40) return sign;
    // Below 2^-6 the codes count steps of 2^-9: n = round_even(a / 2^-9), and code 8 is 2^-6 itself.
    const s: u6 = @intCast(141 - e);
    const m: u64 = (a & 0x7FFFFF) | 0x800000;
    const half = @as(u64, 1) << (s - 1);
    const rem = m & ((@as(u64, 1) << s) - 1);
    var n = m >> s;
    if (rem > half or (rem == half and n & 1 == 1)) n += 1;
    return sign | @as(u8, @intCast(n));
}

/// The value of an E4M3 code; 0x7F and 0xFF are NaN.
pub fn e4m3Value(code: u8) f32 {
    const e: u5 = @intCast((code >> 3) & 15);
    const m: f32 = @floatFromInt(code & 7);
    const v: f32 = if (e == 15 and code & 7 == 7)
        std.math.nan(f32)
    else if (e == 0)
        m * 0x1p-9
    else
        (8 + m) * std.math.ldexp(@as(f32, 1), @as(i32, e) - 10);
    return if (code & 0x80 != 0) -v else v;
}

/// One state's weight code: the fp16 sum of the mcg codebook, then E4M3 (the format's rounding).
pub fn codebookCode(state: u16) u8 {
    return e4m3FromF32(@floatCast(exl3.codebookValue(.mcg, state)));
}

/// Every state's code, computed once for fast decoding.
pub const Codebook = struct {
    codes: [65536]u8,

    pub fn init(c: *Codebook) void {
        for (&c.codes, 0..) |*v, s| v.* = codebookCode(@intCast(s));
    }
};

fn word(tile: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, tile[4 * i ..][0..4], .little);
}

/// The 16-bit state of stream value p: the bits ending at (p + 1) k, circular, MSB first in little-endian u32 words.
pub fn tileState(k: u8, tile: []const u8, p: usize) u16 {
    const ring = 256 * @as(usize, k);
    const start = ((p + 1) * k + ring - 16) % ring;
    const off: u5 = @intCast(start % 32);
    const lo = word(tile, start / 32);
    if (off <= 16) return @truncate(lo >> @intCast(16 - @as(u6, off)));
    const hi = word(tile, (start / 32 + 1) % (ring / 32));
    return @truncate((lo << off - 16) | (hi >> @intCast(48 - @as(u6, off))));
}

/// Value p = 8 l + j sits at row 2 (l mod 4) + (j & 1) + 8 ((j >> 1) & 1), column l / 4 + 8 (j >> 2).
pub fn placement(p: usize) [2]u8 {
    const l = p / 8;
    const j = p % 8;
    return .{ @intCast(2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1)), @intCast(l / 4 + 8 * (j >> 2)) };
}

/// A tile's 256 codes at their (row, column), written with the given row stride.
pub fn decodeTile(book: *const Codebook, k: u8, tile: []const u8, out: []u8, stride: usize) void {
    for (0..256) |p| {
        const at = placement(p);
        out[@as(usize, at[0]) * stride + at[1]] = book.codes[tileState(k, tile, p)];
    }
}

/// Tile order of a run: gate/up halves put tile (k16, c) at 8 k16 + c, down halves tile (b, r, c) at 64 b + 8 r + c.
pub const Order = enum { out_major, in_major };

/// The codes of a run of tiles as a row-major (rows, cols) matrix, rows the input side.
pub fn decodeRun(book: *const Codebook, k: u8, order: Order, bytes: []const u8, rows: usize, cols: usize, out: []u8) !void {
    const tb = tileBytes(k);
    if ((order == .out_major and cols != 128) or (order == .in_major and rows != 128) or rows % 16 != 0 or
        cols % 128 != 0 or bytes.len != rows * cols / 256 * tb or out.len != rows * cols) return error.BadExlr8Run;
    for (0..rows / 16) |tr| for (0..cols / 16) |tc| {
        const index = switch (order) {
            .out_major => 8 * tr + tc,
            .in_major => 64 * (tc / 8) + 8 * tr + tc % 8,
        };
        decodeTile(book, k, bytes[index * tb ..][0..tb], out[(16 * tr) * cols + 16 * tc ..], cols);
    };
}

fn fwht(v: []f64, stride: usize) void {
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * d) for (base..base + d) |i| {
            const a = v[i * stride];
            const b = v[(i + d) * stride];
            v[i * stride] = a + b;
            v[(i + d) * stride] = a - b;
        };
    }
}

/// W = diag(suh) Hb T Hb diag(svh), exact in f64: sums stay multiples of 2^-9 below 2^16, scales add 22 bits.
pub fn reconstruct(codes: []const u8, rows: usize, cols: usize, suh: []const f32, svh: []const f32, out: []f64) !void {
    if (rows % 128 != 0 or cols % 128 != 0 or codes.len != rows * cols or out.len != codes.len or
        suh.len != rows or svh.len != cols) return error.BadExlr8Reconstruct;
    var block: [128 * 128]f64 = undefined;
    for (0..rows / 128) |bi| for (0..cols / 128) |bj| {
        reconstructBlock(codes, cols, bi, bj, suh, svh, &block);
        for (0..128) |r| @memcpy(out[(bi * 128 + r) * cols + bj * 128 ..][0..128], block[r * 128 ..][0..128]);
    };
}

/// Block (bi, bj) of W as a row-major 128 x 128 array, from row-major codes `cols` wide.
pub fn reconstructBlock(codes: []const u8, cols: usize, bi: usize, bj: usize, suh: []const f32, svh: []const f32, out: *[128 * 128]f64) void {
    for (0..128) |r| for (0..128) |c| {
        out[r * 128 + c] = e4m3Value(codes[(bi * 128 + r) * cols + bj * 128 + c]);
    };
    for (0..128) |c| fwht(out[c..], 128);
    for (0..128) |r| fwht(out[r * 128 ..], 1);
    for (0..128) |r| for (0..128) |c| {
        out[r * 128 + c] *= @as(f64, suh[bi * 128 + r]) * @as(f64, svh[bj * 128 + c]) / 128.0;
    };
}

/// The reference's fp32 Hadamard entry: 1/sqrt(128) rounded to f64, then to f32.
pub const had_f32: f32 = @floatCast(1.0 / @sqrt(@as(f64, 128)));

fn had(i: usize, k: usize) f32 {
    return if (@popCount(i & k) % 2 == 0) had_f32 else -had_f32;
}

/// Block (bi, bj) of W in fp32 in the reference's order: X = H T, then X' H, each a fused chain in k order.
pub fn reconstructBlockF32(codes: []const u8, cols: usize, bi: usize, bj: usize, suh: []const f32, svh: []const f32, out: *[128 * 128]f32) void {
    var t: [128][128]f32 = undefined;
    for (0..128) |k| for (0..128) |c| {
        t[k][c] = e4m3Value(codes[(bi * 128 + k) * cols + bj * 128 + c]);
    };
    var x: [128][128]f32 = undefined;
    for (0..128) |i| {
        for (0..128) |c| {
            var acc: f32 = 0;
            for (0..128) |k| acc = @mulAdd(f32, had(i, k), t[k][c], acc);
            x[i][c] = acc * suh[bi * 128 + i];
        }
    }
    for (0..128) |r| for (0..128) |j| {
        var acc: f32 = 0;
        for (0..128) |m| acc = @mulAdd(f32, x[r][m], had(m, j), acc);
        out[r * 128 + j] = acc * svh[bj * 128 + j];
    };
}

/// W in fp32 exactly as the reference loader computes it (the decode the format's authors check against).
pub fn reconstructF32(codes: []const u8, rows: usize, cols: usize, suh: []const f32, svh: []const f32, out: []f32) !void {
    if (rows % 128 != 0 or cols % 128 != 0 or codes.len != rows * cols or out.len != codes.len or
        suh.len != rows or svh.len != cols) return error.BadExlr8Reconstruct;
    var block: [128 * 128]f32 = undefined;
    for (0..rows / 128) |bi| for (0..cols / 128) |bj| {
        reconstructBlockF32(codes, cols, bi, bj, suh, svh, &block);
        for (0..128) |r| @memcpy(out[(bi * 128 + r) * cols + bj * 128 ..][0..128], block[r * 128 ..][0..128]);
    };
}

/// The sum of |T| over each 128 x 128 block: a bound's scale for fp32 reconstructions of that block.
pub fn blockAbs(codes: []const u8, rows: usize, cols: usize, out: []f64) void {
    @memset(out, 0);
    for (0..rows) |r| for (0..cols) |c| {
        out[(r / 128) * (cols / 128) + c / 128] += @abs(e4m3Value(codes[r * cols + c]));
    };
}

test "E4M3 rounding is nearest even with saturation, and codes decode to their values" {
    for (0..256) |c| {
        const code: u8 = @intCast(c);
        const v = e4m3Value(code);
        if (std.math.isNan(v)) continue;
        try std.testing.expectEqual(code, e4m3FromF32(v));
    }
    try std.testing.expectEqual(@as(u8, 0x7E), e4m3FromF32(1e9));
    try std.testing.expectEqual(@as(u8, 0xFE), e4m3FromF32(-std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 0x7E), e4m3FromF32(463.9));
    try std.testing.expectEqual(@as(u8, 0x00), e4m3FromF32(0x1p-10)); // a tie between 0 and 2^-9 goes to 0
    try std.testing.expectEqual(@as(u8, 0x02), e4m3FromF32(3 * 0x1p-10)); // a tie between 1 and 2 steps goes to 2
    try std.testing.expectEqual(@as(u8, 0x80), e4m3FromF32(-0x1p-12));
    try std.testing.expectEqual(@as(u8, 0x08), e4m3FromF32(31 * 0x1p-11)); // 7.75 steps round up to 2^-6
    try std.testing.expectEqual(@as(u8, 0x38), e4m3FromF32(1.0625)); // 1 + 1/16 is a tie: down to 1.0
    try std.testing.expectEqual(@as(u8, 0x3A), e4m3FromF32(1.1875)); // 1 + 3/16 ties up to 1.25
}

test "word-based state extraction equals the scalar EXL3 bitstream at K2, K3 and K4" {
    var prng = std.Random.DefaultPrng.init(0x45584C5238);
    var bytes: [128]u8 align(4) = undefined;
    for ([_]u8{ 2, 3, 4 }) |k| for (0..6) |_| {
        prng.random().bytes(&bytes);
        const words = std.mem.bytesAsSlice(i16, @as([]align(2) const u8, @alignCast(bytes[0 .. 32 * @as(usize, k)])));
        var want: [256]u32 = undefined;
        exl3.tileStates(.{ .halves = 2 * @as(u16, k) }, words, &want);
        for (0..256) |p| try std.testing.expectEqual(@as(u16, @intCast(want[p])), tileState(k, bytes[0 .. 32 * @as(usize, k)], p));
    };
}

test "a reconstruction of one sign-only block is the transform of T, divided by 128" {
    const gpa = std.testing.allocator;
    const codes = try gpa.alloc(u8, 128 * 128);
    defer gpa.free(codes);
    @memset(codes, 0);
    codes[3 * 128 + 5] = e4m3FromF32(2.0);
    const ones: [128]f32 = @splat(1);
    const w = try gpa.alloc(f64, 128 * 128);
    defer gpa.free(w);
    try reconstruct(codes, 128, 128, &ones, &ones, w);
    for (0..128) |i| for (0..128) |j| {
        const sign: f64 = if ((@popCount(i & 3) + @popCount(j & 5)) % 2 == 0) 1 else -1;
        try std.testing.expectEqual(sign * 2.0 / 128.0, w[i * 128 + j]);
    };
}
