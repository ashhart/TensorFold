//! Huffman JPEG (sequential or progressive) to RGBA as Pillow's libjpeg-turbo decodes it: islow IDCT, fancy upsampling.

const std = @import("std");
const Allocator = std.mem.Allocator;
const image = @import("image.zig");

/// The natural index of each zigzag position; 16 more entries absorb a corrupt run past the block, as libjpeg's do.
const zigzag = [64]u8{ 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 } ++ @as([16]u8, @splat(63));

const fast_bits = 9;

const Huffman = struct {
    fast: [1 << fast_bits]u16 = @splat(0), // length << 8 | symbol for codes of at most fast_bits bits, else 0
    maxcode: [17]i32 = @splat(-1), // the largest code of each length 1..16; -1 when none (or no table)
    offset: [17]i32 = @splat(0), // a code's symbol index less the code
    symbols: [256]u8 = @splat(0),

    fn init(counts: *const [16]u8, symbols: []const u8) error{Invalid}!Huffman {
        var h: Huffman = .{};
        @memcpy(h.symbols[0..symbols.len], symbols);
        var code: i32 = 0;
        var k: i32 = 0;
        for (1..17) |len| {
            h.offset[len] = k - code;
            for (0..counts[len - 1]) |_| {
                if (code + 1 >= @as(i32, 1) << @intCast(len)) return error.Invalid;
                if (len <= fast_bits) {
                    const shift: u4 = @intCast(fast_bits - len);
                    const base = @as(usize, @intCast(code)) << shift;
                    @memset(h.fast[base..][0 .. @as(usize, 1) << shift], @intCast(len << 8 | h.symbols[@intCast(k)]));
                }
                code += 1;
                k += 1;
            }
            if (counts[len - 1] > 0) h.maxcode[len] = code - 1;
            code <<= 1;
        }
        return h;
    }
};

/// Entropy-coded bits: stuffed 0xFF 0x00 read as 0xFF; at a marker or the end, zeros, as libjpeg pads.
const Bits = struct {
    data: []const u8,
    at: usize,
    acc: u64 = 0, // left-aligned
    n: u7 = 0,

    fn fill(b: *Bits) void {
        while (b.n <= 56) : (b.n += 8) {
            var byte: u64 = 0;
            if (b.at < b.data.len and b.data[b.at] != 0xFF) {
                byte = b.data[b.at];
                b.at += 1;
            } else if (b.at + 1 < b.data.len and b.data[b.at] == 0xFF and b.data[b.at + 1] == 0) {
                byte = 0xFF;
                b.at += 2;
            }
            b.acc |= byte << @intCast(56 - b.n);
        }
    }

    fn get(b: *Bits, k: u5) u32 {
        if (k == 0) return 0;
        b.fill();
        const v: u32 = @intCast(b.acc >> @intCast(64 - @as(u7, k)));
        b.acc <<= k;
        b.n -= k;
        return v;
    }

    fn symbol(b: *Bits, h: *const Huffman) error{Invalid}!u8 {
        b.fill();
        const f = h.fast[@intCast(b.acc >> (64 - fast_bits))];
        var len: u5 = @intCast(f >> 8);
        var sym: u8 = @truncate(f);
        if (f == 0) {
            len = fast_bits + 1;
            while (len <= 16 and @as(i32, @intCast(b.acc >> @intCast(64 - @as(u7, len)))) > h.maxcode[len]) len += 1;
            if (len > 16) return error.Invalid;
            const at = @as(i32, @intCast(b.acc >> @intCast(64 - @as(u7, len)))) + h.offset[len];
            if (at < 0 or at >= h.symbols.len) return error.Invalid;
            sym = h.symbols[@intCast(at)];
        }
        b.acc <<= len;
        b.n -= len;
        return sym;
    }

    /// A magnitude category's `s` bits as the signed value they code.
    fn signed(b: *Bits, s: u8) error{Invalid}!i32 {
        if (s > 15) return error.Invalid;
        if (s == 0) return 0;
        const v: i32 = @intCast(b.get(@intCast(s)));
        return if (v < @as(i32, 1) << @intCast(s - 1)) v - (@as(i32, 1) << @intCast(s)) + 1 else v;
    }
};

const Component = struct {
    id: u8,
    h: u8,
    v: u8,
    tq: u8,
    quant: ?[64]u16 = null, // the table in force at the component's first scan
    width: usize = 0, // samples, before upsampling
    height: usize = 0,
    bw: usize = 0, // blocks stored a row and rows of blocks: whole MCUs
    bh: usize = 0,
    coefs: []i16 = &.{},
    pred: i32 = 0,
    dc: u2 = 0,
    ac: u2 = 0,

    fn block(c: *Component, bx: usize, by: usize) *[64]i16 {
        return c.coefs[(by * c.bw + bx) * 64 ..][0..64];
    }
};

const Decoder = struct {
    data: []const u8,
    quant: [4][64]u16 = @splat(@splat(0)),
    dc: [4]Huffman = @splat(.{}),
    ac: [4]Huffman = @splat(.{}),
    comps: [3]Component = undefined,
    ncomp: usize = 0,
    width: usize = 0,
    height: usize = 0,
    hmax: u8 = 1,
    vmax: u8 = 1,
    mcux: usize = 0,
    mcuy: usize = 0,
    progressive: bool = false,
    restart: u16 = 0,
    jfif: bool = false,
    adobe: ?u8 = null,
    exif: ?[]const u8 = null,
    eobrun: u32 = 0,
};

pub fn decode(a: Allocator, data: []const u8) image.DecodeError!image.Decoded {
    var d: Decoder = .{ .data = data };
    defer for (d.comps[0..d.ncomp]) |c| a.free(c.coefs);
    var at: usize = 2;
    while (true) {
        while (at < data.len and data[at] != 0xFF) at += 1;
        while (at < data.len and data[at] == 0xFF) at += 1;
        if (at >= data.len) return error.Invalid;
        const m = data[at];
        at += 1;
        if (m == 0xD9) break;
        if (m >= 0xD0 and m <= 0xD8 or m == 0x01) continue;
        if (data.len - at < 2) return error.Invalid;
        const len = std.mem.readInt(u16, data[at..][0..2], .big);
        if (len < 2 or len > data.len - at) return error.Invalid;
        const body = data[at + 2 .. at + len];
        at += len;
        switch (m) {
            0xC0, 0xC1, 0xC2 => try frame(a, &d, body, m == 0xC2),
            0xC4 => try tables(&d, body),
            0xDB => try quantTables(&d, body),
            0xDD => d.restart = if (body.len >= 2) std.mem.readInt(u16, body[0..2], .big) else return error.Invalid,
            0xDA => at = try scan(&d, body, at),
            0xE0 => d.jfif = d.jfif or (body.len >= 14 and std.mem.startsWith(u8, body, "JFIF\x00")),
            0xE1 => if (d.exif == null and std.mem.startsWith(u8, body, "Exif\x00\x00")) {
                d.exif = body;
            },
            0xEE => if (d.adobe == null and body.len >= 12 and std.mem.startsWith(u8, body, "Adobe")) {
                d.adobe = body[11];
            },
            0xC3, 0xC5...0xC7, 0xC8...0xCB, 0xCD...0xCF => return error.Invalid,
            else => {},
        }
    }
    if (d.ncomp == 0) return error.Invalid;
    for (d.comps[0..d.ncomp]) |*c| if (c.quant == null) return error.Invalid;
    return .{ .width = d.width, .height = d.height, .rgba = try pixels(a, &d), .exif = d.exif };
}

fn frame(a: Allocator, d: *Decoder, body: []const u8, progressive: bool) image.DecodeError!void {
    if (d.ncomp != 0 or body.len < 6 or body[0] != 8) return error.Invalid;
    const n = body[5];
    if (n != 1 and n != 3 or body.len < 6 + 3 * @as(usize, n)) return error.Invalid;
    d.height = std.mem.readInt(u16, body[1..3], .big);
    d.width = std.mem.readInt(u16, body[3..5], .big);
    try image.fits(d.width, d.height);
    d.progressive = progressive;
    const comps = d.comps[0..n];
    for (comps, 0..) |*c, i| {
        const p = body[6 + 3 * i ..][0..3];
        c.* = .{ .id = p[0], .h = p[1] >> 4, .v = p[1] & 15, .tq = p[2] };
        if (c.h < 1 or c.h > 4 or c.v < 1 or c.v > 4 or c.tq > 3) return error.Invalid;
        d.hmax = @max(d.hmax, c.h);
        d.vmax = @max(d.vmax, c.v);
    }
    d.mcux = std.math.divCeil(usize, d.width, 8 * @as(usize, d.hmax)) catch unreachable;
    d.mcuy = std.math.divCeil(usize, d.height, 8 * @as(usize, d.vmax)) catch unreachable;
    for (comps) |*c| {
        if (d.hmax % c.h != 0 or d.vmax % c.v != 0) return error.Invalid;
        c.width = std.math.divCeil(usize, d.width * c.h, d.hmax) catch unreachable;
        c.height = std.math.divCeil(usize, d.height * c.v, d.vmax) catch unreachable;
        c.bw = d.mcux * c.h;
        c.bh = d.mcuy * c.v;
    }
    for (comps) |*c| {
        c.coefs = try a.alloc(i16, c.bw * c.bh * 64);
        @memset(c.coefs, 0);
        d.ncomp += 1;
    }
}

fn tables(d: *Decoder, body: []const u8) error{Invalid}!void {
    var rest = body;
    while (rest.len > 0) {
        if (rest.len < 17 or rest[0] >> 4 > 1 or rest[0] & 15 > 3) return error.Invalid;
        const counts = rest[1..17];
        var n: usize = 0;
        for (counts) |c| n += c;
        if (n > 256 or rest.len < 17 + n) return error.Invalid;
        const h = try Huffman.init(counts, rest[17 .. 17 + n]);
        if (rest[0] >> 4 == 0) d.dc[rest[0] & 15] = h else d.ac[rest[0] & 15] = h;
        rest = rest[17 + n ..];
    }
}

fn quantTables(d: *Decoder, body: []const u8) error{Invalid}!void {
    var rest = body;
    while (rest.len > 0) {
        const wide = rest[0] >> 4 == 1;
        const size: usize = if (wide) 129 else 65;
        if (rest[0] >> 4 > 1 or rest[0] & 15 > 3 or rest.len < size) return error.Invalid;
        for (0..64) |k| d.quant[rest[0] & 15][zigzag[k]] = if (wide) std.mem.readInt(u16, rest[1 + 2 * k ..][0..2], .big) else rest[1 + k];
        rest = rest[size..];
    }
}

/// One scan's coefficients; returns where its entropy-coded data ends.
fn scan(d: *Decoder, body: []const u8, at: usize) error{Invalid}!usize {
    if (d.ncomp == 0 or body.len < 1) return error.Invalid;
    const ns = body[0];
    if (ns < 1 or ns > d.ncomp or body.len < 4 + 2 * @as(usize, ns)) return error.Invalid;
    var in: [3]*Component = undefined;
    var blocks: usize = 0;
    for (in[0..ns], 0..) |*p, i| {
        const sel = body[1 + 2 * i ..][0..2];
        p.* = for (d.comps[0..d.ncomp]) |*c| {
            if (c.id == sel[0]) break c;
        } else return error.Invalid;
        if (sel[1] >> 4 > 3 or sel[1] & 15 > 3) return error.Invalid;
        p.*.dc = @intCast(sel[1] >> 4);
        p.*.ac = @intCast(sel[1] & 15);
        p.*.pred = 0;
        if (p.*.quant == null) p.*.quant = d.quant[p.*.tq];
        blocks += @as(usize, p.*.h) * p.*.v;
    }
    const tail = body[1 + 2 * @as(usize, ns) ..][0..3];
    const ss, const se, const ah, const al = .{ tail[0], tail[1], tail[2] >> 4, tail[2] & 15 };
    if (ns > 1 and blocks > 10 or se > 63 or ss > se or ah > 13 or al > 13) return error.Invalid;
    if (d.progressive and (ss == 0) != (se == 0) or d.progressive and ss > 0 and ns != 1) return error.Invalid;
    d.eobrun = 0;
    var b: Bits = .{ .data = d.data, .at = at };
    const single = ns == 1;
    const cols = if (single) std.math.divCeil(usize, in[0].width, 8) catch unreachable else d.mcux;
    const rows = if (single) std.math.divCeil(usize, in[0].height, 8) catch unreachable else d.mcuy;
    for (0..rows * cols) |mcu| {
        if (d.restart > 0 and mcu > 0 and mcu % d.restart == 0) {
            b.acc, b.n = .{ 0, 0 };
            if (b.at + 1 < b.data.len and b.data[b.at] == 0xFF and b.data[b.at + 1] >= 0xD0 and b.data[b.at + 1] <= 0xD7) b.at += 2;
            for (in[0..ns]) |c| c.pred = 0;
            d.eobrun = 0;
        }
        const mx, const my = .{ mcu % cols, mcu / cols };
        if (single) {
            try coefficients(d, &b, in[0], in[0].block(mx, my), ss, se, ah, al);
        } else for (in[0..ns]) |c| for (0..c.v) |v| for (0..c.h) |h| {
            try coefficients(d, &b, c, c.block(mx * c.h + h, my * c.v + v), ss, se, ah, al);
        };
    }
    return b.at;
}

/// One block's share of a scan, as jdhuff.c (sequential) and jdphuff.c (progressive) decode it.
fn coefficients(d: *Decoder, b: *Bits, c: *Component, coef: *[64]i16, ss: u8, se: u8, ah: u8, al: u8) error{Invalid}!void {
    if (!d.progressive or ss == 0 and ah == 0) {
        c.pred +%= try b.signed(try b.symbol(&d.dc[c.dc]));
        coef[0] = @truncate(@as(i64, c.pred) << @intCast(if (d.progressive) al else 0));
    } else if (ss == 0) {
        if (b.get(1) != 0) coef[0] |= @as(i16, 1) << @intCast(al);
    }
    if (!d.progressive) {
        var k: usize = 1;
        while (k < 64) {
            const rs = try b.symbol(&d.ac[c.ac]);
            if (rs & 15 != 0) {
                k += rs >> 4;
                coef[zigzag[k]] = @intCast(try b.signed(rs & 15));
                k += 1;
            } else if (rs >> 4 == 15) k += 16 else break;
        }
        return;
    }
    if (ss == 0) return;
    if (ah == 0) {
        if (d.eobrun > 0) {
            d.eobrun -= 1;
            return;
        }
        var k: usize = ss;
        while (k <= se) : (k += 1) {
            const rs = try b.symbol(&d.ac[c.ac]);
            const r: u5 = @intCast(rs >> 4);
            if (rs & 15 != 0) {
                k += r;
                coef[zigzag[k]] = @truncate(@as(i64, try b.signed(rs & 15)) << @intCast(al));
            } else if (r == 15) k += 15 else {
                d.eobrun = (@as(u32, 1) << r) - 1 + b.get(r);
                break;
            }
        }
        return;
    }
    const p1 = @as(i16, 1) << @intCast(al);
    var k: usize = ss;
    if (d.eobrun == 0) while (k <= se) : (k += 1) {
        const rs = try b.symbol(&d.ac[c.ac]);
        var r: u5 = @intCast(rs >> 4);
        var value: i16 = 0;
        if (rs & 15 != 0) {
            value = if (b.get(1) != 0) p1 else -p1;
        } else if (r != 15) {
            d.eobrun = (@as(u32, 1) << r) + b.get(r);
            break;
        }
        while (k <= se) : (k += 1) {
            const v = &coef[zigzag[k]];
            if (v.* != 0) {
                if (b.get(1) != 0 and v.* & p1 == 0) v.* +%= if (v.* >= 0) p1 else -p1;
            } else {
                if (r == 0) break;
                r -= 1;
            }
        }
        if (value != 0) coef[zigzag[k]] = value;
    };
    if (d.eobrun > 0) {
        while (k <= se) : (k += 1) {
            const v = &coef[zigzag[k]];
            if (v.* != 0 and b.get(1) != 0 and v.* & p1 == 0) v.* +%= if (v.* >= 0) p1 else -p1;
        }
        d.eobrun -= 1;
    }
}

/// jidctint.c's islow IDCT of a dequantized block into 8 rows of `out`, saturated as the SIMD builds store it.
fn idct(coef: *const [64]i16, q: *const [64]u16, out: []u8, stride: usize) void {
    var ws: [64]i64 = undefined;
    for (0..8) |col| {
        var v: [8]i64 = undefined;
        for (&v, 0..) |*x, r| x.* = @as(i64, coef[r * 8 + col]) * q[r * 8 + col];
        for (idct1(v, 13 - 2), 0..) |x, r| ws[r * 8 + col] = x;
    }
    for (0..8) |r| for (idct1(ws[r * 8 ..][0..8].*, 13 + 2 + 3), 0..) |x, col| {
        out[r * stride + col] = @intCast(std.math.clamp(x + 128, 0, 255));
    };
}

/// One 8-point pass of jpeg_idct_islow, descaled by `shift` with rounding.
fn idct1(v: [8]i64, shift: u6) [8]i64 {
    const z1 = (v[2] + v[6]) * 4433;
    const e2 = z1 + v[6] * -15137;
    const e3 = z1 + v[2] * 6270;
    const e0 = (v[0] + v[4]) * 8192;
    const e1 = (v[0] - v[4]) * 8192;
    const t10, const t13, const t11, const t12 = .{ e0 + e3, e0 - e3, e1 + e2, e1 - e2 };
    const z5 = (v[7] + v[5] + v[3] + v[1]) * 9633;
    const o1 = (v[7] + v[1]) * -7373;
    const o2 = (v[5] + v[3]) * -20995;
    const o3 = (v[7] + v[3]) * -16069 + z5;
    const o4 = (v[5] + v[1]) * -3196 + z5;
    const t0 = v[7] * 2446 + o1 + o3;
    const t1 = v[5] * 16819 + o2 + o4;
    const t2 = v[3] * 25172 + o2 + o3;
    const t3 = v[1] * 12299 + o1 + o4;
    const half = @as(i64, 1) << (shift - 1);
    return .{ (t10 + t3 + half) >> shift, (t11 + t2 + half) >> shift, (t12 + t1 + half) >> shift, (t13 + t0 + half) >> shift, (t13 - t0 + half) >> shift, (t12 - t1 + half) >> shift, (t11 - t2 + half) >> shift, (t10 - t3 + half) >> shift };
}

/// jdcolor.c's YCbCr tables: Cr->R, Cb->B, and the two G terms (ONE_HALF folded into Cb's).
const ycc = blk: {
    @setEvalBranchQuota(10000);
    const fix = struct {
        fn f(x: f64) i32 {
            return @intFromFloat(x * 65536 + 0.5);
        }
    }.f;
    var t: [4][256]i32 = undefined;
    for (0..256) |i| {
        const x = @as(i32, @intCast(i)) - 128;
        t[0][i] = (fix(1.40200) * x + 32768) >> 16;
        t[1][i] = (fix(1.77200) * x + 32768) >> 16;
        t[2][i] = -fix(0.71414) * x;
        t[3][i] = -fix(0.34414) * x + 32768;
    }
    break :blk t;
};

/// Every block's IDCT, then each output row's upsampled components converted to RGBA.
fn pixels(a: Allocator, d: *Decoder) Allocator.Error![]u8 {
    var planes: [3][]u8 = undefined;
    var made: usize = 0;
    defer for (planes[0..made]) |p| a.free(p);
    for (d.comps[0..d.ncomp], planes[0..d.ncomp]) |*c, *p| {
        const stride = c.bw * 8;
        p.* = try a.alloc(u8, stride * c.bh * 8);
        made += 1;
        for (0..c.bh) |by| for (0..c.bw) |bx| idct(c.block(bx, by), &c.quant.?, p.*[(by * 8 * stride) + bx * 8 ..], stride);
    }
    const rgb = d.ncomp == 3 and !d.jfif and (if (d.adobe) |t| t == 0 else d.comps[0].id == 'R' and d.comps[1].id == 'G' and d.comps[2].id == 'B');
    const row = try a.alloc(u8, 3 * (d.width + 8));
    defer a.free(row);
    const out = try a.alloc(u8, d.width * d.height * 4);
    for (0..d.height) |y| {
        for (d.comps[0..d.ncomp], planes[0..d.ncomp], 0..) |*c, p, i| upsample(d, c, p, y, row[i * (d.width + 8) ..][0 .. d.width + 8]);
        const s0, const s1, const s2 = .{ row[0..d.width], row[d.width + 8 ..][0..d.width], row[2 * (d.width + 8) ..][0..d.width] };
        for (0..d.width) |x| {
            const y0: i32 = s0[x];
            out[(y * d.width + x) * 4 ..][0..4].* = if (d.ncomp == 1)
                .{ s0[x], s0[x], s0[x], 255 }
            else if (rgb)
                .{ s0[x], s1[x], s2[x], 255 }
            else
                .{ limit(y0 + ycc[0][s2[x]]), limit(y0 + ((ycc[3][s1[x]] + ycc[2][s2[x]]) >> 16)), limit(y0 + ycc[1][s1[x]]), 255 };
        }
    }
    return out;
}

fn limit(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// Output row `y` of a component as jdsample.c makes it: h2v1 and h2v2 triangle filters, h1v2's, else replication.
fn upsample(d: *const Decoder, c: *const Component, plane: []const u8, y: usize, out: []u8) void {
    const stride = c.bw * 8;
    const hx, const vx = .{ d.hmax / c.h, d.vmax / c.v };
    const dw = c.width;
    if (hx == 1 and vx == 1) {
        @memcpy(out[0..d.width], plane[y * stride ..][0..d.width]);
    } else if (hx == 2 and vx == 1 and dw > 2) {
        const in = plane[y * stride ..][0..dw];
        out[0] = in[0];
        out[1] = @intCast((@as(u16, in[0]) * 3 + in[1] + 2) >> 2);
        for (1..dw - 1) |i| {
            out[2 * i] = @intCast((@as(u16, in[i]) * 3 + in[i - 1] + 1) >> 2);
            out[2 * i + 1] = @intCast((@as(u16, in[i]) * 3 + in[i + 1] + 2) >> 2);
        }
        out[2 * dw - 2] = @intCast((@as(u16, in[dw - 1]) * 3 + in[dw - 2] + 1) >> 2);
        out[2 * dw - 1] = in[dw - 1];
    } else if (vx == 2 and (hx == 1 or hx == 2 and dw > 2)) {
        const r = y / 2;
        const near = plane[r * stride ..][0..dw];
        const far = plane[(if (y % 2 == 0) r -| 1 else @min(r + 1, c.height - 1)) * stride ..][0..dw];
        if (hx == 1) {
            for (out[0..dw], near, far) |*o, n, f| o.* = @intCast((@as(u16, n) * 3 + f + 1 + y % 2) >> 2);
            return;
        }
        var sums: [3]u16 = .{ 0, @as(u16, near[0]) * 3 + far[0], @as(u16, near[1]) * 3 + far[1] };
        out[0] = @intCast((sums[1] * 4 + 8) >> 4);
        out[1] = @intCast((sums[1] * 3 + sums[2] + 7) >> 4);
        for (1..dw - 1) |i| {
            sums = .{ sums[1], sums[2], @as(u16, near[i + 1]) * 3 + far[i + 1] };
            out[2 * i] = @intCast((sums[1] * 3 + sums[0] + 8) >> 4);
            out[2 * i + 1] = @intCast((sums[1] * 3 + sums[2] + 7) >> 4);
        }
        out[2 * dw - 2] = @intCast((sums[2] * 3 + sums[1] + 8) >> 4);
        out[2 * dw - 1] = @intCast((sums[2] * 4 + 7) >> 4);
    } else for (out[0..d.width], 0..) |*o, x| {
        o.* = plane[(y / vx) * stride + x / hx];
    }
}
