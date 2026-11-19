//! The DSpark drafter's own FP8 copy of the shared vocabulary head: E4M3 codes and E8M0 scales a 32x32 block.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const weights = @import("weights.zig");

const Config = host.config.Config;
const Pack = host.pack.Pack;

/// The drafter's head in one buffer: codes [vocab][hidden], then scales [vocab / 32][hidden / 32].
pub const Head = struct {
    buf: mtl.Buffer,
    fp8: weights.Fp8,

    pub fn deinit(h: *Head) void {
        h.buf.deinit();
    }
};

/// The release's bf16 head quantized for the drafter's rows; null when a dimension is not a whole number of 32s.
pub fn load(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, c: *const Config) !?Head {
    const n: usize = c.vocab;
    const k: usize = c.hidden;
    if (n % 32 != 0 or k % 32 != 0) return null;
    const info = try pack.need("head.weight");
    if (!info.is(.bf16, &.{ n, k })) return error.BadNativeTensor;
    const src = try a.alloc(u16, n * k);
    defer a.free(src);
    try pack.read(io, info, .{ .offset = 0, .len = info.nbytes }, std.mem.sliceAsBytes(src));
    const soff = std.mem.alignForward(usize, n * k, 256);
    const sbytes = n / 32 * (k / 32);
    const buf = try device.buffer(soff + sbytes, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const out = buf.contents();
    quantize(src, n, k, out[0 .. n * k], out[soff..][0..sbytes]);
    return .{ .buf = buf, .fp8 = .{ .w = .{ .buf = buf }, .s = .{ .buf = buf, .off = soff }, .k = @intCast(k), .n = @intCast(n) } };
}

const max_workers = 16;

/// bf16 w [n][k] into E4M3 codes [n][k] and E8M0 scales [n / 32][k / 32]; workers take whole block rows.
pub fn quantize(w: []const u16, n: usize, k: usize, codes: []u8, scales: []u8) void {
    const rows = n / 32;
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, rows / 64));
    const per = (rows + workers - 1) / workers;
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |i| threads[i] = std.Thread.spawn(.{}, blockRows, .{ w, k, codes, scales, @min(rows, i * per), @min(rows, (i + 1) * per) }) catch null;
    blockRows(w, k, codes, scales, 0, @min(rows, per));
    for (1..workers) |i| if (threads[i]) |t| t.join() else blockRows(w, k, codes, scales, @min(rows, i * per), @min(rows, (i + 1) * per));
}

/// Block rows [first, last): each 32x32 block's scale, then its codes.
fn blockRows(w: []const u16, k: usize, codes: []u8, scales: []u8, first: usize, last: usize) void {
    for (first..last) |br| for (0..k / 32) |bc| {
        var amax: f32 = 0;
        for (0..32) |r| for (w[(br * 32 + r) * k + bc * 32 ..][0..32]) |b| {
            amax = @max(amax, @abs(bf16(b)));
        };
        const e = scaleExp(amax);
        scales[br * (k / 32) + bc] = @intCast(e + 127);
        for (0..32) |r| {
            const at = (br * 32 + r) * k + bc * 32;
            for (w[at..][0..32], codes[at..][0..32]) |b, *q| q.* = e4m3(std.math.ldexp(bf16(b), -e));
        }
    };
}

fn bf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The least power of two 2^e with amax <= 448 * 2^e, e in [-126, 120] (E8M0 codes 1 to 247; 0 for a zero block).
fn scaleExp(amax: f32) i32 {
    if (!(amax > 0) or !std.math.isFinite(amax)) return 0;
    var e: i32 = @intFromFloat(@ceil(std.math.log2(amax / 448.0)));
    while (e < 120 and amax > std.math.ldexp(@as(f32, 448), e)) e += 1;
    while (e > -126 and amax <= std.math.ldexp(@as(f32, 448), e - 1)) e -= 1;
    return std.math.clamp(e, -126, 120);
}

/// E4M3 (the fn form: no infinity, 0x7F NaN) of v, nearest even, saturating at 448.
pub fn e4m3(v: f32) u8 {
    const bits: u32 = @bitCast(v);
    const sign: u8 = @intCast((bits >> 24) & 0x80);
    const a = bits & 0x7FFFFFFF;
    if (a >= 0x43E00000) return sign | 0x7E; // 448 and past it (and NaN, never in a finite head)
    if (a < 0x3C800000) {
        // under 2^-6: steps of 2^-9, so a * 512 rounded to even (8 is 2^-6 itself, code 0x08)
        const x = @as(f32, @bitCast(a)) * 512.0;
        const f = @floor(x);
        const d = x - f;
        const q: u8 = @intFromFloat(f);
        return sign | (q + @intFromBool(d > 0.5 or (d == 0.5 and q & 1 == 1)));
    }
    // three mantissa bits kept, nearest even; a carry moves the exponent up exactly
    const r = (a + 0x7FFFF + ((a >> 20) & 1)) >> 20;
    const code = ((r >> 3) - (127 - 7) * 1) << 3 | (r & 7);
    return sign | @as(u8, @intCast(@min(code, 0x7E)));
}

test "E4M3 codes round-trip every finite code and round ties to even" {
    const dec = host.engram_rows.e4m3;
    for (0..256) |i| {
        const code: u8 = @intCast(i);
        if (code & 0x7F == 0x7F) continue;
        try std.testing.expectEqual(code, e4m3(dec(code)));
    }
    try std.testing.expectEqual(@as(u8, 0x38), e4m3(1.0625)); // between 1 and 1.125: to 1
    try std.testing.expectEqual(@as(u8, 0x3A), e4m3(1.1875)); // between 1.125 and 1.25: to 1.25
    try std.testing.expectEqual(@as(u8, 0x7E), e4m3(1000));
    try std.testing.expectEqual(@as(u8, 0x00), e4m3(0x1p-11)); // a quarter step: 0
    try std.testing.expectEqual(@as(u8, 0x02), e4m3(0x1p-9 * 2.5)); // 2.5 steps: to 2
    try std.testing.expectEqual(@as(u8, 0x81), e4m3(-0x1p-9));
}

test "a block's scale puts its largest magnitude at or under 448, and the codes decode within half a step" {
    const a = std.testing.allocator;
    const n = 64;
    const k = 96;
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(3);
    for (w, 0..) |*b, i| {
        const v = (prng.random().float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), @as(i32, @intCast(i % 7)) - 9);
        b.* = @intCast(@as(u32, @bitCast(v)) >> 16);
    }
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    var scales: [(n / 32) * (k / 32)]u8 = undefined;
    quantize(w, n, k, codes, &scales);
    const one = try a.alloc(u8, n * k);
    defer a.free(one);
    var s1: [(n / 32) * (k / 32)]u8 = undefined;
    blockRows(w, k, one, &s1, 0, n / 32);
    try std.testing.expectEqualSlices(u8, one, codes);
    try std.testing.expectEqualSlices(u8, &s1, &scales);
    for (0..n) |r| for (0..k) |i| {
        const e = @as(i32, scales[(r / 32) * (k / 32) + i / 32]) - 127;
        const got = host.engram_rows.e4m3(codes[r * k + i]) * std.math.ldexp(@as(f32, 1), e);
        const want = bf16(w[r * k + i]);
        try std.testing.expect(@abs(want) <= 448 * std.math.ldexp(@as(f32, 1), e));
        try std.testing.expect(@abs(got - want) <= @max(@abs(want) / 16, std.math.ldexp(@as(f32, 1), e - 10)));
    };
}
