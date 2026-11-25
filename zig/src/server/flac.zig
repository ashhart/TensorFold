//! A 16-bit FLAC stream of a speech take: STREAMINFO, 4096-sample frames, fixed-predictor subframes, Rice residuals.
const std = @import("std");
const quantize16 = @import("audio_out.zig").quantize16;
const Allocator = std.mem.Allocator;

pub const block_size = 4096;

/// Big-endian bit packing into a growing byte list.
pub const Bits = struct {
    out: *std.ArrayList(u8),
    a: Allocator,
    byte: u8 = 0,
    used: u4 = 0,

    pub fn put(w: *Bits, value: u64, count: u7) Allocator.Error!void {
        var left = count;
        while (left > 0) {
            const take: u7 = @min(left, 8 - @as(u7, w.used));
            left -= take;
            const part: u8 = @truncate((value >> @intCast(left)) & ((@as(u64, 1) << @intCast(take)) - 1));
            w.byte = if (take == 8) part else (w.byte << @intCast(take)) | part;
            w.used += @intCast(take);
            if (w.used == 8) {
                try w.out.append(w.a, w.byte);
                w.byte = 0;
                w.used = 0;
            }
        }
    }

    pub fn zeros(w: *Bits, count: u64) Allocator.Error!void {
        var left = count;
        while (left > 0) {
            const n: u7 = @intCast(@min(left, 56));
            try w.put(0, n);
            left -= n;
        }
    }

    /// Pads the last byte with zero bits.
    pub fn flush(w: *Bits) Allocator.Error!void {
        if (w.used > 0) try w.put(0, 8 - @as(u7, w.used));
    }
};

pub fn crc8(bytes: []const u8) u8 {
    var c: u8 = 0;
    for (bytes) |b| {
        c ^= b;
        for (0..8) |_| c = if (c & 0x80 != 0) (c << 1) ^ 0x07 else c << 1;
    }
    return c;
}

pub fn crc16(bytes: []const u8) u16 {
    var c: u16 = 0;
    for (bytes) |b| {
        c ^= @as(u16, b) << 8;
        for (0..8) |_| c = if (c & 0x8000 != 0) (c << 1) ^ 0x8005 else c << 1;
    }
    return c;
}

/// The frame header's rate code: a listed rate, else 16 bits of Hz or tens of Hz, else STREAMINFO's (0).
fn rateCode(rate: u32) u4 {
    return switch (rate) {
        88200 => 1,
        176400 => 2,
        192000 => 3,
        8000 => 4,
        16000 => 5,
        22050 => 6,
        24000 => 7,
        32000 => 8,
        44100 => 9,
        48000 => 10,
        96000 => 11,
        else => if (rate <= 0xFFFF) 13 else if (rate % 10 == 0 and rate / 10 <= 0xFFFF) 14 else 0,
    };
}

/// A frame number in FLAC's UTF-8-like coding, in the fewest bytes (1 to 7 for up to 36 bits).
pub fn putNumber(w: *Bits, n: u64) Allocator.Error!void {
    if (n < 0x80) return w.put(n, 8);
    var extra: u6 = 1; // continuation bytes; the lead byte keeps 6 - extra bits
    while (n >> @intCast(5 * @as(u7, extra) + 6) != 0) extra += 1;
    const lead: u8 = @truncate(@as(u16, 0xFF00) >> @intCast(extra + 1));
    try w.put(lead | @as(u8, @intCast(n >> @intCast(6 * @as(u7, extra)))), 8);
    var i = extra;
    while (i > 0) {
        i -= 1;
        try w.put(0x80 | ((n >> @intCast(6 * @as(u7, i))) & 0x3F), 8);
    }
}

pub fn predict(s: []const i16, order: usize, i: usize) i64 {
    return switch (order) {
        0 => 0,
        1 => s[i - 1],
        2 => 2 * @as(i64, s[i - 1]) - s[i - 2],
        3 => 3 * @as(i64, s[i - 1]) - 3 * @as(i64, s[i - 2]) + s[i - 3],
        4 => 4 * @as(i64, s[i - 1]) - 6 * @as(i64, s[i - 2]) + 4 * @as(i64, s[i - 3]) - s[i - 4],
        else => unreachable,
    };
}

fn fold(r: i64) u64 {
    return if (r >= 0) @intCast(2 * r) else @intCast(-2 * r - 1);
}

/// One channel's subframe: constant, the cheapest fixed order (0-4) and Rice parameter, or verbatim.
fn subframe(w: *Bits, s: []const i16) Allocator.Error!void {
    if (std.mem.allEqual(i16, s, s[0])) {
        try w.put(0, 8);
        return w.put(@as(u16, @bitCast(s[0])), 16);
    }
    var best: struct { bits: u64, order: usize = 0, k: u6 = 0, fixed: bool = false } = .{ .bits = 8 + 16 * @as(u64, s.len) };
    for (0..@min(4, s.len - 1) + 1) |order| for (0..15) |k| {
        var bits: u64 = 8 + 16 * order + 2 + 4 + 4;
        for (order..s.len) |i| {
            bits += (fold(s[i] - predict(s, order, i)) >> @intCast(k)) + 1 + k;
            if (bits >= best.bits) break;
        }
        if (bits < best.bits) best = .{ .bits = bits, .order = order, .k = @intCast(k), .fixed = true };
    };
    if (!best.fixed) {
        try w.put(0b10, 8);
        for (s) |x| try w.put(@as(u16, @bitCast(x)), 16);
        return;
    }
    try w.put(@as(u64, 8 + best.order) << 1, 8);
    for (s[0..best.order]) |x| try w.put(@as(u16, @bitCast(x)), 16);
    try w.put(0, 2 + 4); // Rice coding with 4-bit parameters, partition order 0
    try w.put(best.k, 4);
    for (best.order..s.len) |i| {
        const u = fold(s[i] - predict(s, best.order, i));
        try w.zeros(u >> best.k);
        try w.put(1, 1);
        try w.put(u, best.k);
    }
}

/// The take quantized as the WAV reply quantizes it, each sample on every channel, as a FLAC stream.
pub fn encode(a: Allocator, take: []const f32, rate: u32, channels: u8) Allocator.Error![]u8 {
    const s = try a.alloc(i16, take.len);
    defer a.free(s);
    var md5 = std.crypto.hash.Md5.init(.{});
    for (take, s) |x, *q| {
        q.* = quantize16(x);
        const le = std.mem.toBytes(std.mem.nativeToLittle(i16, q.*));
        for (0..channels) |_| md5.update(&le);
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var w: Bits = .{ .out = &out, .a = a };
    try out.appendSlice(a, "fLaC\x80\x00\x00\x22"); // the last metadata block: STREAMINFO, 34 bytes
    const most: u16 = @intCast(std.math.clamp(s.len, 16, block_size));
    try w.put(most, 16); // smallest block but the last
    try w.put(most, 16);
    try w.put(0, 24 + 24); // frame sizes unknown
    try w.put(rate, 20);
    try w.put(channels - 1, 3);
    try w.put(16 - 1, 5);
    try w.put(s.len, 36);
    try out.appendSlice(a, &md5.finalResult());
    var at: usize = 0;
    var number: u64 = 0;
    while (at < s.len) : (number += 1) {
        const n = @min(block_size, s.len - at);
        const start = out.items.len;
        const code = rateCode(rate);
        try w.put(0x3FFE << 2, 16); // sync, reserved bit, fixed block size
        try w.put(if (n == block_size) 12 else 7, 4);
        try w.put(code, 4);
        try w.put(channels - 1, 4); // independent channels
        try w.put(4 << 1, 4); // 16 bits a sample, reserved bit
        try putNumber(&w, number);
        if (n != block_size) try w.put(n - 1, 16);
        if (code == 13) try w.put(rate, 16) else if (code == 14) try w.put(rate / 10, 16);
        try out.append(a, crc8(out.items[start..]));
        for (0..channels) |_| try subframe(&w, s[at..][0..n]);
        try w.flush();
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u16, crc16(out.items[start..]))));
        at += n;
    }
    return out.toOwnedSlice(a);
}

test {
    _ = @import("flac_test.zig");
}
