//! FLAC streams decoded back to their 16-bit samples by a reader of the format's own: headers, CRCs, MD5 and subframes.
const std = @import("std");
const flac = @import("flac.zig");
const quantize16 = @import("audio_out.zig").quantize16;
const Allocator = std.mem.Allocator;

const Reader = struct {
    bytes: []const u8,
    at: usize = 0, // in bits

    fn bits(r: *Reader, count: u7) !u64 {
        var v: u64 = 0;
        for (0..count) |_| {
            if (r.at >= 8 * r.bytes.len) return error.Short;
            v = (v << 1) | ((r.bytes[r.at / 8] >> @intCast(7 - r.at % 8)) & 1);
            r.at += 1;
        }
        return v;
    }

    fn unary(r: *Reader) !u64 {
        var q: u64 = 0;
        while (try r.bits(1) == 0) q += 1;
        return q;
    }

    fn signed(r: *Reader) !i16 {
        return @bitCast(@as(u16, @intCast(try r.bits(16))));
    }

    fn byte(r: *Reader) usize {
        std.debug.assert(r.at % 8 == 0);
        return r.at / 8;
    }
};

const Decoded = struct { rate: u32, channels: u8, samples: []i16 };

/// The interleaved samples of a stream `flac.encode` wrote, every check of the format applied on the way.
fn decode(a: Allocator, bytes: []const u8) !Decoded {
    if (bytes.len < 42 or !std.mem.eql(u8, bytes[0..8], "fLaC\x80\x00\x00\x22")) return error.BadStream;
    var r: Reader = .{ .bytes = bytes, .at = 8 * 8 };
    const smallest = try r.bits(16);
    const largest = try r.bits(16);
    _ = try r.bits(48);
    const rate: u32 = @intCast(try r.bits(20));
    const channels: u8 = @intCast(try r.bits(3) + 1);
    if (try r.bits(5) != 15) return error.BadStream;
    const total: usize = @intCast(try r.bits(36));
    const digest = bytes[r.byte()..][0..16];
    r.at += 16 * 8;
    const out = try a.alloc(i16, total * channels);
    errdefer a.free(out);
    const block = try a.alloc(i16, flac.block_size);
    defer a.free(block);
    var done: usize = 0;
    var number: u64 = 0;
    while (r.byte() < bytes.len) : (number += 1) {
        const start = r.byte();
        if (try r.bits(16) != 0xFFF8) return error.BadSync;
        const size_code = try r.bits(4);
        const rate_code = try r.bits(4);
        if (try r.bits(4) != channels - 1 or try r.bits(4) != 8) return error.BadFrame;
        const lead_byte = bytes[r.byte()];
        const length: usize = if (lead_byte < 0x80) 1 else @clz(~lead_byte);
        var got: u64 = try r.bits(8) & (if (length == 1) 0x7F else @as(u64, 0xFF) >> @intCast(length + 1));
        for (1..length) |_| {
            const next = try r.bits(8);
            if (next & 0xC0 != 0x80) return error.BadNumber;
            got = (got << 6) | (next & 0x3F);
        }
        const shortest: usize = if (got < 0x80) 1 else if (got < 0x800) 2 else if (got < 0x10000) 3 else if (got < 0x200000) 4 else 5;
        if (got != number or length != shortest) return error.BadNumber;
        const n: usize = switch (size_code) {
            12 => flac.block_size,
            7 => @intCast(try r.bits(16) + 1),
            else => return error.BadFrame,
        };
        if (rate_code == 13 or rate_code == 14) _ = try r.bits(16);
        if (try r.bits(8) != flac.crc8(bytes[start .. r.byte() - 1])) return error.BadCrc;
        for (0..channels) |c| {
            const head = try r.bits(8);
            const kind = head >> 1;
            const s = block[0..n];
            if (kind == 0) {
                @memset(s, try r.signed());
            } else if (kind == 1) {
                for (s) |*x| x.* = try r.signed();
            } else if (kind >= 8 and kind <= 12) {
                const order: usize = @intCast(kind - 8);
                for (s[0..order]) |*x| x.* = try r.signed();
                if (try r.bits(6) != 0) return error.BadResidual;
                const k: u7 = @intCast(try r.bits(4));
                for (order..n) |i| {
                    const u = (try r.unary() << @intCast(k)) | try r.bits(k);
                    const residual: i64 = if (u & 1 == 0) @intCast(u >> 1) else -@as(i64, @intCast(u >> 1)) - 1;
                    s[i] = @intCast(flac.predict(s, order, i) + residual);
                }
            } else return error.BadSubframe;
            for (s, 0..) |x, i| out[(done + i) * channels + c] = x;
        }
        r.at = (r.at + 7) / 8 * 8;
        if (try r.bits(16) != flac.crc16(bytes[start .. r.byte() - 2])) return error.BadCrc;
        if (n != flac.block_size and done + n != total) return error.BadFrame;
        if (n < smallest and done + n != total or n > largest) return error.BadFrame;
        done += n;
    }
    if (done != total) return error.BadFrame;
    var md5 = std.crypto.hash.Md5.init(.{});
    md5.update(std.mem.sliceAsBytes(out));
    var want: [16]u8 = undefined;
    md5.final(&want);
    if (!std.mem.eql(u8, &want, digest)) return error.BadDigest;
    return .{ .rate = rate, .channels = channels, .samples = out };
}

fn expectRoundTrip(take: []const f32, rate: u32, channels: u8) !void {
    const a = std.testing.allocator;
    const bytes = try flac.encode(a, take, rate, channels);
    defer a.free(bytes);
    const got = try decode(a, bytes);
    defer a.free(got.samples);
    try std.testing.expectEqual(rate, got.rate);
    try std.testing.expectEqual(channels, got.channels);
    try std.testing.expectEqual(take.len * channels, got.samples.len);
    for (take, 0..) |x, i| for (0..channels) |c| try std.testing.expectEqual(quantize16(x), got.samples[i * channels + c]);
}

test "flac: silence, a tone, full-scale square waves and noise decode to the WAV reply's samples" {
    const a = std.testing.allocator;
    const take = try a.alloc(f32, 3 * flac.block_size + 917);
    defer a.free(take);
    var prng: std.Random.DefaultPrng = .init(7);
    for (take, 0..) |*x, i| x.* = switch (i / 3000) {
        0 => if (i < 700) 0 else @floatCast(0.7 * @sin(0.05 * @as(f64, @floatFromInt(i)))),
        1 => if (i % 2 == 0) 1.0 else -1.0,
        2 => prng.random().float(f32) * 2 - 1,
        else => @floatCast(0.3 * @sin(0.001 * @as(f64, @floatFromInt(i)))),
    };
    for ([_]u8{ 1, 2 }) |channels| for ([_]u32{ 44100, 22050, 11025, 192000 }) |rate| try expectRoundTrip(take, rate, channels);
    try expectRoundTrip(take[0..1], 24000, 1);
    try expectRoundTrip(take[0..15], 24000, 2);
    try expectRoundTrip(take[0..flac.block_size], 48000, 1);
    try expectRoundTrip(take[0..0], 44100, 1);
    // the WAV reply's own samples: speech.zig's in-place quantization of the same take
    const bytes = try flac.encode(a, take, 44100, 1);
    defer a.free(bytes);
    const got = try decode(a, bytes);
    defer a.free(got.samples);
    const copy = try a.dupe(f32, take);
    defer a.free(copy);
    try std.testing.expectEqualSlices(u8, try @import("speech.zig").quantizeInPlace(copy), std.mem.sliceAsBytes(got.samples));
}

test "flac: frame numbers take the fewest bytes, as UTF-8 would" {
    const a = std.testing.allocator;
    for ([_]struct { u64, []const u8 }{
        .{ 0x7F, "\x7F" },                      .{ 0x80, "\xC2\x80" },                              .{ 0x7FF, "\xDF\xBF" },
        .{ 0x800, "\xE0\xA0\x80" },             .{ 0xFFFF, "\xEF\xBF\xBF" },                        .{ 0x10000, "\xF0\x90\x80\x80" },
        .{ 0x3FFFFFF, "\xFB\xBF\xBF\xBF\xBF" }, .{ (1 << 36) - 1, "\xFE\xBF\xBF\xBF\xBF\xBF\xBF" },
    }) |case| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        var w: flac.Bits = .{ .out = &out, .a = a };
        try flac.putNumber(&w, case[0]);
        try std.testing.expectEqualSlices(u8, case[1], out.items);
    }
}

test "flac: a 30-second take at 44.1 kHz needs frame numbers past 0x7FF and still decodes" {
    const a = std.testing.allocator;
    const take = try a.alloc(f32, 0x801 * flac.block_size);
    defer a.free(take);
    for (take, 0..) |*x, i| x.* = @floatCast(0.25 * @sin(0.01 * @as(f64, @floatFromInt(i))));
    try expectRoundTrip(take, 44100, 1);
}
