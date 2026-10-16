//! XXH64 from the public xxHash specification: EXLR8's per-half and per-block checksums (seed 0).

const std = @import("std");

const p1: u64 = 0x9E3779B185EBCA87;
const p2: u64 = 0xC2B2AE3D27D4EB4F;
const p3: u64 = 0x165667B19E3779F9;
const p4: u64 = 0x85EBCA77C2B2AE63;
const p5: u64 = 0x27D4EB2F165667C5;

fn round(acc: u64, lane: u64) u64 {
    return std.math.rotl(u64, acc +% lane *% p2, 31) *% p1;
}

fn merge(acc: u64, v: u64) u64 {
    return (acc ^ round(0, v)) *% p1 +% p4;
}

/// A streaming hash: update with any split of the input gives the one-shot value.
pub const Hasher = struct {
    v: [4]u64,
    seed: u64,
    buf: [32]u8 = undefined,
    held: usize = 0,
    total: u64 = 0,

    pub fn init(seed: u64) Hasher {
        return .{ .v = .{ seed +% p1 +% p2, seed +% p2, seed, seed -% p1 }, .seed = seed };
    }

    fn stripe(h: *Hasher, s: *const [32]u8) void {
        inline for (0..4) |i| h.v[i] = round(h.v[i], std.mem.readInt(u64, s[i * 8 ..][0..8], .little));
    }

    pub fn update(h: *Hasher, bytes: []const u8) void {
        var in = bytes;
        h.total += in.len;
        if (h.held > 0) {
            const take = @min(32 - h.held, in.len);
            @memcpy(h.buf[h.held..][0..take], in[0..take]);
            h.held += take;
            in = in[take..];
            if (h.held < 32) return;
            h.stripe(&h.buf);
            h.held = 0;
        }
        while (in.len >= 32) : (in = in[32..]) h.stripe(in[0..32]);
        @memcpy(h.buf[0..in.len], in);
        h.held = in.len;
    }

    pub fn final(h: *const Hasher) u64 {
        var acc: u64 = if (h.total >= 32)
            std.math.rotl(u64, h.v[0], 1) +% std.math.rotl(u64, h.v[1], 7) +% std.math.rotl(u64, h.v[2], 12) +% std.math.rotl(u64, h.v[3], 18)
        else
            h.seed +% p5;
        if (h.total >= 32) for (h.v) |v| {
            acc = merge(acc, v);
        };
        acc +%= h.total;
        var tail: []const u8 = h.buf[0..h.held];
        while (tail.len >= 8) : (tail = tail[8..]) {
            acc ^= round(0, std.mem.readInt(u64, tail[0..8], .little));
            acc = std.math.rotl(u64, acc, 27) *% p1 +% p4;
        }
        if (tail.len >= 4) {
            acc ^= @as(u64, std.mem.readInt(u32, tail[0..4], .little)) *% p1;
            acc = std.math.rotl(u64, acc, 23) *% p2 +% p3;
            tail = tail[4..];
        }
        for (tail) |b| {
            acc ^= @as(u64, b) *% p5;
            acc = std.math.rotl(u64, acc, 11) *% p1;
        }
        acc ^= acc >> 33;
        acc *%= p2;
        acc ^= acc >> 29;
        acc *%= p3;
        return acc ^ (acc >> 32);
    }
};

pub fn hash(seed: u64, bytes: []const u8) u64 {
    var h = Hasher.init(seed);
    h.update(bytes);
    return h.final();
}

test "the empty input and Zig's own XXH64 agree with ours at every length and split" {
    try std.testing.expectEqual(@as(u64, 0xEF46DB3751D8E999), hash(0, ""));
    var data: [300]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    for (0..data.len) |n| for ([_]u64{ 0, 1, 0x9E3779B97F4A7C15 }) |seed| {
        const want = std.hash.XxHash64.hash(seed, data[0..n]);
        try std.testing.expectEqual(want, hash(seed, data[0..n]));
        var h = Hasher.init(seed);
        var at: usize = 0;
        var step: usize = 1;
        while (at < n) : (step = step % 37 + 3) {
            const take = @min(step, n - at);
            h.update(data[at..][0..take]);
            at += take;
        }
        try std.testing.expectEqual(want, h.final());
    };
}
