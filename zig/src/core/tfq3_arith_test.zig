//! TFQ3's shared arithmetic against the copies it replaced (main 91dbadce65), kept here verbatim as the oracles.
const std = @import("std");
const f = @import("tfq3_format.zig");
const t = std.testing;
const max = std.math.maxInt;

/// Flash Next tfq3_check.zig:87 and GLM tfq3_check.zig:127 on main: one body, refusing as Tfq3Overflow.
fn mainMatWords(r: usize, c: usize, bits: usize) error{Tfq3Overflow}!usize {
    const rc = std.math.mul(usize, r, c) catch return error.Tfq3Overflow;
    return (std.math.mul(usize, rc, bits) catch return error.Tfq3Overflow) / 32;
}

/// CUDA tfq3_format.zig:40 on main.
fn mainValidBits(bits: u32) bool {
    return bits == 3 or bits == 4 or bits == 5;
}

/// Flash Next tfq3_check.zig:79, the inline loop in `header` on main.
fn mainFlashNextOverlap(head: usize, all: []const f.Span) bool {
    for (all, 0..) |a, i| {
        if (a.len > 0 and a.off < head) return true;
        for (all[i + 1 ..]) |b| if (a.len > 0 and b.len > 0 and a.off < b.off + b.len and b.off < a.off + a.len) return true;
    }
    return false;
}

/// GLM tfq3_check.zig:93 `disjoint` on main, true where it refused.
fn mainGlmOverlap(head: usize, spans: []const f.Span) bool {
    for (spans, 0..) |a, i| {
        if (a.len > 0 and a.off < head) return true;
        for (spans[i + 1 ..]) |b| {
            if (a.len == 0 or b.len == 0) continue;
            if (a.off < b.off + b.len and b.off < a.off + a.len) return true;
        }
    }
    return false;
}

const dims = [_]usize{ 0, 1, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 640, 2048, 2560, 4096, max(u16), max(u32) - 1, max(u32), 1 << 32, 1 << 33, max(usize) / 32, max(usize) / 5, max(usize) / 4, max(usize) / 3, max(usize) / 2, max(usize) - 1, max(usize) };
const widths = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 8, 16, 32, max(u32), max(usize) };

test "TFQ3 arithmetic: matWords equals main's checked body on every edge, null exactly where main refused" {
    for (dims) |r| for (dims) |c| for (widths) |bits| {
        const got = f.checkedMatWords(r, c, bits);
        if (mainMatWords(r, c, bits)) |want| try t.expectEqual(@as(?usize, want), got) else |_| try t.expectEqual(@as(?usize, null), got);
    };
    try t.expectEqual(@as(?usize, 204800), f.checkedMatWords(640, 2560, 4)); // a Flash Next gate or up projection at 4 bits
    try t.expectEqual(@as(?usize, 256000), f.checkedMatWords(2560, 640, 5));
    try t.expectEqual(@as(?usize, 786432), f.checkedMatWords(2048, 4096, 3)); // a GLM projection at 3 bits
    try t.expectEqual(@as(?usize, 0), f.checkedMatWords(1, 7, 4)); // floor: 28 bits are no whole word
    try t.expectEqual(@as(?usize, 1), f.checkedMatWords(1, 8, 4));
    try t.expectEqual(@as(?usize, 274), f.checkedMatWords(17, 129, 4)); // off the 16 x 128 grid (CUDA's tensorWords: 256)
    try t.expectEqual(@as(?usize, null), f.checkedMatWords(1 << 33, 1 << 33, 0)); // R * C overflows: refused though bits is 0
    try t.expectEqual(@as(?usize, null), f.checkedMatWords(1 << 62, 1, 4)); // the second product overflows; / 32 would fit
    try t.expectEqual(@as(?usize, max(usize) / 32), f.checkedMatWords(max(usize), 1, 1));
}

test "TFQ3 arithmetic: validBits is CUDA's 3/4/5 and Flash Next's 3..5 range on every u32 edge" {
    var bits: u32 = 0;
    while (bits < 70) : (bits += 1) {
        try t.expectEqual(mainValidBits(bits), f.validBits(bits));
        try t.expectEqual(!(bits < 3 or bits > 5), f.validBits(bits)); // Flash Next tfq3_check.zig:100 on main
    }
    for ([_]u32{ 1 << 31, max(u32) - 1, max(u32) }) |b| try t.expect(!f.validBits(b));
}

test "TFQ3 arithmetic: uint4Misaligned is every copy's 4-bit, off-a-uint4 test, and nothing else" {
    const offs = [_]u64{ 0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 204800, 204801, max(u32), 1 << 32, max(u64) - 3, max(u64) - 1, max(u64) };
    var bits: u32 = 0;
    while (bits < 9) : (bits += 1) for (offs) |o| try t.expectEqual(bits == 4 and o % 4 != 0, f.uint4Misaligned(bits, o));
    try t.expect(f.uint4Misaligned(4, max(u64)));
    try t.expect(!f.uint4Misaligned(4, max(u64) - 3));
    try t.expect(!f.uint4Misaligned(5, 1)); // 3/5-bit words are read one at a time
    try t.expect(!f.uint4Misaligned(max(u32), 1));
}

test "TFQ3 arithmetic: overlaps refuses exactly what main's Flash Next loop and GLM disjoint refused" {
    // every three spans with off 0..6 and len 0..4 (ends fit, as both callers check first), two header lengths
    var n: usize = 0;
    for ([_]usize{ 0, 3 }) |head| for (0..35) |x| for (0..35) |y| for (0..35) |z| {
        const all = [3]f.Span{ .{ .off = x / 5, .len = x % 5 }, .{ .off = y / 5, .len = y % 5 }, .{ .off = z / 5, .len = z % 5 } };
        const want = mainFlashNextOverlap(head, &all);
        try t.expectEqual(want, mainGlmOverlap(head, &all));
        try t.expectEqual(want, f.overlaps(head, &all));
        n += @intFromBool(want);
    };
    try t.expect(n > 0 and n < 2 * 35 * 35 * 35);
    const s = struct {
        fn sp(off: usize, len: usize) f.Span {
            return .{ .off = off, .len = len };
        }
    }.sp;
    try t.expect(!f.overlaps(144, &.{ s(16384, 32768), s(49152, 16), s(65536, 1) })); // Flash Next's valid shard
    try t.expect(!f.overlaps(144, &.{ s(144, 10), s(154, 10), s(164, 10) })); // touching ends share no byte
    try t.expect(!f.overlaps(144, &.{ s(400, 10), s(300, 10), s(200, 10) })); // reversed order, still disjoint
    try t.expect(!f.overlaps(144, &.{ s(0, 0), s(200, 0), s(200, 0) })); // empty spans never overlap, even at 0
    try t.expect(f.overlaps(144, &.{ s(143, 1), s(200, 10), s(300, 10) })); // the header's last byte
    try t.expect(f.overlaps(144, &.{ s(200, 10), s(300, 10), s(209, 1) })); // first and third share byte 209
    try t.expect(f.overlaps(144, &.{ s(200, 100), s(250, 1), s(400, 1) })); // one inside another
    try t.expect(!f.overlaps(0, &.{}));
}
