//! The exactness check's comparison: one-row steps' logits against the same rows' from windows, bit for bit.
const std = @import("std");

/// How many rows of `vocab` bf16 logits differ in any bit between `a` and `b`; the first ones' indices go to `first`.
pub fn differing(a: []const u16, b: []const u16, vocab: usize, first: []usize) usize {
    std.debug.assert(vocab > 0 and a.len == b.len and a.len % vocab == 0);
    var n: usize = 0;
    for (0..a.len / vocab) |r| {
        if (std.mem.eql(u16, a[r * vocab ..][0..vocab], b[r * vocab ..][0..vocab])) continue;
        if (n < first.len) first[n] = r;
        n += 1;
    }
    return n;
}

/// Flips the lowest bit of one logit in row `row`: the planted difference the check must report.
pub fn plant(rows: []u16, vocab: usize, row: usize) void {
    rows[row * vocab + vocab / 2] ^= 1;
}

test "llama exact: equal rows pass, a planted row is reported by index, and only it" {
    const vocab = 8;
    var a: [4 * vocab]u16 = undefined;
    for (&a, 0..) |*x, i| x.* = @intCast(0x3f80 + i);
    var b = a;
    var first: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 0), differing(&a, &b, vocab, &first));
    plant(&b, vocab, 2);
    try std.testing.expectEqual(@as(usize, 1), differing(&a, &b, vocab, &first));
    try std.testing.expectEqual(@as(usize, 2), first[0]);
    plant(&b, vocab, 2);
    try std.testing.expectEqual(@as(usize, 0), differing(&a, &b, vocab, &first));
}
