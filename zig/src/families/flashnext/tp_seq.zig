//! Speed-up decode's messages, numbered the same on both Macs: each one's flag value and its slot in both windows.
const std = @import("std");

/// A decode message: the flag value its landing stores (every kind counts up together) and its send and receive slot.
pub const Msg = struct { flag: u32, slot: u32 };

/// Decode messages in post order: every kind alternates two slots by the shared count.
pub const Seqs = struct {
    x: u32 = 0, // decode messages so far: the flag values

    pub fn exchange(s: *Seqs) Msg {
        return s.next();
    }

    pub fn pick(s: *Seqs) Msg {
        return s.next();
    }

    pub fn reduce(s: *Seqs) Msg {
        return s.next();
    }

    /// The slot the next reduce's partial is written into before it is posted.
    pub fn nextReduceSlot(s: Seqs) u32 {
        return (s.x +% 1) % 2;
    }

    fn next(s: *Seqs) Msg {
        s.x +%= 1;
        return .{ .flag = s.x, .slot = s.x % 2 };
    }
};

/// The window rows whose experts and MoE branch `rank` computes, [lo, hi): rank 0 the first half, rounded up.
pub fn ownRows(rank: u32, rows: usize) [2]usize {
    const half = (rows + 1) / 2;
    return if (rank == 0) .{ 0, half } else .{ half, rows };
}

test "both Macs number every message alike, flags rising in post order" {
    var s: [2]Seqs = .{ .{}, .{} };
    for (0..2) |k| {
        const m = [_]Msg{ s[k].reduce(), s[k].exchange(), s[k].reduce(), s[k].pick(), s[k].reduce() };
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5 }, &.{ m[0].flag, m[1].flag, m[2].flag, m[3].flag, m[4].flag });
    }
    try std.testing.expectEqual(s[0], s[1]);
    try std.testing.expectEqual([2]usize{ 0, 1 }, ownRows(0, 1));
    try std.testing.expectEqual([2]usize{ 1, 1 }, ownRows(1, 1));
}
