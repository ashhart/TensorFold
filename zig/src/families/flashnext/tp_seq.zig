//! Speed-up decode's messages, numbered the same on both Macs: each one's flag value and its slot in both windows.
const std = @import("std");

/// A decode message: the flag value its landing stores (every kind counts up together) and its send and receive slot.
pub const Msg = struct { flag: u32, slot: u32 };

/// Decode messages in post order: exchanges and picks alternate slots by the shared count, reduces by their own.
pub const Seqs = struct {
    x: u32 = 0, // decode messages so far: the flag values
    r: u32 = 0, // reduces so far: rank 0's 1-row merges never wait, so only the peer's next reduce frees a slot

    pub fn exchange(s: *Seqs) Msg {
        s.x +%= 1;
        return .{ .flag = s.x, .slot = s.x % 2 };
    }

    pub fn pick(s: *Seqs) Msg {
        s.x +%= 1;
        return .{ .flag = s.x, .slot = s.x % 2 };
    }

    pub fn reduce(s: *Seqs) Msg {
        s.x +%= 1;
        s.r +%= 1;
        return .{ .flag = s.x, .slot = s.r % 2 };
    }

    /// The slot the next reduce's partial is written into before it is posted.
    pub fn nextReduceSlot(s: Seqs) u32 {
        return (s.r +% 1) % 2;
    }
};

/// The window rows whose experts and MoE branch `rank` computes, [lo, hi): rank 0 the first half, rounded up.
pub fn ownRows(rank: u32, rows: usize) [2]usize {
    const half = (rows + 1) / 2;
    return if (rank == 0) .{ 0, half } else .{ half, rows };
}

pub const Kind = enum { reduce, exchange, pick };

/// A message's bytes for a window of `rows`: fp32 partials, `rank`'s MoE branch rows (bf16), or (value, index) picks.
pub fn bytes(kind: Kind, rank: u32, rows: usize) usize {
    const own = ownRows(rank, rows);
    return switch (kind) {
        .reduce => rows * D * 4,
        .exchange => (own[1] - own[0]) * D * 2,
        .pick => rows * 8,
    };
}

const D = 2560;
const Ev = struct { op: enum { write, post, poll }, kind: Kind, msg: Msg };

/// One Mac's GPU-side steps in a decode round, in replay.zig windowEncode's order under speed-up mode.
fn round(evs: *std.ArrayList(Ev), a: std.mem.Allocator, s: *Seqs, rank: u32, rows: usize) !void {
    const own = ownRows(rank, rows);
    const peer_rows = rows - (own[1] - own[0]);
    for (1..rows) |_| { // the MTP head's drafts: one argmax swap each
        const m = s.pick();
        try evs.appendSlice(a, &.{ .{ .op = .write, .kind = .pick, .msg = m }, .{ .op = .post, .kind = .pick, .msg = m }, .{ .op = .poll, .kind = .pick, .msg = m } });
    }
    var pending: ?Msg = null;
    for (0..48) |i| {
        if (pending) |e| { // the combine posts the last exchange and waits only for rows the peer made
            try evs.append(a, .{ .op = .post, .kind = .exchange, .msg = e });
            if (peer_rows > 0) try evs.append(a, .{ .op = .poll, .kind = .exchange, .msg = e });
        }
        if (i % 4 != 3) { // a DeltaNet layer: the out-projection partial into the reduce's slot, posted, the peer's read
            const m = s.reduce();
            try evs.appendSlice(a, &.{ .{ .op = .write, .kind = .reduce, .msg = m }, .{ .op = .post, .kind = .reduce, .msg = m }, .{ .op = .poll, .kind = .reduce, .msg = m } });
        }
        const e = s.exchange();
        if (own[1] > own[0]) try evs.append(a, .{ .op = .write, .kind = .exchange, .msg = e });
        pending = e;
    }
    try evs.append(a, .{ .op = .post, .kind = .exchange, .msg = pending.? });
    if (peer_rows > 0) try evs.append(a, .{ .op = .poll, .kind = .exchange, .msg = pending.? });
    const m = s.pick();
    try evs.appendSlice(a, &.{ .{ .op = .write, .kind = .pick, .msg = m }, .{ .op = .post, .kind = .pick, .msg = m }, .{ .op = .poll, .kind = .pick, .msg = m } });
}

fn find(evs: []const Ev, op: @FieldType(Ev, "op"), flag: u32) ?usize {
    for (evs, 0..) |ev, i| if (ev.op == op and ev.msg.flag == flag) return i;
    return null;
}

/// Whether Mac p, at its step w, has read a message the peer posted after reading p's message `flag`.
fn answered(macs: [2][]const Ev, p: usize, flag: u32, w: usize) bool {
    const peer = macs[1 - p];
    const read = find(peer, .poll, flag) orelse return false;
    for (peer[read + 1 ..]) |ev| if (ev.op == .post) {
        if (find(macs[p], .poll, ev.msg.flag)) |at| if (at < w) return true;
    };
    return false;
}

test "a decode slot is written again only after the peer answered its last message, at every window width" {
    const a = std.testing.allocator;
    for ([_][]const usize{ &.{ 1, 1, 1 }, &.{ 4, 4, 4 }, &.{ 1, 3, 1, 8, 1 }, &.{ 2, 1, 16, 1, 5 } }) |widths| {
        var macs: [2]std.ArrayList(Ev) = .{ .empty, .empty };
        defer for (&macs) |*m| m.deinit(a);
        for (0..2) |rank| {
            var s: Seqs = .{};
            for (widths) |rows| try round(&macs[rank], a, &s, @intCast(rank), rows);
        }
        const both = [2][]const Ev{ macs[0].items, macs[1].items };
        for (0..2) |p| {
            var last: [3][2]?u32 = @splat(.{ null, null });
            for (both[p], 0..) |ev, w| if (ev.op == .write) {
                const slot = &last[@backingInt(ev.kind)][ev.msg.slot];
                if (slot.*) |before| if (!answered(both, p, before, w)) {
                    std.debug.print("rank {d} rewrites {s} slot {d} (message {d}) before the peer answered message {d}; widths {any}\n", .{ p, @tagName(ev.kind), ev.msg.slot, ev.msg.flag, before, widths });
                    return error.SlotRewrittenUnanswered;
                };
                slot.* = ev.msg.flag;
            };
        }
    }
}

test "a message's bytes follow its own window: both ranks' branches cover it once, partials and picks every row" {
    for (1..17) |rows| {
        try std.testing.expectEqual(rows * D * 2, bytes(.exchange, 0, rows) + bytes(.exchange, 1, rows));
        try std.testing.expectEqual(rows * D * 4, bytes(.reduce, 1, rows));
        try std.testing.expectEqual(rows * 8, bytes(.pick, 0, rows));
    }
    try std.testing.expectEqual(@as(usize, 0), bytes(.exchange, 1, 1)); // a 1-row window's merge: rank 1 sends a flag alone
}

test "both Macs number every message alike, flags rising in post order" {
    var s: [2]Seqs = .{ .{}, .{} };
    for (0..2) |k| {
        const m = [_]Msg{ s[k].reduce(), s[k].exchange(), s[k].reduce(), s[k].pick(), s[k].reduce() };
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5 }, &.{ m[0].flag, m[1].flag, m[2].flag, m[3].flag, m[4].flag });
        try std.testing.expectEqualSlices(u32, &.{ 1, 0, 1 }, &.{ m[0].slot, m[2].slot, m[4].slot }); // reduces alternate by their own count
    }
    try std.testing.expectEqual(s[0], s[1]);
    try std.testing.expectEqual(@as(u32, 0), s[0].nextReduceSlot());
    try std.testing.expectEqual([2]usize{ 0, 1 }, ownRows(0, 1));
    try std.testing.expectEqual([2]usize{ 1, 1 }, ownRows(1, 1));
}
