//! Real prefix identities and lockstep votes: a wrong RAM entry becomes a miss on both ranks.
const std = @import("std");
const sessions = @import("kv").sessions;
const lockstep = @import("lockstep.zig");
const Index = sessions.prefix.Index(u64);
const Want = sessions.restore.Want;
const t = std.testing;

const Pair = struct {
    votes: [2]std.atomic.Value(u8) = .{ .init(0), .init(0) },
    ranks: [2]Rank = undefined,

    const Rank = struct {
        pair: *Pair,
        me: usize,
        index: *Index,
        id: u32,
        want: Want,
        accepted: bool = true,
        err: ?anyerror = null,
    };

    fn all(ptr: *anyopaque, ok: bool) !bool {
        const r: *Rank = @ptrCast(@alignCast(ptr));
        r.pair.votes[r.me].store(if (ok) 1 else 2, .release);
        var spins: usize = 0;
        while (r.pair.votes[1 - r.me].load(.acquire) == 0) : (spins += 1) {
            if (spins > 1 << 30) return error.Timeout;
            std.atomic.spinLoopHint();
        }
        return ok and r.pair.votes[1 - r.me].load(.acquire) == 1;
    }

    fn run(r: *Rank) void {
        r.accepted = lockstep.agreed(.{ .ctx = r, .all = all }, r.want.check(r.index, r.id), "restore identity", @intCast(r.me)) catch |err| {
            r.err = err;
            return;
        };
    }
};

test "restore identity: wrong same-id RAM key position and missing id agree a miss on both ranks" {
    var indexes = [_]Index{ Index.init(t.allocator, 4), Index.init(t.allocator, 4) };
    defer for (&indexes) |*x| x.deinit();
    const ids = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var digests: [2]sessions.prefix.Digest = undefined;
    sessions.prefix.chain(1, &ids, 4, &digests);
    const leader = (try indexes[0].insert(1, &ids, &digests, true, 0)).id;
    const follower = (try indexes[1].insert(1, &ids, &digests, true, 0)).id;
    try t.expectEqual(leader, follower);
    const entry = indexes[0].get(leader);
    const want: Want = .{ .pos = entry.pos, .key = entry.key };
    for (0..4) |scenario| {
        const local = indexes[1].get(follower);
        local.key = want.key;
        local.pos = want.pos;
        if (scenario == 0) local.key[0] ^= 1;
        if (scenario == 1) local.pos += 1;
        var pair: Pair = .{};
        for (&pair.ranks, 0..) |*r, me| r.* = .{
            .pair = &pair,
            .me = me,
            .index = &indexes[me],
            .id = if (scenario == 2 and me == 1) follower + 1000 else leader,
            .want = want,
        };
        const a = try std.Thread.spawn(.{}, Pair.run, .{&pair.ranks[0]});
        const b = try std.Thread.spawn(.{}, Pair.run, .{&pair.ranks[1]});
        a.join();
        b.join();
        for (pair.ranks) |r| {
            try t.expectEqual(@as(?anyerror, null), r.err);
            try t.expectEqual(scenario == 3, r.accepted);
        }
    }
}

test {
    _ = @import("session_protocol.zig");
}
