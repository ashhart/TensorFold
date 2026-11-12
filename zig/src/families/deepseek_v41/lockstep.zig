//! A session operation every rank runs in the leader's order, then agreed: a request's error on any rank is "not done".

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

const log = std.log.scoped(.dsv41);

/// cuda/graph_cache.zig's Agree: every rank's verdict AND-ed (a min all-reduce of one int), every rank in order.
pub const Agree = struct {
    ctx: *anyopaque,
    all: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
};

/// Faults no request recovers from: the transport (tp.collective.Error), the device (cuda's driver errors), memory.
const fatal = [_][]const u8{ "Aborted", "PeerFailed", "Timeout", "BackendFailed", "CudaFailed", "CudaContextLost", "OutOfDeviceMemory", "DriverUnavailable", "MissingSymbol", "OutOfMemory" };

pub fn isFatal(err: anyerror) bool {
    for (fatal) |name| if (std.mem.eql(u8, @errorName(err), name)) return true;
    return false;
}

/// `result`: this rank's outcome of `what`; true when every rank's went through. A fatal error is returned as it is.
pub fn agreed(agree: ?Agree, result: anyerror!void, what: []const u8, rank: u32) !bool {
    var ok = true;
    result catch |err| {
        if (isFatal(err)) return err;
        if (!quiet) log.warn("sessions: rank {d}: {s} failed ({t}): every rank takes it as not done, the request goes on", .{ rank, what, err });
        ok = false;
    };
    if (agree) |a| ok = try a.all(a.ctx, ok);
    return ok;
}

// -- host test: ranks on threads, the agreement an in-process AND ---------------------------------------------------

const Ranks = struct {
    /// each round's votes: 0 not yet, 1 ok, 2 not
    votes: [8][4]std.atomic.Value(u8) = @splat(@splat(.init(0))),
    ranks: [4]Rank = undefined,
    world: u32,

    const Rank = struct { p: *Ranks, me: u32, round: u32 = 0 };

    fn agree(p: *Ranks, me: u32) Agree {
        p.ranks[me] = .{ .p = p, .me = me, .round = p.ranks[me].round };
        return .{ .ctx = &p.ranks[me], .all = all };
    }

    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const k: *Rank = @ptrCast(@alignCast(ctx));
        const r = k.round;
        k.round += 1;
        k.p.votes[r][k.me].store(if (ok) 1 else 2, .release);
        var every = ok;
        for (k.p.votes[r][0..k.p.world]) |*v| {
            var spins: u64 = 0;
            while (v.load(.acquire) == 0) : (spins += 1) {
                if (spins > 1 << 32) return error.Timeout; // a rank that never votes: the test fails, not hangs
                std.atomic.spinLoopHint();
            }
            every = every and v.load(.acquire) == 1;
        }
        return every;
    }
};

/// One rank's requests: the first fails on rank 0 (a NotInRam), the second goes through on all, the last rank faults.
fn runRank(p: *Ranks, me: u32, got: *[3]bool, err: *?anyerror) void {
    const first: anyerror!void = if (me == 0) error.NotInRam else {};
    got[0] = agreed(p.agree(me), first, "restore", me) catch |e| {
        err.* = e;
        return;
    };
    got[1] = agreed(p.agree(me), {}, "restore", me) catch |e| {
        err.* = e;
        return;
    };
    // a transport fault is not a request's: it is returned (the rank's fail-fast), after its own vote
    const last = me + 1 == p.world;
    const peer: anyerror!void = if (last) error.PeerFailed else {};
    got[2] = agreed(p.agree(me), peer, "restore", me) catch |e| {
        err.* = e;
        if (last) _ = Ranks.all(&p.ranks[me], false) catch {};
        return;
    };
}

test "lockstep: rank 0's failed request is not done on two or three ranks, and the next request runs on all" {
    for ([_]u32{ 2, 3 }) |w| {
        var p: Ranks = .{ .world = w };
        for (&p.ranks, 0..) |*k, i| k.* = .{ .p = &p, .me = @intCast(i) };
        var got: [4][3]bool = @splat(.{ true, false, true });
        var errs: [4]?anyerror = @splat(null);
        var threads: [4]std.Thread = undefined;
        for (0..w) |i| threads[i] = try std.Thread.spawn(.{}, runRank, .{ &p, @as(u32, @intCast(i)), &got[i], &errs[i] });
        for (threads[0..w]) |t| t.join();
        for (got[0..w]) |g| {
            try std.testing.expect(!g[0]); // the failed request: not done on any rank
            try std.testing.expect(g[1]); // the next: done on all
        }
        for (0..w - 1) |i| {
            try std.testing.expect(!got[i][2]); // the others hear the fault as a failed vote
            try std.testing.expectEqual(@as(?anyerror, null), errs[i]);
        }
        try std.testing.expectEqual(@as(?anyerror, error.PeerFailed), errs[w - 1]);
    }
    try std.testing.expect(isFatal(error.CudaFailed) and !isFatal(error.NotInRam) and !isFatal(error.PoolExhausted));
}
