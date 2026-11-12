//! The pair link's liveness: rank 1 replays rank 0's rounds, and both ranks stop when the link dies.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lb = @import("qwen27_lanes.zig");
const pair_mod = @import("qwen27_pair.zig");

/// Rank 1's follower: replay rank 0's rounds in order; a rank that cannot replay stops to be restarted.
pub fn follow(b: *lb.Metal) void {
    const l = b.link orelse return;
    while (true) {
        const seen = l.pair.next(l.gpa, &l.words) catch |err| {
            if (err != error.PairStopped) failFollow("a round", @errorName(err));
            return;
        };
        const k = seen orelse return;
        // a frame is checked whole before any word is read as a field or a slot indexes the runner
        const command = pair_mod.decode(k, l.words.items, b.runner.slots) catch |err| failFollow("a command", .{ k, @errorName(err) });
        replay(b, command);
    }
}

/// One checked command on rank 1's runner, as rank 0 ran it.
fn replay(b: *lb.Metal, command: pair_mod.Command) void {
    switch (command) {
        .begin => |slot| {
            b.dropRound();
            b.runner.reset(slot) catch |err| return failFollow("begin", @errorName(err));
        },
        .chunk => |c| {
            const session = q.session.Session{ .runner = b.runner, .slot = c.slot };
            if (c.decoded) {
                session.decodeChunk(c.ids, c.last) catch |err| return failFollow("decode chunk", @errorName(err));
            } else {
                session.promptChunk(c.ids, c.last) catch |err| return failFollow("prompt chunk", @errorName(err));
            }
        },
        .step => |s| (q.session.Session{ .runner = b.runner, .slot = s.slot }).step(s.token) catch |err| return failFollow("step", @errorName(err)),
        .tree => |t| {
            const parents = b.gpa.alloc(i32, t.parents.len) catch |err| return failFollow("tree parents", @errorName(err));
            defer b.gpa.free(parents);
            for (t.parents, parents) |word, *p| p.* = @bitCast(word);
            const round = q.round_plan.Round.init(b.gpa, &.{.{ .slot = t.slot, .start = b.runner.offsets[t.slot], .capacity = b.runner.capacity, .ids = t.ids, .parents = parents }}, @intCast(b.runner.model.config.conv_kernel), b.runner.slots, @intCast(b.runner.model.shard.global.vocab)) catch |err| return failFollow("tree round", @errorName(err));
            if (b.round) |*old| old.deinit();
            b.round = round;
            b.runner.verifyHead(&b.round.?, .all) catch |err| return failFollow("tree verify", @errorName(err));
        },
        .commit => |c| {
            const round = &(b.round orelse return failFollow("commit without a tree", c.slot));
            if (round.slots[0] != c.slot) return failFollow("commit for another slot's tree", .{ c.slot, round.slots[0] });
            b.runner.keep(round, &.{c.path}) catch |err| return failFollow("tree keep", @errorName(err));
            round.deinit();
            b.round = null;
        },
        .end => {},
    }
}

/// This rank can no longer hold rank 0's state: it stops rather than serve on a stale half, to be restarted.
fn failFollow(what: []const u8, detail: anytype) noreturn {
    std.log.err("qwen27 rank 1: cannot replay rank 0's {s} ({any}): stopping so both ranks are restarted", .{ what, detail });
    std.process.exit(1);
}

/// Both ranks stop when the link dies: a dead peer must not leave HTTP up on a half, so a supervisor restarts both.
pub fn watch(p: *lb.Pair, quit: *const std.atomic.Value(bool)) void {
    var seen = p.heartbeats();
    var since = mtl.clock.seconds();
    while (!quit.load(.acquire)) {
        if (p.failed()) down(p, "the pair link is down");
        p.beat();
        const beats = p.heartbeats();
        const now = mtl.clock.seconds();
        if (beats != seen) {
            seen = beats;
            since = now;
        } else if (now - since > link_silence_s) down(p, "the peer stopped signalling");
        const ts: platform.c.timespec = .{ .sec = 0, .nsec = @intCast(250 * std.time.ns_per_ms) };
        _ = platform.c.nanosleep(&ts, null);
    }
}

/// Beats the watchdog may miss before the peer counts as gone: twelve 250 ms intervals, longer than any seam.
const link_silence_s: f64 = 3.0;

/// The link is not a serving state: name the half that is gone, then end the process the supervisor watches.
fn down(p: *lb.Pair, what: []const u8) noreturn {
    std.log.err("qwen27 rank {d}: {s}: exiting so both ranks are restarted", .{ p.rank, what });
    std.process.exit(1);
}

/// The pair's threads, owned by the host so `close` joins them before the pair they read is torn down.
pub const Workers = struct {
    follower: ?std.Thread = null,
    watcher: ?std.Thread = null,
    quit: std.atomic.Value(bool) = .init(false),

    /// Fail closed: a rank whose watchdog cannot start stops, since a dead peer would leave it serving a half.
    pub fn startWatch(w: *Workers, p: *lb.Pair, a: std.mem.Allocator, why: *[]const u8) !void {
        w.watcher = std.Thread.spawn(.{}, watch, .{ p, &w.quit }) catch |err| {
            why.* = try std.fmt.allocPrint(a, "the pair link's watchdog would not start ({s}): a dead peer would leave this rank up, so this rank stops", .{@errorName(err)});
            return error.PairNoWatchdog;
        };
    }

    /// Rank 1 replays rank 0's rounds after the host is up; rank 0 serves its own requests.
    pub fn startFollower(w: *Workers, p: *lb.Pair, b: *lb.Metal, a: std.mem.Allocator, why: *[]const u8) !void {
        if (p.rank != 1) return;
        w.follower = std.Thread.spawn(.{}, follow, .{b}) catch |err| {
            why.* = try std.fmt.allocPrint(a, "rank 1's follower would not start ({s})", .{@errorName(err)});
            return err;
        };
    }

    /// Stops both threads before the pair they read is torn down: the watchdog first, then the follower.
    pub fn stop(w: *Workers, p: *lb.Pair) void {
        w.quit.store(true, .release);
        p.stopFollowing();
        if (w.follower) |t| t.join();
        if (w.watcher) |t| t.join();
    }
};
