//! Slow readers: a stream whose client is `limit` tokens behind sits out rounds; one stalled past `stall_ns` ends.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

pub const Limits = struct {
    limit: u32 = 1024, // a streamed reply's unread tokens at which its stream sits out rounds; it runs again at half
    stall_ns: i96 = 60 * std.time.ns_per_s, // a held stream whose client reads nothing this long is cancelled
    pauses: std.atomic.Value(u64) = .init(0), // streams held so far (counted on the engine thread)
    stalls: std.atomic.Value(u64) = .init(0), // and cancelled for a stalled reader
};

/// One job's hold (engine thread only): this host paused its stream; at `since` its client had `seen` tokens unread.
pub const Watch = struct { held: bool = false, since: i96 = 0, seen: u32 = 0, stalled: bool = false };

/// A turn's started streams: held for their readers, and free to run (a backend's own pause is neither).
pub const Count = struct { held: usize = 0, running: usize = 0 };

/// Before each turn (engine thread): hold readers behind, release caught-up ones, cancel stalled; all held: wait 2 ms.
pub fn step(h: *LaneHost) void {
    const c = watch(h, null);
    if (c.held == 0 or c.running > 0) return;
    h.lock();
    defer h.unlock();
    // an arrival or a cancel signals `wake`; a queued request that cannot be admitted no longer spins the loop
    if (h.cancels.items.len == 0 and !h.closing) h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(2), .clock = .awake } }) catch {};
}

/// Holds and releases the admitted jobs' streams by their readers' unread tokens, at `at` (null: the clock's now).
pub fn watch(h: *LaneHost, at: ?i96) Count {
    const lim = &h.unread_limits;
    var c: Count = .{};
    var now = at;
    // `admitted` changes only on this thread: the scan takes no lock
    for (h.admitted.items) |job| {
        if (!job.started or job.stream.finished) continue;
        const w = &job.backlog;
        const n = if (job.request.unread) |u| u.load(.monotonic) else 0;
        if (!w.held) {
            if (job.stream.paused) continue; // its backend's own pause (a larger session to free): not this host's
            if (n < lim.limit) {
                c.running += 1;
                continue;
            }
            if (job.stream.inflight != null) {
                job.stream.mode = .drain; // its next round lands the step it queued ahead and queues none
                c.running += 1;
                continue;
            }
            job.stream.paused = true;
            w.* = .{ .held = true, .since = clock(h, &now), .seen = n };
            _ = lim.pauses.fetchAdd(1, .monotonic);
            c.held += 1;
            continue;
        }
        if (n <= lim.limit / 2) {
            job.stream.paused = false;
            w.held = false;
            c.running += 1;
            continue;
        }
        c.held += 1;
        const t = clock(h, &now);
        if (n < w.seen) {
            w.seen = n; // its client read: the stall's clock starts again
            w.since = t;
        }
        if (w.stalled or t - w.since < lim.stall_ns) continue;
        h.lock();
        const sent = if (h.cancels.append(h.gpa, job.id)) |_| true else |_| false;
        h.unlock();
        if (!sent) continue; // no memory for the cancel: the next turn tries again
        w.stalled = true; // takeCancels ends it as a disconnect's cancel does, and frees its lane
        _ = lim.stalls.fetchAdd(1, .monotonic);
    }
    return c;
}

/// Whether `job`'s reader is `limit` tokens behind: the lone driver hands such a stream back for the host to hold.
pub fn behind(h: *const LaneHost, job: *const Job) bool {
    const unread = job.request.unread orelse return false;
    return unread.load(.monotonic) >= h.unread_limits.limit;
}

fn clock(h: *const LaneHost, now: *?i96) i96 {
    if (now.*) |t| return t;
    const t = std.Io.Clock.awake.now(h.io).toNanoseconds();
    now.* = t;
    return t;
}

test {
    _ = @import("lane_host_backlog_test.zig");
}
