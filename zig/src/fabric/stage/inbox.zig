//! Rank 0's receive side with rounds in flight: a thread takes each round's echo and rows off the ring, in order.
const std = @import("std");
const frame = @import("frame.zig");
const Ring = @import("ring.zig").Ring;

/// One round's frames from the rank before rank 0: the plan's echo, then its rows when it has any.
pub const Job = struct {
    round: u64,
    echo: []u8,
    /// The rows frame's payload (empty: a round without rows, the echo only).
    rows: []u8,
    echo_len: usize = 0,
    head: frame.Header = undefined,
    err: ?anyerror = null,
    done: bool = false,
};

/// Jobs in round order, taken by one thread, so the last stage never waits on rank 0 while rank 0 sends.
pub const Inbox = struct {
    pub const most = 64;

    io: std.Io,
    ring: *Ring,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    jobs: [most]*Job = undefined,
    posted: usize = 0,
    taken: usize = 0,
    quit: bool = false,
    /// The first failure: every later job gets it, since the link may be mid-frame.
    failed: ?anyerror = null,
    thread: ?std.Thread = null,

    /// The thread started on `ib`, which must stay where it is until `stop`.
    pub fn start(ib: *Inbox) !void {
        ib.thread = try std.Thread.spawn(.{}, run, .{ib});
    }

    /// A round's job, in the order the rounds were sent; at most `most` unfinished at once.
    pub fn post(ib: *Inbox, job: *Job) !void {
        ib.mutex.lockUncancelable(ib.io);
        defer ib.mutex.unlock(ib.io);
        if (ib.posted - ib.taken == most) return error.TooManyInFlight;
        ib.jobs[ib.posted % most] = job;
        ib.posted += 1;
        ib.cond.broadcast(ib.io);
    }

    /// Until `job` is taken off the ring; its error, if any.
    pub fn wait(ib: *Inbox, job: *Job) !void {
        ib.mutex.lockUncancelable(ib.io);
        defer ib.mutex.unlock(ib.io);
        while (!job.done) ib.cond.waitUncancelable(ib.io, &ib.mutex);
        if (job.err) |e| return e;
    }

    pub fn pending(ib: *Inbox) usize {
        ib.mutex.lockUncancelable(ib.io);
        defer ib.mutex.unlock(ib.io);
        return ib.posted - ib.taken;
    }

    /// The thread ended; a job still waiting on the ring ends once the ring stops or breaks.
    pub fn stop(ib: *Inbox) void {
        ib.mutex.lockUncancelable(ib.io);
        ib.quit = true;
        ib.cond.broadcast(ib.io);
        ib.mutex.unlock(ib.io);
        if (ib.thread) |t| t.join();
        ib.thread = null;
    }

    fn run(ib: *Inbox) void {
        while (true) {
            ib.mutex.lockUncancelable(ib.io);
            while (ib.taken == ib.posted and !ib.quit) ib.cond.waitUncancelable(ib.io, &ib.mutex);
            if (ib.taken == ib.posted) {
                ib.mutex.unlock(ib.io);
                return;
            }
            const job = ib.jobs[ib.taken % most];
            const failed = ib.failed;
            ib.mutex.unlock(ib.io);
            const err: ?anyerror = failed orelse if (take(ib.ring, job)) null else |e| e;
            ib.mutex.lockUncancelable(ib.io);
            job.err = err;
            job.done = true;
            if (err != null and ib.failed == null) ib.failed = err;
            ib.taken += 1;
            ib.cond.broadcast(ib.io);
            ib.mutex.unlock(ib.io);
        }
    }

    // only this thread receives while it runs; rank 0's thread sends (the ring's two links are separate sockets)
    fn take(ring: *Ring, job: *Job) !void {
        const echo = try ring.recv(.plan, job.round, job.echo);
        job.echo_len = @intCast(echo.bytes);
        if (job.rows.len > 0) job.head = try ring.recv(.rows, job.round, job.rows);
    }
};
