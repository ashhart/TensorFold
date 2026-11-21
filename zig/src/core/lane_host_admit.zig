//! Which queued request a lane host admits next: the head, or one behind a head that waits for pool room.
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const lanes = @import("lanes");
const prefix = @import("lane_host_prefix.zig");
const fetch = @import("lane_host_fetch.zig");
const pressure = @import("lane_host_pressure.zig");
const ledger = @import("lane_host_ledger.zig");
const Wait = @import("wait_reason.zig").Wait;

/// The queued job to admit (its index), or null to hold, each job held stores why; the caller holds the lock.
pub fn pick(h: *LaneHost) ?usize {
    if (pressure.holds(h)) return hold(h, .memory_pressure); // under memory pressure nothing starts or is read ahead
    if (!h.stale) fetch.queue(h); // disk states read ahead for queued requests, a lane free or not
    // a stale device takes no reply: `learn.refuse` ends the queued ones instead
    if (h.stale) return hold(h, .engine_unavailable);
    if (h.queued.items.len == 0 or h.admitted.items.len >= @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes)) return hold(h, .engine_busy);
    prefix.release(h); // requests whose awaited cut is in (or will not come) go on first
    while (true) { // a request that waits for a cut another prompt is filling stays queued, passed over
        const at = choose(h) orelse return null;
        if (!prefix.waits(h, h.queued.items[at])) return at;
        h.queued.items[at].wait_reason = .prefix_cut;
    }
}

/// Every queued job waits for `why`; no job is picked.
fn hold(h: *LaneHost, why: Wait) ?usize {
    for (h.queued.items) |j| j.wait_reason = why;
    return null;
}

/// Whether queued `job` is passed over: it waits for another fill's cut or for its disk state's read.
fn passed(h: *LaneHost, job: *const Job) bool {
    return passedFor(h, job) != null;
}

/// What queued `job` is passed over for (null: it is not).
fn passedFor(h: *LaneHost, job: *const Job) ?Wait {
    if (prefix.waiting(job)) return .prefix_cut;
    if (fetch.held(h, job)) return .state_read;
    return null;
}

/// Whether every queued request waits for a cut or a read: an idle engine sleeps until a read wakes it (lock held).
pub fn holds(h: *LaneHost) bool {
    for (h.queued.items) |j| if (!passed(h, j)) return false;
    return h.queued.items.len > 0;
}

/// The head, or one behind a head that waits for pool room, passing over prefix and read waiters.
fn choose(h: *LaneHost) ?usize {
    const first = for (h.queued.items, 0..) |j, i| {
        j.wait_reason = passedFor(h, j) orelse break i;
    } else return null;
    if (h.info_.admit == null and !ledger.priced(h)) return first;
    // alone: the whole pool is free (kept prefixes give their blocks back); a ledger counts leases, not live streams
    if (if (ledger.priced(h)) !ledger.outstanding(h) else h.core.activeCount() == 0) return first;
    const head = h.queued.items[first];
    if (fits(h, head)) return first;
    const room: Wait = if (ledger.priced(h)) .ledger_bytes else .pool_room;
    head.wait_reason = room;
    if (head.waits_since == null) head.waits_since = h.admissions;
    // it waits on the streams admitted before it waited: while one runs, later requests that fit go first; then none do
    for (h.admitted.items) |j| if (j.seq < head.waits_since.?) {
        for (h.queued.items[first + 1 ..], first + 1..) |j2, i| {
            j2.wait_reason = passedFor(h, j2) orelse if (fits(h, j2)) return i else room;
        }
        return null;
    };
    // the rest stay behind the head that waits for room
    for (h.queued.items[first + 1 ..]) |j2| j2.wait_reason = passedFor(h, j2) orelse room;
    return null;
}

/// Whether `job` fits now: the host's ledger when it has one, else the family's own pool.
fn fits(h: *const LaneHost, job: *const Job) bool {
    if (ledger.priced(h)) return ledger.fits(h, job);
    const a = h.info_.admit.?;
    return a.fits(a.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}

/// Whether to admit several at once: prompts batch (beside chunked fills up to many_rows), two wait, two lanes free.
pub fn batches(h: *LaneHost) bool {
    if (!h.core.batchesPrompts()) return false;
    if (h.core.fills() and h.core.backend.vtable.many_rows == 0) return false;
    h.lock();
    defer h.unlock();
    const lanes_now = @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes);
    return h.queued.items.len >= 2 and lanes_now -| h.admitted.items.len >= 2;
}

/// Every queued request a lane is free for, prompts in one pass (lone and warm ones as before); false: none taken.
pub fn many(h: *LaneHost) bool {
    var jobs: [lanes.Engine.max_together]*Job = undefined;
    var began: [lanes.Engine.max_together]i96 = undefined;
    var n: usize = 0;
    var took = false;
    while (n < jobs.len) {
        const job = switch (h.takeJob()) {
            .none => break,
            .failed => {
                took = true;
                continue;
            },
            .job => |j| j,
        };
        took = true;
        const at = h.setup(job) orelse continue;
        // a prompt-only job ends through `launch`: after `addStreams` nothing would complete it
        if (h.startsAlone(job) or job.stream.promptOnly() or (h.core.fills() and !h.core.batchable(&job.stream))) { // long, image, resumed: filled
            _ = h.launch(job, at);
            continue;
        }
        jobs[n] = job;
        began[n] = at;
        n += 1;
    }
    if (n == 1) return h.launch(jobs[0], began[0]);
    var streams: [lanes.Engine.max_together]*lanes.Stream = undefined;
    for (jobs[0..n], 0..) |job, i| streams[i] = &job.stream;
    var errs: [lanes.Engine.max_together]?anyerror = undefined;
    var done: [lanes.Engine.max_together]bool = undefined;
    const load = @import("lane_host_load.zig");
    const t0 = load.clock(h);
    h.core.addStreams(streams[0..n], errs[0..n], done[0..n]);
    load.creditBatch(h, jobs[0..n], done[0..n], t0); // one return, one credit: before any job reports alone
    for (jobs[0..n], began[0..n], errs[0..n]) |job, at, err| {
        if (err) |e| {
            _ = h.failedPass(job, e);
            continue;
        }
        h.prefilled(job, at);
        _ = h.deliver(job);
    }
    return took;
}
