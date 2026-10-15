//! Which queued request a lane host admits next: the head, or one behind a head that waits for pool room.
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const lanes = @import("lanes");
const lone = @import("lane_host_lone.zig");

/// The queued job to admit (its index), or null to hold; the caller holds the lock and a lane is free.
pub fn pick(h: *LaneHost) ?usize {
    const a = h.info_.admit orelse return 0;
    if (h.core.activeCount() == 0) return 0; // alone: the whole pool is free (kept prefixes give their blocks back)
    const head = h.queued.items[0];
    if (fits(a, head)) return 0;
    if (head.waits_since == null) head.waits_since = h.admissions;
    // it waits on the streams admitted before it waited: while one runs, later requests that fit go first; then none do
    for (h.admitted.items) |j| if (j.seq < head.waits_since.?) {
        for (h.queued.items[1..], 1..) |j2, i| if (fits(a, j2)) return i;
        return null;
    };
    return null;
}

fn fits(a: api.Admit, job: *const Job) bool {
    return a.fits(a.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}

/// Every queued request a lane is free for, prompts in one pass where the backend takes several; else one at a time.
pub fn many(h: *LaneHost) bool {
    if (!h.core.batchesPrompts() or h.core.fills()) return one(h);
    h.lock();
    const waiting = h.queued.items.len;
    const free = h.info_.lanes -| h.admitted.items.len;
    h.unlock();
    if (waiting < 2 or free < 2) return one(h);
    var jobs: [lanes.Engine.max_together]*Job = undefined;
    var n: usize = 0;
    while (n < jobs.len and n < free) {
        const job = h.takeJob() orelse break;
        if (!h.setup(job)) continue;
        if ((job.request.max_tokens == 0 and h.info_.warm_turns) or lone.fits(h, job)) {
            _ = h.launch(job);
            continue;
        }
        jobs[n] = job;
        n += 1;
    }
    if (n == 0) return true;
    if (n == 1) return h.launch(jobs[0]);
    var streams: [lanes.Engine.max_together]*lanes.Stream = undefined;
    for (jobs[0..n], 0..) |job, i| streams[i] = &job.stream;
    var errs: [lanes.Engine.max_together]?anyerror = undefined;
    h.core.addStreams(streams[0..n], errs[0..n]);
    for (jobs[0..n], errs[0..n]) |job, err| {
        if (err) |e| {
            _ = switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => h.requeue(job),
                else => h.drop(job, h.words(e)),
            };
            continue;
        }
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }
    return true;
}

/// The next queued request admitted alone: its stream set up and launched.
fn one(h: *LaneHost) bool {
    const job = h.takeJob() orelse return false;
    if (!h.setup(job)) return true;
    return h.launch(job);
}
