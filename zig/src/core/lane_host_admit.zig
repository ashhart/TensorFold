//! Which queued request a lane host admits next: the head, or one behind a head that waits for pool room.
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

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
