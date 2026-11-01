//! A lane host's warm pass: a prompt-only request on a `warm_turns` engine, prefilled so the cache keeps its end.
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// A prompt-only pass on a `warm_turns` engine: the backend prefills, the cache keeps its end (a mark, if keyed on)
pub fn pass(h: *LaneHost, job: *Job, began: i96) bool {
    const be = h.core.backend;
    if (h.cache) |store| if (job.marks.len == 0 and (job.request.untaught or store.rules.lookahead > 0)) return h.cancel(job); // it would keep nothing
    @import("lane_host_load.zig").calling(h, job);
    be.prefill(&job.stream) catch |e| { // a yield ends it like a cancel: what it prefilled is kept
        if (e != error.Cancelled and e != error.Yielded) return @import("lane_host_end.zig").dropFailed(h, job, e);
        if (e == error.Yielded) @import("lane_host_load.zig").yielded(h);
        be.release(&job.stream);
        return h.cancel(job);
    };
    h.prefilled(job, began);
    if (h.cache) |store| _ = if (!job.request.untaught and store.rules.lookahead == 0) store.keepBorn(job.cacheKey(), job.request.bornLen(), @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request)) else false;
    be.release(&job.stream);
    h.remove(job);
    h.finish(job, .length, "");
    return true;
}
