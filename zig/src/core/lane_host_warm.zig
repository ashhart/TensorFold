//! A lane host's warm pass: a prompt-only request on a `warm_turns` engine, prefilled so the cache keeps its end.
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const Store = @import("prompt_cache.zig").Store;

/// A prompt-only pass on a `warm_turns` engine: the backend prefills, the cache keeps its end (a mark, if keyed on)
pub fn pass(h: *LaneHost, job: *Job, began: i96) bool {
    const be = h.core.backend;
    if (h.cache) |store| if (job.marks.len == 0 and !keepsEnd(store, job)) return h.cancel(job); // it would keep nothing
    @import("lane_host_load.zig").calling(h, job);
    be.prefill(&job.stream) catch |e| { // a yield ends it like a cancel: what it prefilled is kept
        if (e != error.Cancelled and e != error.Yielded) return @import("lane_host_end.zig").dropFailed(h, job, e);
        if (e == error.Yielded) @import("lane_host_load.zig").yielded(h);
        be.release(&job.stream);
        return h.cancel(job);
    };
    h.prefilled(job, began);
    if (h.cache) |store| _ = if (keepsEnd(store, job)) store.keepBornFrame(job.cacheKey(), job.request.bornLen(), @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request), job.request.images, job.frame) else false;
    be.release(&job.stream);
    h.remove(job);
    h.finish(job, .length, "");
    return true;
}

/// Whether the cache keeps the pass's end: not untaught, no lookahead, and with `warm_usable` a resumable row.
fn keepsEnd(store: *const Store, job: *const Job) bool {
    return !job.request.untaught and store.rules.lookahead == 0 and
        (!store.rules.warm_usable or store.usable(@intCast(job.request.prompt.len), job.request.chunks));
}

/// Foreground work arriving during the quiet interval delays this untouched pass instead of cancelling it.
pub fn deferJob(h: *LaneHost, job: *Job) bool {
    h.lock();
    for (h.admitted.items, 0..) |j, i| if (j == job) {
        _ = h.admitted.orderedRemove(i);
        break;
    };
    @import("lane_host_ledger.zig").release(h, job);
    var at: usize = 0;
    while (at < h.queued.items.len and !h.queued.items[at].request.background) at += 1;
    h.queued.insert(h.gpa, at, job) catch {
        h.unlock();
        h.finish(job, .failed, "out of memory");
        return true;
    };
    var discarded: @import("lane_warm_queue.zig").Discarded = .{};
    @import("lane_warm_queue.zig").prune(h, job, &discarded);
    if (@import("std").mem.indexOfScalar(*Job, h.queued.items, job) != null) h.load.requeued += 1;
    h.unlock();
    discarded.finish(h);
    return true;
}
