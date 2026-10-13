//! A lane host's lone driver: the backend's own loop runs a lone stream until a foreground request or a cancel waits.
const LaneHost = @import("lane_host.zig").LaneHost;
const status = @import("lane_host_status.zig");
const Job = LaneHost.Job;

/// An idle backend driver takes a lone drafted request, sampled when supported; never one with logprobs.
pub fn fits(h: *LaneHost, job: *Job) bool {
    const r = job.request;
    const lone = h.lone orelse return false;
    if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null or r.logprobs != null or r.prompt_logprobs != null) return false;
    h.lock();
    defer h.unlock();
    return h.admitted.items.len == 1 and !h.foregroundWaiting() and h.cancels.items.len == 0 and h.core.activeCount() == 0;
}

/// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
pub fn run(h: *LaneHost, job: *Job, began: i96) bool {
    h.lone_job = job;
    job.delivered = 0;
    const lone = h.lone.?;
    const Hooks = struct {
        fn committed(ctx: *anyopaque) void {
            const host: *LaneHost = @ptrCast(@alignCast(ctx));
            const j = host.lone_job.?;
            if (!j.prefill_sent) {
                j.prefill_sent = true;
                host.prefilled(j, j.began);
            }
            host.send(j);
            status.noteLive(host);
        }
        fn yield(ctx: *anyopaque) bool {
            const host: *LaneHost = @ptrCast(@alignCast(ctx));
            host.lock();
            defer host.unlock();
            return host.foregroundWaiting() or host.cancels.items.len > 0 or host.closing;
        }
    };
    const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
    h.lone_job = null;
    const handed = paused catch |e| {
        if (!job.prefill_sent) LaneHost.emit(job, .{ .prefilled = 0 });
        h.remove(job);
        h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
        return true;
    };
    if (!job.prefill_sent) h.prefilled(job, began);
    if (handed) {
        h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
        h.send(job);
        return true;
    }
    if (h.deliver(job)) h.remove(job);
    return true;
}
