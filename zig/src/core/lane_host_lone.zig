//! A lane host's lone driver: the backend's own loop runs a lone stream until a foreground request or a cancel waits.
const LaneHost = @import("lane_host.zig").LaneHost;
const status = @import("lane_host_status.zig");
const Job = LaneHost.Job;

/// An idle backend driver takes a lone drafted request, sampled, with logprobs, a budget or a grammar when supported.
pub fn fits(h: *LaneHost, job: *Job) bool {
    if (!takes(h, job)) return false;
    h.lock();
    defer h.unlock();
    return h.admitted.items.len == 1 and !h.foregroundWaiting() and h.cancels.items.len == 0 and h.core.activeCount() == 0;
}

fn takes(h: *LaneHost, job: *Job) bool {
    const r = job.request;
    const lone = h.lone orelse return false;
    const think = lone.min_budget > 0 and (r.think_budget == 0 or r.think_budget > lone.min_budget);
    return !((r.sampling != null and !lone.sampled) or !r.drafts or ((r.think_budget > 0 or r.loop_guard) and !think) or r.call != null or (r.structure != null and !lone.structures) or (r.logprobs != null and !lone.logprobs) or r.prompt_logprobs != null);
}

/// After a round: the lane core's last live stream goes back to the driver when nothing else waits or is due.
pub fn reclaim(h: *LaneHost) void {
    const lone = h.lone orelse return;
    if (!lone.resumes) return; // a driver that only opens streams would prefill this one again
    h.lock();
    const idle = h.admitted.items.len == 1 and h.queued.items.len == 0 and h.filling.items.len == 0 and !h.foregroundWaiting() and h.cancels.items.len == 0 and !h.closing;
    const job = if (idle) h.admitted.items[0] else null;
    h.unlock();
    const j = job orelse return;
    const s = &j.stream;
    // a forced close the driver would hand straight back stays in the lane core
    if (s.paused or s.force.items.len > 0 or s.budgetWithin(lone.min_budget) or !takes(h, j) or h.core.activeCount() != 1) return;
    if (!h.core.detach(s)) return;
    j.prefill_sent = true; // the lane core sent it
    _ = run(h, j, j.began, true);
}

/// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
pub fn run(h: *LaneHost, job: *Job, began: i96, resumed: bool) bool {
    h.lone_job = job;
    if (!resumed) job.delivered = 0;
    const lone = h.lone.?;
    const Hooks = struct {
        const Context = struct { host: *LaneHost, partial: bool = false };
        fn committed(ctx: *anyopaque) void {
            const c: *Context = @ptrCast(@alignCast(ctx));
            const host = c.host;
            const j = host.lone_job.?;
            if (!j.prefill_sent) {
                j.prefill_sent = true;
                host.prefilled(j, j.began);
            }
            host.send(j);
            status.noteLive(host);
        }
        fn yield(ctx: *anyopaque) bool {
            const c: *Context = @ptrCast(@alignCast(ctx));
            const host = c.host;
            host.lock();
            defer host.unlock();
            return host.foregroundWaiting() or host.cancels.items.len > 0 or host.closing;
        }
        fn prefillYield(ctx: *anyopaque) bool {
            if (!yield(ctx)) return false;
            const c: *Context = @ptrCast(@alignCast(ctx));
            c.partial = true;
            return true;
        }
    };
    var context: Hooks.Context = .{ .host = h };
    const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = &context, .committed = Hooks.committed, .yield = Hooks.yield, .resumed = resumed, .prefill_yield = Hooks.prefillYield });
    h.lone_job = null;
    const handed = paused catch |e| {
        if (!job.prefill_sent) LaneHost.emit(job, .{ .prefilled = 0 });
        if (e == error.Cancelled) return h.cancel(job);
        h.remove(job);
        h.finish(job, .failed, h.words(e));
        return true;
    };
    if (context.partial) {
        if (!handed or !h.core.fills()) return h.drop(job, "the backend cannot hand off an unfinished prompt");
        h.filling.append(h.gpa, job) catch return h.drop(job, "out of memory");
        return true;
    }
    if (!job.prefill_sent) h.prefilled(job, began);
    if (handed) {
        h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
        h.send(job);
        return true;
    }
    _ = h.deliver(job); // a finished job leaves the lists inside deliver, before finish frees it
    return true;
}
