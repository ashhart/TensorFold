//! A lane host's decision passes: HTTP threads queue them; the engine thread runs them while no stream lives.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// One queued decision pass: its task in, the failure and completion its requester waits on.
pub const Work = struct {
    task: api.DecisionTask,
    failed: ?error{ Failed, Refused } = null,
    finished: bool = false, // written under the host's mutex; `decided` is broadcast with it
};

/// Engine.score: the prompt's label logits queued for the engine thread (a score has no Refused).
pub fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
    var work = Work{ .task = .{ .score = .{ .prompt = prompt, .labels = labels, .logits = logits, .mass = mass } } };
    wait(self(ctx), &work) catch return error.Failed;
    return work.task.score.out;
}

/// Engine.decide: the packed decision queued for the engine thread; its refusal comes back as given.
pub fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
    var work = Work{ .task = .{ .decide = d } };
    return wait(self(ctx), &work);
}

/// Queue `work` and sleep until the engine thread's pass lands; it stays valid on this stack frame.
pub fn wait(h: *LaneHost, work: *Work) error{ Failed, Refused }!void {
    h.mutex.lockUncancelable(h.io);
    if (h.closing or h.stale or h.decider == null) {
        h.mutex.unlock(h.io);
        return error.Failed;
    }
    h.decisions.append(h.gpa, work) catch {
        h.mutex.unlock(h.io);
        return error.Failed;
    };
    h.mutex.unlock(h.io);
    h.wake.signal(h.io);
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    while (!work.finished) h.decided.waitUncancelable(h.io, &h.mutex);
    if (work.failed) |e| return e;
}

/// The oldest queued decision's pass: an idle engine runs it, a stale one fails the waiters instead.
pub fn step(h: *LaneHost) void {
    h.lock();
    if (h.stale) {
        h.unlock();
        return failAll(h);
    }
    const idle = !h.closing and h.decider != null and h.admitted.items.len == 0 and h.filling.items.len == 0 and h.core.activeCount() == 0;
    const work: ?*Work = if (idle and h.decisions.items.len > 0) h.decisions.orderedRemove(0) else null;
    h.unlock();
    const w = work orelse return;
    const decider = h.decider.?;
    decider.run(decider.ctx, &w.task) catch |e| {
        w.failed = e;
    };
    h.lock();
    w.finished = true;
    h.unlock();
    h.decided.broadcast(h.io);
}

/// Every queued decision failed (the host is closing or the device went stale): requesters wake.
pub fn failAll(h: *LaneHost) void {
    h.lock();
    while (h.decisions.items.len > 0) {
        const work = h.decisions.orderedRemove(0);
        work.failed = error.Failed;
        work.finished = true;
    }
    h.unlock();
    h.decided.broadcast(h.io);
}

fn self(ctx: *anyopaque) *LaneHost {
    return @ptrCast(@alignCast(ctx));
}
