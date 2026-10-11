//! A lane host's Sliding Weights lessons: queued while streams run, stepped only when the engine is idle.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Engine.learn: queue a lesson for the engine thread, refused when closing or without a learner.
pub fn submit(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    if (h.closing) return error.Closed;
    if (h.learner == null) return error.Unsupported;
    h.lessons.append(h.gpa, .{ .request = request, .sink = sink }) catch return error.Busy;
    h.wake.signal(h.io);
}

/// Learning runs only when nothing else waits: no stream, queued request or cancel (called under the lock).
pub fn ready(h: *const LaneHost) bool {
    const work = h.lesson_open or h.lessons.items.len > 0;
    return h.learner != null and work and h.queued.items.len == 0 and h.admitted.items.len == 0 and h.cancels.items.len == 0;
}

/// One unit of learning: open the next lesson if none is open, then step it; moved weights drop every kept prompt.
pub fn step(h: *LaneHost) void {
    const learner = h.learner.?;
    if (!h.lesson_open) {
        h.mutex.lockUncancelable(h.io);
        const lesson = h.lessons.orderedRemove(0);
        h.mutex.unlock(h.io);
        learner.begin(learner.ctx, lesson.request, lesson.sink) catch |e| {
            const failed: api.LearnEvent = .{ .done = .{ .message = h.words(e) } };
            return lesson.sink.event(lesson.sink.ctx, &failed);
        };
        h.lesson_open = true;
    }
    const s = learner.step(learner.ctx);
    if (s.changed) if (h.cache) |store| store.clear();
    h.lesson_open = !s.done;
}
