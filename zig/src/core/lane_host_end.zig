//! How a lane host ends jobs it will not run on: cancellations named since the last round, a failed round, a close.
const std = @import("std");
const fill = @import("lane_host_fill.zig");
const lane_halt = @import("lane_halt.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// Cancels queued and admitted jobs named since the last round; the names are taken whole, so none is lost to memory.
pub fn takeCancels(h: *LaneHost) void {
    h.lock();
    var ids = h.cancels;
    h.cancels = .empty;
    h.unlock();
    defer ids.deinit(h.gpa);
    for (ids.items) |id| {
        var found: ?*Job = null;
        // the status reader walks `admitted` under this lock: the removal joins the walk's critical section
        h.lock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            found = h.queued.orderedRemove(i);
            break;
        };
        if (found == null) for (h.admitted.items, 0..) |job, i| if (job.id == id) {
            fill.unfill(h, job);
            if (job.started and !job.stream.finished) h.core.discard(&job.stream);
            found = h.admitted.orderedRemove(i);
            break;
        };
        h.unlock();
        if (found) |job| {
            if (job.started and !job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .cancelled, "");
        }
    }
}

/// A failed round ends every job it admitted, with the backend's error; the list is taken whole, not copied.
pub fn failAll(h: *LaneHost, message: []const u8) void {
    std.log.warn("lane host: a round failed ({s}): every admitted stream fails", .{message});
    h.core.sweep() catch {};
    h.lock();
    var jobs = h.admitted;
    h.admitted = .empty;
    h.filling.clearRetainingCapacity();
    h.unlock();
    defer jobs.deinit(h.gpa);
    for (jobs.items) |job| {
        // a job admitted this round has no stream yet when the planner's admitted callback fails
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
    }
}

/// Ends every queued and admitted job cancelled (the cancellations named before it first), copying nothing.
pub fn closeAll(h: *LaneHost) void {
    if (h.halt_reason) |reason| return lane_halt.haltAll(h, reason);
    takeCancels(h);
    h.lock();
    var queued = h.queued;
    h.queued = .empty;
    var admitted = h.admitted;
    h.admitted = .empty;
    h.filling.clearRetainingCapacity();
    h.unlock();
    defer queued.deinit(h.gpa);
    defer admitted.deinit(h.gpa);
    for (admitted.items) |job| {
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .cancelled, "");
    }
    for (queued.items) |job| h.finish(job, .cancelled, "");
}
