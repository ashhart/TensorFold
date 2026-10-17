//! How a lane host ends jobs it will not run on: cancellations named since the last round, a failed round, a close.
const std = @import("std");
const api = @import("engine_api.zig");
const fill = @import("lane_host_fill.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const Id = api.Id;

/// Cancels queued and admitted jobs named since the last round.
pub fn takeCancels(h: *LaneHost) void {
    h.mutex.lockUncancelable(h.io);
    const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
    h.cancels.clearRetainingCapacity();
    var dropped: std.ArrayList(*Job) = .empty;
    for (ids) |id| {
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
            break;
        };
    }
    h.mutex.unlock(h.io);
    for (dropped.items) |job| h.finish(job, .cancelled, "");
    dropped.deinit(h.gpa);
    for (ids) |id| {
        var found: ?*Job = null;
        // the status reader walks `admitted` under this lock: the removal joins the walk's critical section
        h.lock();
        for (h.admitted.items, 0..) |job, i| if (job.id == id) {
            fill.unfill(h, job);
            if (!job.stream.finished) h.core.discard(&job.stream);
            _ = h.admitted.orderedRemove(i);
            found = job;
            break;
        };
        h.unlock();
        if (found) |job| h.finish(job, .cancelled, "");
    }
    h.gpa.free(ids);
}

/// A failed round ends every stream it held, with the backend's error.
pub fn failAll(h: *LaneHost, message: []const u8) void {
    h.core.sweep() catch {};
    h.lock();
    const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
    h.admitted.clearRetainingCapacity();
    h.filling.clearRetainingCapacity();
    h.unlock();
    for (jobs) |job| {
        if (!job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
    }
    h.gpa.free(jobs);
}

pub fn closeAll(h: *LaneHost) void {
    h.lock();
    for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
    for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
    h.unlock();
    takeCancels(h);
}
