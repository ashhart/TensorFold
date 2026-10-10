//! A stop's drain deadline on the lane host: the rounds end at a boundary and what is left fails with a reason.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// `LaneHost.halt`: the engine thread stops at the next round boundary; what is left fails with `reason`.
pub fn halt(h: *LaneHost, reason: []const u8) void {
    h.mutex.lockUncancelable(h.io);
    h.closing = true;
    h.halt_reason = reason;
    h.wake.broadcast(h.io);
    h.mutex.unlock(h.io);
    if (h.thread) |t| t.join();
    h.thread = null;
}

/// `halt`: between rounds, every request queued, filling or running fails with `reason` (its stream discarded).
pub fn haltAll(h: *LaneHost, reason: []const u8) void {
    h.lock();
    const queued = h.gpa.dupe(*Job, h.queued.items) catch &.{};
    h.queued.clearRetainingCapacity();
    const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
    h.admitted.clearRetainingCapacity();
    h.filling.clearRetainingCapacity(); // a job still filling is admitted: it fails below, its lane freed
    h.unlock();
    if (queued.len + jobs.len > 0) std.log.warn("lane host: stopping at a round boundary: {d} running and {d} queued request(s) end ({s})", .{ jobs.len, queued.len, reason });
    for (jobs) |job| {
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, reason);
    }
    for (queued) |job| h.finish(job, .failed, reason);
    h.gpa.free(queued);
    h.gpa.free(jobs);
}
