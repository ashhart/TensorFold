//! A stop's drain deadline on the lane host: the rounds end at a boundary and what is left fails with a reason.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

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
    var queued = h.queued;
    h.queued = .empty;
    var jobs = h.admitted;
    h.admitted = .empty;
    h.filling.clearRetainingCapacity(); // a job still filling is admitted: it fails below, its lane freed
    h.unlock();
    if (queued.items.len + jobs.items.len > 0) std.log.warn("lane host: stopping at a round boundary: {d} running and {d} queued request(s) end ({s})", .{ jobs.items.len, queued.items.len, reason });
    for (jobs.items) |job| {
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, reason);
    }
    for (queued.items) |job| h.finish(job, .failed, reason);
    queued.deinit(h.gpa);
    jobs.deinit(h.gpa);
}
