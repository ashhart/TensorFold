//! Reservation-time protection for queued and admitted prompt resumptions, scoped to the host's serial lookup.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

pub fn lookup(h: *LaneHost, job: *Job) !pc.Lookup {
    const store = h.cache.?;
    const before = store.reservation_guard;
    store.reservation_guard = .{ .ctx = job, .needed = needed };
    defer store.reservation_guard = before;
    const r = job.request;
    return store.lookupUserFrame(h.gpa, job.cacheKey(), r.bornLen(), r.history_len, r.rewind_len, r.user_end, r.shared_prefixes, r.chunks, h.spans(r), r.images, job.frame);
}

fn needed(ptr: *anyopaque, e: *const pc.Entry) bool {
    const current: *Job = @ptrCast(@alignCast(ptr));
    const h = current.host;
    h.lock();
    defer h.unlock();
    for (h.queued.items) |job| if (extends(current, job, e)) return true;
    for (h.admitted.items) |job| if (extends(current, job, e)) return true;
    return false;
}

fn extends(current: *const Job, job: *const Job, e: *const pc.Entry) bool {
    const r = job.request;
    if (job == current or r.promptOnly() or r.prompt_logprobs != null or r.untaught) return false;
    if (job.started and job.entry == null) return false;
    // An image frame is prepared at admission: retain possible matches rather than guess from compact keys.
    if (r.images.len > 0) return true;
    const key = job.cacheKey();
    return e.at < key.len and e.tokens.len <= key.len and std.mem.eql(u32, e.tokens, key[0..e.tokens.len]);
}
