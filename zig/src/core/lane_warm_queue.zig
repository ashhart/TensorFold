//! Bound untouched warm jobs and discard ones already covered by later exact requests or kept endpoints.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const modes = @import("cache_modes.zig");

pub const Discarded = struct {
    jobs: [lanes.Engine.max_together + 1]*Job = undefined,
    len: usize = 0,
    fn add(d: *Discarded, job: *Job) void {
        std.debug.assert(d.len < d.jobs.len);
        d.jobs[d.len] = job;
        d.len += 1;
    }
    /// Removed before setup: release request arenas after unlocking, without calling a thread-owned backend.
    pub fn finish(d: *const Discarded, h: *LaneHost) void {
        for (d.jobs[0..d.len]) |job| {
            std.debug.assert(!job.started and job.entry == null and job.marks.len == 0 and job.key.len == 0);
            std.debug.assert(!job.load.on and job.cache_promise.bytes == 0 and job.cache_promise.slots == 0);
            std.debug.assert(job.lease.bytes == 0 and !job.lease.bound);
            const queued = if (job.admitted_ns) |at| @as(f64, @floatFromInt(@max(0, at - job.submitted))) / 1e9 else 0;
            LaneHost.emit(job, .{ .finished = .{ .reason = .cancelled, .stats = .{ .queued_seconds = queued }, .message = "background prefill superseded or queue full" } });
            h.gpa.destroy(job);
        }
    }
};

fn warm(h: *const LaneHost, job: *const Job) bool {
    const r = job.request;
    return h.info_.warm_turns and r.background and r.max_tokens == 0 and !r.promptOnly();
}

fn through(starts: []const u32, at: usize) []const u32 {
    for (starts, 0..) |v, i| if (v > at) return starts[0..i];
    return starts;
}

fn frame(h: *const LaneHost, r: *const api.Request) u32 {
    return if (r.images.len > 0) if (h.frame_key) |f| f(r.images) else 0 else 0;
}

/// The newer request extends the complete older key with the same arithmetic cuts, spans and image identities.
fn extends(h: *const LaneHost, older: *const Job, newer: *const Job) bool {
    const old = older.request;
    const next = newer.request;
    const key = old.cacheKey();
    const extended = next.cacheKey();
    if (key.len == 0 or key.len > extended.len or !std.mem.eql(u32, key, extended[0..key.len])) return false;
    if (!modes.equal(h.spans(old), h.spans(next), @intCast(key.len)) or
        !std.mem.eql(u32, through(old.chunks, key.len), through(next.chunks, key.len)) or frame(h, old) != frame(h, next)) return false;
    var n: usize = 0;
    for (next.images) |image| if (image.at < key.len) {
        if (n == old.images.len) return false;
        const x = old.images[n];
        if (x.at != image.at or x.tokens != image.tokens or x.gh != image.gh or x.gw != image.gw or x.hash != image.hash) return false;
        n += 1;
    };
    var old_n: usize = 0;
    for (old.images) |image| old_n += @intFromBool(image.at < key.len);
    return n == old_n;
}

/// After insertion, prune superseded warm jobs and retain at most one lane-count of queued warm requests (locked).
pub fn prune(h: *LaneHost, current: *Job, out: *Discarded) void {
    if (!h.info_.warm_turns) return;
    if (warm(h, current)) {
        for (h.admitted.items) |other| if (other != current and extends(h, current, other)) {
            if (remove(h, current, out)) return;
        };
        for (h.queued.items) |other| if (other != current and extends(h, current, other) and
            (!warm(h, other) or other.request.bornLen() > current.request.bornLen()))
        {
            if (remove(h, current, out)) return;
        };
    }
    var i: usize = 0;
    while (i < h.queued.items.len) {
        const old = h.queued.items[i];
        if (old != current and warm(h, old) and extends(h, old, current)) {
            out.add(h.queued.orderedRemove(i));
        } else i += 1;
    }
    const cap = @max(1, @min(h.info_.lanes, lanes.Engine.max_together));
    while (true) {
        var count: usize = 0;
        var oldest: ?usize = null;
        for (h.queued.items, 0..) |job, at| if (warm(h, job)) {
            count += 1;
            if (oldest == null or job.submitted < h.queued.items[oldest.?].submitted) oldest = at;
        };
        if (count <= cap) return;
        out.add(h.queued.orderedRemove(oldest.?));
    }
}

fn remove(h: *LaneHost, job: *Job, out: *Discarded) bool {
    const at = std.mem.indexOfScalar(*Job, h.queued.items, job) orelse return false;
    out.add(h.queued.orderedRemove(at));
    return true;
}

/// Skip backend work only for an exact, live endpoint, including request cuts, decoded intervals and image frame.
pub fn skipKept(h: *LaneHost, job: *Job) bool {
    if (!warm(h, job) or job.request.untaught or LaneHost.cancelled(job)) return false;
    const store = h.cache orelse return false;
    if (store.rules.lookahead != 0) return false;
    const r = job.request;
    const key = job.cacheKey();
    if (key.len == 0) return false;
    const f = frame(h, r);
    for (store.entries.items) |e| {
        if (e.expired or e.at != key.len or e.tokens.len != key.len or e.image_frame != f or
            !@import("prompt_cache_cuts.zig").matches(store, e, r.chunks) or
            !modes.equal(e.decode_spans, h.spans(r), e.at) or !@import("cache_images.zig").equal(e.images, r.images, e.at)) continue;
        const first = if (f == 0 or job.key.len > 0) key[0] else 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(f, std.mem.asBytes(&key[0])) & 0x7fff_ffff));
        if (e.tokens[0] != first or !std.mem.eql(u32, e.tokens[1..], key[1..])) continue;
        h.remove(job);
        h.finish(job, .length, "");
        return true;
    }
    return false;
}
