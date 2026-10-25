//! The prefix wait: a request whose prompt shares a cut another prompt is filling waits for that fill to keep it.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// A queued request's wait for `leader` (a filling job, while it is still named `id`) to stand at `at`, from `since`.
pub const Waiting = struct { leader: ?*const Job = null, id: api.Id = 0, at: u32 = 0, since: i96 = 0, waited: bool = false };

/// The wait's bound (TENSORFOLD_PREFIX_WAIT_MS, 0: off) and what it did.
pub const Wait = struct {
    ns: i96 = @as(i96, default_ms) * std.time.ns_per_ms,
    waits: u64 = 0, // requests that waited for another fill's cut
    timeouts: u64 = 0, // of them, released at the bound with the cut not reached

    pub const default_ms: u32 = 60_000;

    pub fn fromEnv() Wait {
        return fromText(std.mem.span(std.c.getenv("TENSORFOLD_PREFIX_WAIT_MS") orelse return .{}));
    }

    /// A bound from its text: whole milliseconds, 0 off; anything else is the default.
    pub fn fromText(text: []const u8) Wait {
        const t = std.mem.trim(u8, text, " \t");
        if (t.len == 0 or t[0] < '0' or t[0] > '9') return .{};
        const ms = std.fmt.parseInt(u32, t, 10) catch return .{};
        return .{ .ns = @as(i96, ms) * std.time.ns_per_ms };
    }
};

/// Whether queued `job` waits for another fill's cut (the lock held).
pub fn waiting(job: *const Job) bool {
    return job.prefix.leader != null;
}

/// Whether `job`, picked for a lane, waits instead: a fill short of a cut its lookup would resume (lock held).
pub fn waits(h: *LaneHost, job: *Job) bool {
    if (job.prefix.waited or h.prefix.ns <= 0 or !h.core.fills()) return false; // a request waits once: no cycle
    const store = h.cache orelse return false;
    const r = job.request;
    if (!keyed(r)) return false;
    const spans = h.spans(r);
    const have: u32 = if (store.find(r.cacheKey(), r.chunks, spans)) |e| e.at else 0;
    const least = have +| @max(store.rules.min_gap, 1); // a cut this far past what is kept already
    var best: Waiting = .{};
    for (h.filling.items) |l| { // admitted fills never wait, so the oldest always runs
        if (!l.started or l.prefilled != null or l.request.background or !keyed(l.request)) continue;
        for (l.marks) |at| {
            if (at <= l.reached or at < least or at <= best.at) continue;
            if (store.wouldServe(r.cacheKey(), r.chunks, spans, l.request.cacheKey(), h.spans(l.request), at))
                best = .{ .leader = l, .id = l.id, .at = at };
        }
    }
    if (best.leader == null) return false;
    best.since = std.Io.Clock.awake.now(h.io).toNanoseconds();
    best.waited = true;
    job.prefix = best;
    h.prefix.waits += 1;
    return true;
}

/// A request whose states the cache keys by its tokens alone, as the host's lookup does (no pictures).
fn keyed(r: *const api.Request) bool {
    return r.embedding == null and r.prompt_logprobs == null and !r.untaught and r.images.len == 0 and r.cacheKey().len > 0;
}

/// Waiting requests go on in queue order: their fill stood at the cut, left the fills, or the bound passed (lock held).
pub fn release(h: *LaneHost) void {
    var now: ?i96 = null;
    for (h.queued.items) |job| {
        const w = job.prefix;
        const leader = w.leader orelse continue;
        const live = for (h.filling.items) |l| {
            if (l == leader and l.id == w.id) break l; // the same job, not a later one at its address
        } else null;
        if (live) |l| if (l.reached < w.at and l.prefilled == null) {
            const t = now orelse std.Io.Clock.awake.now(h.io).toNanoseconds();
            now = t;
            if (t - w.since < h.prefix.ns) continue;
            h.prefix.timeouts += 1;
        };
        job.prefix.leader = null;
    }
}
