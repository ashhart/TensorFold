//! The prefix wait: a cut another prompt fills waits to keep it; an identical prompt joins the whole fill.
const std = @import("std");
const api = @import("engine_api.zig");
const exports = @import("lane_export.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const modes = @import("cache_modes.zig");

/// A queued request's wait for `leader` to stand at `at` from `since`; `whole`: it covers the leader's whole fill.
pub const Waiting = struct { leader: ?*const Job = null, id: api.Id = 0, at: u32 = 0, since: i96 = 0, waited: bool = false, whole: bool = false };

/// The wait's bound (TENSORFOLD_PREFIX_WAIT_MS, 0: off) and what it did.
pub const Wait = struct {
    ns: i96 = @as(i96, default_ms) * std.time.ns_per_ms,
    waits: u64 = 0, // requests that waited for another fill's cut
    joins: u64 = 0, // of them, whole-fill joins: identical prompts that waited for a leader's whole fill
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

/// Whether `job`, picked for a lane, waits instead: a cut its lookup would resume, or its own whole fill (lock held).
pub fn waits(h: *LaneHost, job: *Job) bool {
    if (job.prefix.waited or h.prefix.ns <= 0 or !h.core.fills()) return false; // a request waits once: no cycle
    const store = h.cache orelse return false;
    const r = job.request;
    if (!keyed(r)) return false;
    const spans = h.spans(r);
    const previous_salt = store.salt;
    store.salt = r.cacheSalt();
    defer store.salt = previous_salt;
    const key = exports.resumable(r); // a handoff resumes only before its cut
    const have: u32 = if (store.find(key, r.chunks, spans)) |e| e.at else 0;
    const least = have +| @max(store.rules.min_gap, 1); // a cut this far past what is kept already
    var best: Waiting = .{};
    for (h.filling.items) |l| { // admitted fills never wait, so the oldest always runs
        if (!l.started or l.prefilled != null or l.request.background or !keyed(l.request) or l.request.cacheSalt() != r.cacheSalt()) continue;
        if (l.stream.reuse.pulled) continue; // a fresh pull keeps no mark, or only past its end (#1594 step 2b)
        // a single-flight join: the leader's prompt is the waiter's own; armed only when its fill keeps anything
        if (joinable(h, r, spans, l) and room(h, job, l) and l.marks.len > 0 and store.most != 0 and exports.keeps(l, l.cut[0])) {
            best = .{ .leader = l, .id = l.id, .whole = true };
            break; // filling order: the oldest leader
        }
        for (l.marks) |at| {
            // a pass that began past a mark (slot 0's in-place resume, unseen by the plan) never stands there
            if (at <= l.reached or at <= l.stream.cached or at < least or at <= best.at) continue;
            if (!exports.keeps(l, at)) continue; // a handoff's cut is kept only where its store resumes it
            if (store.wouldServe(key, r.chunks, spans, l.request.cacheKey(), h.spans(l.request), at))
                best = .{ .leader = l, .id = l.id, .at = at };
        }
    }
    if (best.leader == null) return false;
    best.since = std.Io.Clock.awake.now(h.io).toNanoseconds();
    best.waited = true;
    job.prefix = best;
    h.prefix.waits += 1;
    if (best.whole) h.prefix.joins += 1;
    return true;
}

/// Whether filling `l` is the waiter's own prompt: key tokens, raw tokens and spans, no mark consulted (lock held).
fn joinable(h: *LaneHost, r: *const api.Request, spans: []const [2]u32, l: *const Job) bool {
    if (r.cacheSalt() != l.request.cacheSalt()) return false;
    if (!std.mem.eql(u32, r.cacheKey(), l.request.cacheKey())) return false;
    if (!std.mem.eql(u32, r.prompt, l.request.prompt)) return false;
    return modes.equal(spans, h.spans(l.request), @intCast(l.request.prompt.len));
}

/// Whether `l` holds fewer whole-fill waiters than the admission cap (lock held).
fn room(h: *LaneHost, job: *const Job, l: *const Job) bool {
    const cap = @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes);
    var held: usize = 0;
    for (h.queued.items) |j| {
        if (j == job) continue;
        if (j.prefix.whole and j.prefix.leader == l and j.prefix.id == l.id) held += 1;
    }
    return held < cap;
}

/// A request whose states the cache keys by its tokens alone, as the host's lookup does (no pictures).
fn keyed(r: *const api.Request) bool {
    return !r.promptOnly() and r.prompt_logprobs == null and !r.untaught and r.images.len == 0 and r.audios.len == 0 and r.cacheKey().len > 0;
}

/// Waiting requests go on in queue order: the cut stood, the whole fill is done, the leader left, or the bound passed.
pub fn release(h: *LaneHost) void {
    var now: ?i96 = null;
    for (h.queued.items) |job| {
        const w = job.prefix;
        const leader = w.leader orelse continue;
        const live = for (h.filling.items) |l| {
            if (l == leader and l.id == w.id) break l; // the same job, not a later one at its address
        } else null;
        if (live) |l| {
            const pending = if (w.whole) l.prefilled == null else (l.reached < w.at and l.prefilled == null);
            if (pending) {
                const t = now orelse std.Io.Clock.awake.now(h.io).toNanoseconds();
                now = t;
                if (t - w.since < h.prefix.ns) continue;
                h.prefix.timeouts += 1;
            }
        }
        job.prefix.leader = null;
    }
}
