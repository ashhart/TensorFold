//! One lane job owns its stream metadata and reports restored cache entries once.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const recalls = @import("lane_host_recall.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;

pub const Job = struct {
    host: *LaneHost,
    id: Id,
    request: *const Request,
    sink: Sink,
    stream: lanes.Stream = undefined,
    proposer: lanes.SuffixLookup = undefined,
    recall: recalls.Slot = .{}, // its reads of Recall's record (lane_host_recall.zig)
    /// the family's proposer (LaneHost.proposers), in place of `proposer`
    custom: ?lanes.proposer.Proposer = null,
    delivered: usize = 0,
    started: bool = false,
    prefill_sent: bool = false, // a lone driver's prefilled event went out
    fill_began: bool = false, // its first prompt chunk ran
    submitted: i96 = 0,
    admitted_ns: ?i96 = null, // first admission only, retained if a background pass later yields
    fill_wait: usize = 0, // chunk choices it has waited since admission or its last chunk
    began: i96 = 0,
    /// The engine thread says the prompt pass is going: status readers may count this job's fill atomics.
    fill_on: std.atomic.Value(bool) = .init(false),
    prefilled: ?i96 = null,
    entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
    marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
    key: []u32 = &.{}, // the request's cache key with its image frame in the first word (gpa-owned); empty: its own
    kept0: u64 = 0, // the store's kept count when the job looked it up
    seq: u64 = 0, // its admission's number
    waits_since: ?u64 = null, // the admissions count when it first waited at the head for pool room
    reached: u32 = 0, // the furthest mark its prompt pass has stood at, kept or not
    prefix: @import("lane_host_prefix.zig").Waiting = .{}, // its wait for another fill's cut, while queued

    /// Record first admission once, including a family planner and a background pass that later yields.
    pub fn noteAdmission(job: *Job) void {
        if (job.admitted_ns == null) job.admitted_ns = std.Io.Clock.awake.now(job.host.io).toNanoseconds();
    }

    /// The words the cache names this request's states by.
    pub fn cacheKey(job: *const Job) []const u32 {
        return if (job.key.len > 0) job.key else job.request.cacheKey();
    }

    /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
    pub fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const job: *Job = @ptrCast(@alignCast(ptr));
        job.reached = @max(job.reached, at); // a request waiting for this cut goes on, whether the store keeps it or not
        job.reported(); // before a keep can evict the entry the pass restored
        if (job.host.cache) |store| _ = store.keepBorn(job.cacheKey(), job.request.bornLen(), at, s, job.request.chunks, job.host.spans(job.request));
    }

    /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
    pub fn reported(job: *Job) void {
        const e = job.entry orelse return;
        job.entry = null;
        const store = job.host.cache orelse return;
        if (!job.started) return store.release(e);
        job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
        const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
        if (job.stream.reuse_failed or ok) store.resumedWithReason(e, job.cacheKey(), if (ok) null else job.stream.reuse_error orelse error.RestoreRejected) else store.release(e);
    }
};
