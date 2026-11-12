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
    fill_lent_ns: f64 = 0, // decode time its prompt has lent replies (TENSORFOLD_PREFILL_ORDER=srpt)
    began: i96 = 0,
    /// The engine thread says the prompt pass is going: status readers may count this job's fill atomics.
    fill_on: std.atomic.Value(bool) = .init(false),
    load: @import("lane_host_load.zig").Pass = .{}, // this attempt's prompt pass in the load ledger
    prefilled: ?i96 = null,
    entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
    cache_promise: @import("prompt_cache_reserve.zig").Room = .{}, // net room for this pass, until kept or ended
    marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
    frame: u32 = 0, // exact family frame, before its compact cache-word hash
    key: []u32 = &.{}, // the request's cache key with its image frame in the first word (gpa-owned); empty: its own
    kept0: u64 = 0, // the store's kept count when the job looked it up
    seq: u64 = 0, // its admission's number
    waits_since: ?u64 = null, // the admissions count when it first waited at the head for pool room
    reached: u32 = 0, // the furthest mark its prompt pass has stood at, kept or not
    prefix: @import("lane_host_prefix.zig").Waiting = .{}, // its wait for another fill's cut, while queued
    fetch: @import("lane_host_fetch.zig").Ticket = .{}, // its disk state read ahead, while queued
    lease: @import("kv_ledger.zig").Lease = .{}, // its caches' bytes in the host's ledger, from admission to its end
    remote: ?@import("lane_remote.zig").Filled = null, // a prefix another machine filled, until it is restored
    ticket: ?*@import("lane_remote.zig").Ticket = null, // its fill under way while it is queued
    remote_looked: bool = false, // queued, it was weighed for a remote fill once
    verdict: ?@import("lane_remote.zig").Verdict = null, // the remote source's last verdict on it, for the reply
    phases: @import("lane_remote.zig").Phases = .{}, // its pull's phases, once its prefix landed in the stream
    remote_said: @import("lane_remote.zig").Verdict = .small, // the last reason it was kept off the pair that the log gave
    exported: bool = false, // a handoff request's state went to the exporter (or its failure did)
    cut: [1]u32 = .{0}, // a handoff request's one chunk start: its prompt's last token

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
        job.keep(s, at, true);
    }

    /// Every keep of a job's stream at `at`: remote on a remote prefix, within the queue's promise when `promised`.
    pub fn keep(job: *Job, s: *lanes.Stream, at: u32, promised: bool) void {
        const store = job.host.cache orelse return;
        store.keep_remote = s.reuse.remote and !s.reuse_failed; // on another machine's prefix (a refused copy prefilled here)
        defer store.keep_remote = false;
        if (promised) return @import("lane_cache_queue.zig").keep(job.host, job, s, at);
        _ = store.keepBornFrame(job.cacheKey(), job.request.bornLen(), at, s, job.request.chunks, job.host.spans(job.request), job.request.images, job.frame);
    }

    /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
    pub fn reported(job: *Job) void {
        @import("lane_remote.zig").drop(job);
        @import("lane_export.zig").unexported(job);
        const e = job.entry orelse return;
        job.entry = null;
        const store = job.host.cache orelse return;
        if (!job.started) return store.release(e);
        job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
        const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
        if (job.stream.reuse_failed or ok) store.resumedWithReason(e, job.cacheKey(), if (ok) null else job.stream.reuse_error orelse error.RestoreRejected) else store.release(e);
    }
};
