//! A lane host's Recall: each drafted stream's reader of the record, and the prompts and replies its requests add.
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const Store = lanes.recall_store.Store;

/// A job's Recall: its reader of the record while `reading`, and the counters its reply's stats carry.
pub const Slot = struct {
    reader: lanes.recall.Reader = undefined,
    reading: bool = false,
    telemetry: [256]u8 = undefined,
};

/// The record later requests draft from and add to, set once before the server takes requests.
pub fn attach(ctx: *anyopaque, store: *Store) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.lock();
    defer h.unlock();
    h.recall = store;
}

/// A drafted request's reader of the record (null: no record, or a request that drafts nothing).
pub fn reader(h: *LaneHost, job: *Job) ?*lanes.recall.Reader {
    const store = h.recall orelse return null;
    if (!job.request.drafts) return null;
    job.recall.reader = .{ .gpa = h.gpa, .shared = &store.shared };
    job.recall.reading = true;
    return &job.recall.reader;
}

/// Its prompt and reply go in unless it is a background pass (a reply prefilled again) or a lesson's untaught request.
fn writes(job: *const Job) bool {
    return !job.request.background and !job.request.untaught;
}

/// The prompt pass ended: what came before its resume point an earlier request read already.
pub fn prompted(h: *LaneHost, job: *Job) void {
    const store = h.recall orelse return;
    if (writes(job)) store.readPrompt(job.request.prompt, job.stream.cached);
}

/// The stream ended: its reply goes in after its prompt's tail; `stats` with its reader's counters for the reply.
pub fn finished(h: *LaneHost, job: *Job, reason: api.Reason, stats: api.Stats) api.Stats {
    var told = stats;
    const store = h.recall orelse return told;
    if (reason != .failed and writes(job)) store.readReply(job.request.prompt, job.stream.emitted());
    if (job.recall.reading) told.telemetry_json = job.recall.reader.telemetry(&job.recall.telemetry);
    return told;
}

/// A job whose stream is torn down (finished, or requeued for memory) frees its reader.
pub fn release(job: *Job) void {
    if (!job.recall.reading) return;
    job.recall.reader.deinit();
    job.recall.reading = false;
}
