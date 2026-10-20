//! A request's stream on the lane host: its kept prefix looked up, its drafter made, the stream set up.
const std = @import("std");
const lanes = @import("lanes");
const host = @import("lane_host.zig");
const recalls = @import("lane_host_recall.zig");
const LaneHost = host.LaneHost;
const Job = LaneHost.Job;

/// A job's drafter freed: the family's (LaneHost.proposers) or the built-in one.
pub fn freeProposer(h: *LaneHost, job: *Job) void {
    if (job.custom) |c| h.proposers.?.free(h.proposers.?.ctx, c) else job.proposer.deinit();
    job.custom = null;
}

/// An admitted job's stream (its kept prefix looked up, its drafter): false when it failed (the job is finished).
pub fn open(h: *LaneHost, job: *Job) bool {
    const r = job.request;
    var reuse: lanes.stream.Reuse = .{};
    // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
    if (h.frame_key) |frame| if (h.cache != null and r.images.len > 0 and r.cacheKey().len > 0) {
        const f = frame(r.images);
        if (f != 0) {
            job.key = h.gpa.dupe(u32, r.cacheKey()) catch return !h.drop(job, "out of memory");
            job.key[0] = 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(f, std.mem.asBytes(&job.key[0])) & 0x7fff_ffff));
        }
    };
    if (h.cache) |store| if (r.embedding == null and r.prompt_logprobs == null and !r.untaught) if (store.lookupRewind(h.gpa, job.cacheKey(), r.bornLen(), r.history_len, r.rewind_len, r.shared_prefixes, r.chunks, h.spans(r))) |l| {
        job.entry = l.entry;
        job.kept0 = store.counts.kept;
        job.marks = l.marks;
        reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
    } else |_| {};
    if (h.proposers) |ps| {
        job.custom = ps.make(ps.ctx, h.gpa, r.prompt, r.eos) catch return !h.drop(job, "the drafter could not start");
    } else job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return !h.drop(job, "the drafter could not start");
    const recall = recalls.reader(h, job);
    job.stream = lanes.Stream.init(h.gpa, .{
        .id = "request",
        .prompt = r.prompt,
        .embedding = r.embedding,
        .max_new = r.max_tokens,
        .eos = r.eos,
        .sampling = r.sampling,
        .drafts = r.drafts,
        .proposer = job.custom orelse job.proposer.proposer(),
        .recall = recall,
        .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
        .cancel_check = .{ .ptr = job, .check = LaneHost.cancelled },
        .yield_check = if (r.background and r.max_tokens == 0) .{ .ptr = job, .check = LaneHost.yieldsTo } else null,
        .think_budget = r.think_budget,
        .think_close = r.think_close,
        .think_end = if (r.think_end) |t| t else -1,
        .answer_max = r.answer_max,
        .logprobs = r.logprobs,
        .prompt_logprobs = r.prompt_logprobs,
        .loop_guard = r.loop_guard,
        .chunks = r.chunks,
        .history_len = r.history_len,
        .shared_prefixes = r.shared_prefixes,
        .reuse = reuse,
        .grammar = if (r.structure) |st| h.constraint(st) catch |e| {
            freeProposer(h, job);
            recalls.release(job);
            return !h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
        } else null,
        .decode_spans = h.spans(r),
        .decide = r.decide,
        .untaught = r.untaught,
        .images = r.images,
        .key = r.cache_key,
    }) catch {
        freeProposer(h, job);
        recalls.release(job);
        return !h.drop(job, "out of memory");
    };
    job.started = true;
    job.fill_on.store(true, .release); // status readers count the fill from here until the pass reports
    job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
    return true;
}
