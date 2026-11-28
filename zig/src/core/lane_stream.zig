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
    // A family may skip image warm passes before any cache lookup or backend call.
    if (!h.core.backend.warm_images and h.info_.warm_turns and r.background and r.max_tokens == 0 and r.images.len > 0) return !h.cancel(job);
    var reuse: lanes.stream.Reuse = .{};
    var effective_prompt: []const u32 = r.prompt;
    var effective_max_new: u32 = r.max_tokens;
    var memory_active = false;
    // one version for the whole request: memory ops run on this thread, so none lands between this read and the route
    if (h.folded_memory) |fm| if (h.info_.folded_memory) if (fm.version) |v| {
        job.memory_version = v(fm.ctx);
    };
    // the latest user turn's memory gate, only where the engine serves learning (Info.folded_memory, --slide)
    if (h.folded_memory) |fm| if (h.info_.folded_memory and !job.memory_refused and fm.hasUnits(fm.ctx) and !r.untaught and r.memory_query.len > 0 and r.images.len == 0 and !r.promptOnly() and r.prompt_logprobs == null) {
        if (fm.routeAndCompose(fm.ctx, r.memory_query)) |maybe_snap| {
            if (maybe_snap) |handle| {
                const strip: usize = if (r.memory_strip < r.prompt.len) r.memory_strip else 0;
                const tail = r.prompt[strip..];
                const total_len = @as(usize, handle.at) + tail.len;
                const ctx_win: usize = if (h.info_.context_window > 0) h.info_.context_window else std.math.maxInt(usize);
                if (total_len + 1 < ctx_win) {
                    // the prefix is priced as any prompt is: a job its promise cannot cover is refused by name
                    if (!@import("lane_host_ledger.zig").grow(h, job, total_len)) {
                        fm.dropSnap(fm.ctx, handle.saved);
                        return !h.drop(job, @import("lane_host_ledger.zig").memory_unfit);
                    }
                    if (h.gpa.alloc(u32, total_len)) |buf| {
                        @memset(buf[0..handle.at], 0);
                        @memcpy(buf[handle.at..], tail);
                        job.memory_prompt = buf;
                        job.memory_snap = handle.saved;
                        effective_prompt = buf;
                        if (ctx_win < std.math.maxInt(usize)) {
                            const room: u32 = @intCast(ctx_win - total_len);
                            effective_max_new = @min(effective_max_new, room);
                        }
                        reuse = .{ .saved = handle.saved, .at = handle.at, .marks = &.{}, .hook = null };
                        memory_active = true;
                    } else |_| {
                        fm.dropSnap(fm.ctx, handle.saved);
                    }
                } else {
                    fm.dropSnap(fm.ctx, handle.saved);
                }
            }
        } else |err| {
            std.log.warn("folded memory: route/compose failed ({s}); serving unmodified prompt", .{@errorName(err)});
        }
    };
    // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
    if (!memory_active) if (h.frame_key) |frame| if (h.cache != null and r.images.len > 0 and r.cacheKey().len > 0) {
        const f = frame(r.images);
        job.frame = f;
        if (f != 0) {
            job.key = h.gpa.dupe(u32, r.cacheKey()) catch return !h.drop(job, "out of memory");
            job.key[0] = 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(f, std.mem.asBytes(&job.key[0])) & 0x7fff_ffff));
        }
    };
    if (!memory_active and h.cache != null and !r.promptOnly() and r.prompt_logprobs == null and !r.untaught) reuse = kept(h, job);
    if (h.proposers) |ps| {
        job.custom = ps.make(ps.ctx, h.gpa, r.prompt, r.eos) catch return !h.drop(job, "the drafter could not start");
    } else job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return !h.drop(job, "the drafter could not start");
    const recall = if (memory_active) null else recalls.reader(h, job);
    // Ours until Stream.init succeeds: the stream only borrows it, and a stream that never starts never frees it.
    const reply_grammar = if (r.structure) |st| h.constraint(st) catch |e| {
        freeProposer(h, job);
        recalls.release(job);
        return !h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
    } else null;
    job.stream = lanes.Stream.init(h.gpa, .{
        .id = "request",
        .prompt = effective_prompt,
        .embedding = r.embedding,
        .readout = r.readout,
        .max_new = effective_max_new,
        .eos = r.eos,
        .sampling = r.sampling,
        .penalties = r.penalties,
        .logit_bias = r.logit_bias,
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
        .loop_stop = r.loop_stop,
        .chunks = if (memory_active) &.{} else r.chunks,
        .message_starts = if (memory_active) &.{} else r.message_starts,
        .history_len = if (memory_active) 0 else r.history_len,
        .shared_prefixes = if (memory_active) &.{} else r.shared_prefixes,
        .reuse = reuse,
        .grammar = reply_grammar,
        .decode_spans = if (memory_active) &.{} else h.spans(r),
        .decide = r.decide,
        .untaught = r.untaught,
        .images = r.images,
        .audios = r.audios,
        .key = if (memory_active) &.{} else r.cache_key,
        .cache_salt = r.cacheSalt(),
        .memory_active = memory_active,
    }) catch {
        freeProposer(h, job);
        recalls.release(job);
        if (reply_grammar) |g| g.deinit(h.gpa);
        return !h.drop(job, "out of memory");
    };
    job.started = true;
    @import("lane_host_load.zig").began(h, job); // the fill shows here
    job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
    return true;
}

/// The job's kept state looked up: its entry borrowed, its marks and promise planned from what it resumes.
fn kept(h: *LaneHost, job: *Job) lanes.stream.Reuse {
    const store = h.cache.?;
    const l = @import("lane_cache_queue.zig").lookup(h, job) catch return .{};
    job.entry = l.entry;
    job.kept0 = store.counts.kept;
    job.marks = l.marks;
    return .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept, .interrupted = Job.interrupted } };
}

/// A lent state gone at the job's first call: looked up again as at admission, with its fill and load hint.
pub fn lookAgain(h: *LaneHost, job: *Job) void {
    h.gpa.free(job.marks);
    job.marks = &.{};
    job.cache_promise = .{};
    job.stream.reuse = kept(h, job);
    job.stream.fill_at.store(job.stream.reuse.at, .monotonic);
    h.lock();
    defer h.unlock();
    job.load.hint = job.stream.reuse.at;
    job.load.sure = job.load.hint == 0;
}
