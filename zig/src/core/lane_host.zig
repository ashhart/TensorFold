//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const status = @import("lane_host_status.zig");
const learn = @import("lane_host_learn.zig");
const lone = @import("lane_host_lone.zig");
const fill = @import("lane_host_fill.zig");
const admit = @import("lane_host_admit.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Reason = api.Reason;
const Stats = api.Stats;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    filling: std.ArrayList(*Job) = .empty, // admitted jobs whose prompt is still going in, a chunk a round, oldest first
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(status.Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    decoded_rows: bool = false, // the backend prefills Request.decode_spans with decoded rows' arithmetic, cache or not
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    learner: ?api.Learner = null, // the family's Sliding Weights learner; null: learn requests are refused
    lessons: std.ArrayList(Lesson) = .empty, // learn requests waiting for an idle engine
    lesson_open: bool = false, // the learner holds a begun job (engine thread only)
    admissions: u64 = 0, // jobs admitted so far: each job's `seq`
    memory_lanes: ?usize = null, // lanes the backend had memory for at its last LaneMemory, until a job ends
    warm_quiet: u64 = 0, // ns a warm pass waits for other work before its first chunk (0: none), set by its family

    const Lesson = struct { request: *const api.LearnRequest, sink: api.LearnSink };

    pub const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        fill_began: bool = false, // its first prompt chunk ran
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up
        seq: u64 = 0, // its admission's number
        waits_since: ?u64 = null, // the admissions count when it first waited at the head for pool room

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.host.cache) |store| _ = store.keepBorn(job.request.prompt, born(job.request), at, s, job.request.chunks, job.host.spans(job.request));
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            if (job.stream.reuse_failed) store.resumed(e, job.request.prompt, false) else if (job.stream.cached == e.at) store.resumed(e, job.request.prompt, true);
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    pub fn start(h: *LaneHost) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        const closed: api.LearnEvent = .{ .done = .{ .message = "the engine closed" } };
        for (h.lessons.items) |l| l.sink.event(l.sink.ctx, &closed);
        h.lessons.deinit(h.gpa);
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.filling.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = status.read, .memory = memoryFn, .keepalive = keepaliveFn, .learn = learn.submit } };
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) api.SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    pub fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// A background pass's yield hook: a foreground request waits or a stream decodes (read under the lock).
    fn yieldsTo(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.foregroundWaiting() or job.host.core.activeCount() > 0;
    }

    /// A foreground request is queued (they queue ahead of every background one); the caller holds the lock.
    pub fn foregroundWaiting(h: *const LaneHost) bool {
        return h.queued.items.len > 0 and !h.queued.items[0].request.background;
    }

    /// Waits `warm_quiet` for other work (a request, a decoding stream, a cancel): true if none came.
    fn quiet(h: *LaneHost, job: *const Job) bool {
        const until: std.Io.Timeout = .{ .deadline = .fromNow(h.io, .{ .raw = .fromNanoseconds(h.warm_quiet), .clock = .awake }) };
        h.lock();
        defer h.unlock();
        while (!h.closing and !h.foregroundWaiting() and h.core.activeCount() == 0 and std.mem.indexOfScalar(Id, h.cancels.items, job.id) == null) {
            h.wake.waitTimeout(h.io, &h.mutex, until) catch |e| return e == error.Timeout;
        }
        return false;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery (their logprob rows first).
    pub fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            if (job.stream.logprobs != null) emit(job, .{ .logprobs = job.stream.rows.items[job.delivered..emitted.len] });
            emit(job, .{ .tokens = emitted[job.delivered..] });
            status.noteDecoded(h, emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    pub fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, job.stream.problem);
        return true;
    }

    pub fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        h.memory_lanes = null; // a lane may have freed: the next admission asks the backend for memory again
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        var dropped: std.ArrayList(*Job) = .empty;
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                fill.unfill(h, job);
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    pub fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    pub fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    /// The next queued request into the admitted ones (null: none waits, no lane is free, or its caches do not fit yet).
    fn takeJob(h: *LaneHost) ?*Job {
        h.lock();
        if (h.queued.items.len == 0 or h.admitted.items.len >= @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes)) {
            h.unlock();
            return null;
        }
        const at = admit.pick(h) orelse { // a shared cache pool's head waits for room
            h.unlock();
            return null;
        };
        const job = h.queued.orderedRemove(at);
        job.seq = h.admissions;
        h.admissions += 1;
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return null;
        };
        h.unlock();
        return job;
    }

    /// The admitted request's stream (its kept state, drafter and grammar); false when it was dropped instead.
    fn setup(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        if (r.background and r.max_tokens == 0 and h.warm_quiet > 0 and !h.quiet(job)) return h.cancel(job); // other work came first: it never ran
        var reuse: lanes.stream.Reuse = .{};
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        if (h.cache) |store| if (r.prompt_logprobs == null and !r.untaught) if (store.lookupRewind(h.gpa, r.prompt, born(r), r.history_len, r.rewind_len, r.shared_prefixes, r.chunks, h.spans(r))) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch {
            _ = h.drop(job, "the drafter could not start");
            return false;
        };
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .yield_check = if (r.background and r.max_tokens == 0) .{ .ptr = job, .check = yieldsTo } else null,
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .logprobs = r.logprobs,
            .prompt_logprobs = r.prompt_logprobs,
            .loop_guard = r.loop_guard,
            .chunks = r.chunks,
            .history_len = r.history_len,
            .shared_prefixes = r.shared_prefixes,
            .reuse = reuse,
            .grammar = if (r.structure) |st| h.constraint(st) catch |e| {
                job.proposer.deinit();
                {
                _ = h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
                return false;
            }
            } else null,
            .images = r.images,
            .decode_spans = h.spans(r),
            .decide = r.decide,
            .untaught = r.untaught,
        }) catch {
            job.proposer.deinit();
            {
                _ = h.drop(job, "out of memory");
                return false;
            }
        };
        job.started = true;
        job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        return true;
    }

    /// An admitted, set-up request into the engine: a warm pass, the lone driver, the fill queue, or its own prompt pass.
    fn launch(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const began = job.began;
        if (r.max_tokens == 0 and h.info_.warm_turns) return h.warmPass(job, began);
        if (lone.fits(h, job)) return lone.run(h, job, began, false);
        if (h.core.fills()) {
            h.filling.append(h.gpa, job) catch return h.drop(job, "out of memory");
            return true;
        }
        h.core.addStream(&job.stream) catch |e| return switch (e) {
            error.Cancelled => h.cancel(job),
            error.LaneMemory => h.requeue(job),
            else => h.drop(job, h.words(e)),
        };
        h.prefilled(job, began);
        if (h.deliver(job)) h.remove(job);
        return true;
    }
    fn admitOne(h: *LaneHost) bool {
        const job = h.takeJob() orelse return false;
        if (!h.setup(job)) return true;
        return h.launch(job);
    }

    /// Every queued request a lane is free for, their prompts in one pass where the backend takes several
    /// (lanes.Engine.addStreams); the engine's special paths (chunked prompts, warm passes, the lone driver) go
    /// through admitOne as before.
    fn admitMany(h: *LaneHost) bool {
        if (!h.core.batchesPrompts() or h.core.fills()) return h.admitOne();
        h.lock();
        const waiting = h.queued.items.len;
        const free = h.info_.lanes -| h.admitted.items.len;
        h.unlock();
        if (waiting < 2 or free < 2) return h.admitOne();
        var jobs: [lanes.Engine.max_together]*Job = undefined;
        var n: usize = 0;
        while (n < jobs.len and n < free) {
            const job = h.takeJob() orelse break;
            if (!h.setup(job)) continue;
            if ((job.request.max_tokens == 0 and h.info_.warm_turns) or lone.fits(h, job)) {
                _ = h.launch(job);
                continue;
            }
            jobs[n] = job;
            n += 1;
        }
        if (n == 0) return true;
        if (n == 1) return h.launch(jobs[0]);
        var streams: [lanes.Engine.max_together]*lanes.Stream = undefined;
        for (jobs[0..n], 0..) |job, i| streams[i] = &job.stream;
        var errs: [lanes.Engine.max_together]?anyerror = undefined;
        h.core.addStreams(streams[0..n], errs[0..n]);
        for (jobs[0..n], errs[0..n]) |job, err| {
            if (err) |e| {
                _ = switch (e) {
                    error.Cancelled => h.cancel(job),
                    error.LaneMemory => h.requeue(job),
                    else => h.drop(job, h.words(e)),
                };
                continue;
            }
            h.prefilled(job, job.began);
            if (h.deliver(job)) h.remove(job);
        }
        return true;
    }


    /// The next filling job's next prompt chunk: the other streams' rounds run between chunks, not after the prompt.
    fn fillOne(h: *LaneHost) void {
        if (h.filling.items.len == 0) return;
        const at = fill.next(h);
        const job = h.filling.items[at];
        const first = !job.fill_began;
        job.fill_began = true;
        const done = h.core.fillStream(&job.stream, first) catch |e| {
            _ = switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => blk: {
                    fill.unfill(h, job);
                    break :blk h.requeue(job);
                },
                else => h.drop(job, h.words(e)),
            };
            return;
        };
        if (!done) return;
        _ = h.filling.orderedRemove(at);
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }

    /// A reply's grammar from the server's compiler (Info.structures engines get requests that carry one).
    fn constraint(h: *LaneHost, st: api.Structure) !*lanes.grammar.Constraint {
        const c = st.compiler orelse return error.NoGrammarCompiler;
        return c.constraint(h.gpa, h.io, @fromBackingInt(@intCast(@backingInt(st.kind))), st.text, st.after);
    }

    pub fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        if (h.cache) |store| store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        emit(job, .{ .prefilled = job.stream.cached });
        if (job.stream.prompt_logprobs != null) emit(job, .{ .prompt_logprobs = job.stream.prompt_rows.items });
    }

    /// A request's decoded prompt rows, for a backend that keeps their bits (`decoded_rows`); otherwise none.
    fn spans(h: *const LaneHost, r: *const Request) []const [2]u32 {
        return if (h.decoded_rows) r.decode_spans else &.{};
    }

    /// A prompt-only pass on a `warm_turns` engine: the backend prefills, the cache keeps its end (a mark, if keyed on)
    fn warmPass(h: *LaneHost, job: *Job, began: i96) bool {
        const be = h.core.backend;
        be.prefill(&job.stream) catch |e| { // a yield ends it like a cancel: what it prefilled is kept
            if (e != error.Cancelled and e != error.Yielded) return h.drop(job, h.words(e));
            be.release(&job.stream);
            return h.cancel(job);
        };
        h.prefilled(job, began);
        if (h.cache) |store| _ = if (!job.request.untaught and store.rules.lookahead == 0) store.keepBorn(job.request.prompt, born(job.request), @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request)) else false;
        be.release(&job.stream);
        h.remove(job);
        h.finish(job, .length, "");
        return true;
    }

    pub fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    pub fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// No memory for another lane yet: the job waits at the queue's head, and admission holds until a job ends.
    fn requeue(h: *LaneHost, job: *Job) bool {
        if (job.stream.grammar) |g| g.deinit(h.gpa); // the next admission compiles its own
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        job.started = false;
        job.fill_began = false;
        job.entry = null; // looked up again at its next admission
        h.gpa.free(job.marks);
        job.marks = &.{};
        h.lock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            break;
        };
        const others = h.admitted.items.len;
        const waits = others > 0 and blk: { // with nothing else running no memory comes back: it would wait forever
            h.queued.insert(h.gpa, 0, job) catch break :blk false;
            break :blk true;
        };
        h.unlock();
        if (!waits) {
            h.finish(job, .failed, if (others > 0) "out of memory" else alone_words);
            return true;
        }
        h.memory_lanes = others;
        return false;
    }

    const alone_words = "the engine has no memory for this request's caches with nothing else running: lower --context or --parallel";

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    pub fn remove(h: *LaneHost, job: *Job) void {
        fill.unfill(h, job);
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn run(h: *LaneHost) void {
        while (true) {
            h.takeCancels();
            while (h.admitMany()) {}
            h.fillOne();
            status.noteLive(h);
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (h.lesson_open) {
                    h.learner.?.abort(h.learner.?.ctx);
                    h.lesson_open = false;
                }
                if (left == 0) return;
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (h.filling.items.len > 0) {
                    h.unlock();
                    continue;
                }
                if (learn.ready(h)) {
                    h.unlock();
                    learn.step(h);
                    continue;
                }
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
            lone.reclaim(h);
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.filling.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        h.unlock();
        h.takeCancels();
    }
};

/// The prompt length a request's kept states count as: its own, or for a warm pass the request whose reply it prefills.
fn born(r: *const Request) u32 {
    return if (r.born > 0) r.born else @intCast(r.prompt.len);
}

test {
    _ = @import("lane_host_test.zig");
}
