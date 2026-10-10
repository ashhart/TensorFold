//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const status = @import("lane_host_status.zig");
const learn = @import("lane_host_learn.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const Rounds = rounds.Rounds;
pub const Proposers = rounds.Proposers;
const rounds = @import("lane_rounds.zig");
const lane_stream = @import("lane_stream.zig");

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    /// A family's own copy proposer for each request (null: lanes.SuffixLookup at `min_match`)
    proposers: ?Proposers = null,
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
    /// Runs first on the engine thread: a backend whose API state is per thread (a CUDA context) binds it there.
    on_thread: ?ThreadInit = null,
    /// opt-in: a family round planner (prompt pieces, decode rows in one pass); null: whole or `prefill_step` prompts
    rounds: ?Rounds = null,
    pieces: std.ArrayList(Rounds.Piece) = .empty,
    finals: std.ArrayList(usize) = .empty,

    pub const ThreadInit = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

    const Lesson = struct { request: *const api.LearnRequest, sink: api.LearnSink };

    pub const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        /// the family's proposer (LaneHost.proposers), in place of `proposer`
        custom: ?lanes.proposer.Proposer = null,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        fill_began: bool = false, // its first prompt chunk ran
        submitted: i96 = 0,
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        pub fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.host.cache) |store| _ = store.keep(job.request.prompt, at, s, job.request.chunks, job.host.spans(job.request));
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
        if (h.rounds != null and !h.core.backend.pieces()) return error.PiecesUnsupported; // a drafter wrapper runs whole prompts
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
        h.pieces.deinit(h.gpa);
        h.finals.deinit(h.gpa);
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

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink, .submitted = std.Io.Clock.awake.now(h.io).toNanoseconds() };
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

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    pub fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// A background pass's yield hook: a foreground request waits or a stream decodes (read under the lock).
    pub fn yieldsTo(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.foregroundWaiting() or job.host.core.activeCount() > 0;
    }

    /// A foreground request is queued (they queue ahead of every background one); the caller holds the lock.
    fn foregroundWaiting(h: *const LaneHost) bool {
        return h.queued.items.len > 0 and !h.queued.items[0].request.background;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery (their logprob rows first).
    fn send(h: *LaneHost, job: *Job) void {
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
        if (h.rounds) |rd| rd.vtable.left(rd.ctx, @intFromPtr(job));
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            lane_stream.freeProposer(h, job);
        }
        h.gpa.destroy(job);
    }

    /// Cancels queued and admitted jobs named since the last round.
    pub fn takeCancels(h: *LaneHost) void {
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
                h.unfill(job);
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
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        if (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return false;
        }
        if (h.info_.admit) |a| { // a shared cache pool: the next request waits for room (always there when idle)
            const r = h.queued.items[0].request;
            if (h.core.activeCount() > 0 and !a.fits(a.ctx, @intCast(r.prompt.len), r.max_tokens)) {
                h.unlock();
                return false;
            }
        }
        const job = h.queued.orderedRemove(0);
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        if (!lane_stream.open(h, job)) return true;
        const began = job.began;
        if (job.request.max_tokens == 0 and h.info_.warm_turns) return h.warmPass(job, began);
        if (h.loneFits(job)) return h.runLone(job, began);
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

    /// The oldest filling job's next prompt chunk: the other streams' rounds run between chunks, not after the prompt.
    fn fillOne(h: *LaneHost) void {
        if (h.filling.items.len == 0) return;
        const job = h.filling.items[0];
        const first = !job.fill_began;
        job.fill_began = true;
        const done = h.core.fillStream(&job.stream, first) catch |e| {
            _ = switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => blk: {
                    h.unfill(job);
                    break :blk h.requeue(job);
                },
                else => h.drop(job, h.words(e)),
            };
            return;
        };
        if (!done) return;
        _ = h.filling.orderedRemove(0);
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }

    fn unfill(h: *LaneHost, job: *Job) void {
        for (h.filling.items, 0..) |j, i| if (j == job) {
            _ = h.filling.orderedRemove(i);
            return;
        };
    }

    /// A reply's grammar from the server's compiler (Info.structures engines get requests that carry one).
    pub fn constraint(h: *LaneHost, st: api.Structure) !*lanes.grammar.Constraint {
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
    }

    /// A request's decoded prompt rows, for a backend that keeps their bits (`decoded_rows`); otherwise none.
    pub fn spans(h: *const LaneHost, r: *const Request) []const [2]u32 {
        return if (h.decoded_rows) r.decode_spans else &.{};
    }

    /// A prompt-only pass on a `warm_turns` engine: the backend prefills and the cache keeps where it ends.
    fn warmPass(h: *LaneHost, job: *Job, began: i96) bool {
        const be = h.core.backend;
        be.prefill(&job.stream) catch |e| { // a yield ends it like a cancel: what it prefilled is kept
            if (e != error.Cancelled and e != error.Yielded) return h.drop(job, h.words(e));
            be.release(&job.stream);
            return h.cancel(job);
        };
        h.prefilled(job, began);
        if (h.cache) |store| _ = store.keep(job.request.prompt, @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request));
        be.release(&job.stream);
        h.remove(job);
        h.finish(job, .length, "");
        return true;
    }

    /// An idle backend driver takes a lone drafted request, sampled when supported; never one with logprobs.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null or r.logprobs != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and !h.foregroundWaiting() and h.cancels.items.len == 0 and h.core.activeCount() == 0;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                status.noteLive(host);
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.foregroundWaiting() or host.cancels.items.len > 0 or host.closing;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
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

    /// No memory for another lane yet: the job waits at the queue's head, and admission holds until a lane frees.
    fn requeue(h: *LaneHost, job: *Job) bool {
        job.stream.deinit(h.gpa);
        lane_stream.freeProposer(h, job);
        job.started = false;
        job.fill_began = false;
        job.entry = null; // looked up again at its next admission
        h.gpa.free(job.marks);
        job.marks = &.{};
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            break;
        };
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            h.lock();
            return true;
        };
        if (h.admitted.items.len > 0) h.info_.lanes = @intCast(h.admitted.items.len);
        return false;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    pub fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    pub fn remove(h: *LaneHost, job: *Job) void {
        h.unfill(job);
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn run(h: *LaneHost) void {
        if (h.on_thread) |t| t.run(t.ctx) catch |e| std.log.err("lane host: the engine thread's init failed ({t})", .{e});
        // a family planner: prompt pieces and decode rows in one pass; else whole or `prefill_step` chunked prompts
        if (h.rounds) |rd| return rounds.run(h, rd);
        while (true) {
            h.takeCancels();
            while (h.admitOne()) {}
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
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    pub fn failAll(h: *LaneHost, message: []const u8) void {
        std.log.err("lane host: a round failed ({s}): every admitted stream fails", .{message});
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

    pub fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        h.unlock();
        h.takeCancels();
    }
};

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_rounds_test.zig");
}
