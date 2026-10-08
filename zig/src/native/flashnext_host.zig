//! Flash Next behind the native server: one reply at a time on the replay engine (prompt chunks, then GPU-side
//! rounds), greedy only; the engine's thread owns the GPU, requests wait in arrival order (foreground first).
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const snap = tf.flashnext_snapshot;
const learned = tf.flashnext_learned;
const pc = api.prompt_cache;
const cache_fit = @import("cache_fit.zig");
const Allocator = std.mem.Allocator;

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    emitted: std.ArrayList(u32) = .empty,
    queued: i96 = 0, // submitted: the wait before `began` is the engine finishing what it was doing
    began: i96 = 0,
    prefill_sent: bool = false,
    prefilled: ?i96 = null,
    cached: u32 = 0, // prompt tokens restored from the prompt cache
    restore_ns: i96 = 0, // the cache's lookup and restore before the prompt pass
    keep_ns: i96 = 0, // the cache's saves at the pass's marks
};

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    eng: *fx.Engine,
    warm: mtl.keepalive.Target = undefined, // the engine's queue, for the server's idle ticker
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    follower: ?std.Thread = null, // speed-up mode's rank 1: the thread running rank 0's requests
    running: ?*Job = null,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens each round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_prompt: u64 = 0, // the running reply's prompt tokens, for status (which never reads the job: finish frees it)
    live_generated: u64 = 0,
    cache: ?pc.Store = null, // kept prompt states (engine thread only); null: no reuse (rank 1, a zero budget)
    prompt: []const u32 = &.{}, // the request in the engine (speed-up mode names kept states by its tokens)
    saved: std.ArrayList(u32) = .empty, // this request's marks the cache kept (speed-up: rank 1 drops the others)
    learned: ?api.prompt_imprint.Imprint = null, // --learn: shared states on disk, read back by later servers

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    pub fn start(h: *Host) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what waits, lets the running reply end at its next round and joins the thread.
    pub fn stop(h: *Host) void {
        h.lock();
        h.closing = true;
        h.wake.broadcast(h.io);
        h.unlock();
        if (h.thread) |t| t.join();
        h.thread = null;
        h.queued.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
    }

    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn } };
    }

    /// The engine's queue as a keepalive target: the ticker commits a tiny buffer on it while idle.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        const h: *Host = @ptrCast(@alignCast(ctx));
        return .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    }

    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }

    fn now(h: *Host) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .id = id, .request = request, .sink = sink, .queued = h.now() };
        h.lock();
        defer h.unlock();
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        var at = h.queued.items.len; // foreground before background, each in arrival order
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            _ = h.queued.orderedRemove(i);
            h.unlock();
            h.finish(job, .cancelled, .{}, "");
            h.lock();
            return;
        };
        h.cancels.append(h.gpa, id) catch {};
    }

    fn statusFn(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const t = h.now();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= t - window_ns) tokens += m.tokens;
        }
        var n: usize = 0;
        var generation_tokens: u64 = 0;
        if (h.running != null and stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(h.live_prompt + h.live_generated);
            generation_tokens = h.live_generated;
            n = 1;
        }
        out.* = .{
            .running = @intFromBool(h.running != null),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = generation_tokens,
        };
    }

    fn memoryFn(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn emit(job: *Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    fn finish(h: *Host, job: *Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        if (h.running == job) h.running = null; // before finished is out: status must not see a job about to be freed
        h.unlock();
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        job.emitted.deinit(h.gpa);
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *Host, n: usize) void {
        const t = h.now();
        h.lock();
        defer h.unlock();
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < t - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = t, .tokens = n }) catch {};
    }

    fn run(h: *Host) void {
        while (true) {
            h.lock();
            while (h.queued.items.len == 0 and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            }
            if (h.closing) {
                const left = h.gpa.dupe(*Job, h.queued.items) catch &.{};
                h.queued.clearRetainingCapacity();
                h.unlock();
                for (left) |job| h.finish(job, .cancelled, .{}, "");
                h.gpa.free(left);
                return;
            }
            const job = h.queued.orderedRemove(0);
            h.running = job;
            h.live_prompt = job.request.prompt.len;
            h.live_generated = 0;
            h.cancels.clearRetainingCapacity(); // a cancel for an id no longer queued or running
            h.unlock();
            h.serve(job);
            h.lock();
            h.running = null;
            h.live_generated = 0;
            const idle = h.queued.items.len == 0;
            h.unlock();
            if (idle and h.cache != null) { // the next turn's save finds its pages touched, unless a request comes first
                const t0 = h.now();
                const got = h.eng.snap_pool.ready(h.eng.r.device, fx.NEXT_TURN, .{ .ctx = h, .check = waiting }, h.cache.?.room());
                if (got.cap > 0) std.log.info("prompt cache: readied {d} of {d} MiB for the next save in {d:.1} ms", .{ got.touched >> 20, got.cap >> 20, ms(h.now() - t0) });
            }
        }
    }

    fn waiting(ctx: *anyopaque) bool {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        return h.queued.items.len > 0 or h.closing;
    }

    /// The reply's callbacks from the engine's rounds.
    const Ctx = struct {
        h: *Host,
        job: *Job,

        fn prefilled(ctx: *anyopaque) void {
            const c: *Ctx = @ptrCast(@alignCast(ctx));
            const h = c.h;
            const done = h.now();
            h.lock();
            if (done > c.job.began) h.prefill_rate = @as(f64, @floatFromInt(c.job.request.prompt.len - c.job.cached)) / (@as(f64, @floatFromInt(done - c.job.began)) / 1e9);
            h.prefill_at = done;
            c.job.prefilled = done;
            h.unlock();
            c.job.prefill_sent = true;
            emit(c.job, .{ .prefilled = c.job.cached });
        }

        fn marked(ctx: *anyopaque, at: usize) void {
            const c: *Ctx = @ptrCast(@alignCast(ctx));
            const t0 = c.h.now();
            if (c.h.cache) |*store| if (store.keep(c.job.request.prompt, @intCast(at), null, c.job.request.chunks, &.{})) c.h.saved.append(c.h.gpa, @intCast(at)) catch {}; // held here: rank 1 keeps its copy
            c.job.keep_ns += c.h.now() - t0;
        }

        fn tokens(ctx: *anyopaque, toks: []const u32) bool {
            const c: *Ctx = @ptrCast(@alignCast(ctx));
            const job = c.job;
            var matched = false;
            var n: usize = 0;
            for (toks) |t| { // stop strings are checked after each token, as the lane core does
                job.emitted.append(c.h.gpa, t) catch return true;
                n += 1;
                if (job.request.stop) |s| if (s.check(s.ctx, job.emitted.items)) {
                    matched = true;
                    break;
                };
            }
            emit(job, .{ .tokens = toks[0..n] });
            c.h.lock();
            c.h.live_generated = @intCast(job.emitted.items.len);
            c.h.unlock();
            c.h.noteDecoded(n);
            return matched;
        }

        fn cancelled(ctx: *anyopaque) bool {
            const c: *Ctx = @ptrCast(@alignCast(ctx));
            const h = c.h;
            h.lock();
            defer h.unlock();
            if (h.closing) return true;
            return std.mem.indexOfScalar(api.Id, h.cancels.items, c.job.id) != null;
        }
    };

    /// What a reply finishes with: h.finish frees the job and lets the server free its request, so serve calls it last.
    const Fin = struct { reason: api.Reason = .failed, stats: api.Stats = .{}, message: []const u8 = "" };

    fn serve(h: *Host, job: *Job) void {
        var fin: Fin = .{};
        defer h.finish(job, fin.reason, fin.stats, fin.message); // after every defer below: they read the job and its request
        const r = job.request;
        job.began = h.now();
        if (r.sampling) |s| if (s.temperature > 0) {
            emit(job, .{ .prefilled = 0 });
            fin.message = "the native Flash Next engine decodes greedily only: send temperature 0";
            return;
        };
        if (h.eng.followsPeer()) {
            emit(job, .{ .prefilled = 0 });
            fin.message = "speed-up mode: this Mac runs rank 0's requests; send requests to rank 0";
            return;
        }
        var c: Ctx = .{ .h = h, .job = job };
        const out: fx.Out = .{ .ctx = &c, .prefilled = Ctx.prefilled, .tokens = Ctx.tokens, .cancelled = Ctx.cancelled, .marked = Ctx.marked };
        const depth: ?usize = if (r.drafts) null else 0;
        var arena: std.heap.ArenaAllocator = .init(h.gpa);
        defer arena.deinit();
        var plan: pc.Plan = .{};
        const kept0 = if (h.cache) |*store| store.counts.kept else 0;
        const t_begin = h.now();
        h.prompt = r.prompt; // a learned state read back in `begin` is named by these tokens
        if (h.cache) |*store| if (r.prompt.len + r.max_tokens + fx.MARGIN <= tf.flashnext_replay.CAP) {
            plan = store.beginRewind(arena.allocator(), r.prompt, r.history_len, r.rewind_len, r.shared_prefixes, &.{}, null, &.{}) catch .{};
        };
        job.restore_ns = h.now() - t_begin;
        job.cached = plan.from;
        defer if (job.prefilled) |done| std.log.info("prompt pass: {d} -> {d} tokens in {d:.1} ms (waited {d:.1} ms; lookup and restore {d:.1} ms, peer handoff {d:.1} ms, keeps {d:.1} ms)", .{ job.cached, r.prompt.len, ms(done - job.began), ms(job.began - job.queued), ms(job.restore_ns), ms(h.eng.handoff_ns), ms(job.keep_ns) });
        h.saved.clearRetainingCapacity();
        defer if (h.cache) |*store| store.report(r.prompt.len, job.cached, store.counts.kept - kept0);
        defer if (h.eng.r.tp != null) for (plan.marks) |mk| if (std.mem.indexOfScalar(u32, h.saved.items, mk) == null) {
            h.eng.peer_drops.append(h.gpa, fx.keyOf(r.prompt[0 .. mk + 1])) catch {}; // rank 1 kept it, this Mac didn't
        };
        const res = h.eng.generateFrom(r.prompt, plan.from, plan.marks, r.max_tokens, r.eos, depth, out) catch |e| retry: {
            if (e == error.PeerNotResumed) { // rank 1 lacks this state: both Macs read the prompt from the start
                std.log.warn("speed-up mode: rank 1 could not resume at {d}; reading the prompt from the start", .{plan.from});
                if (h.cache) |*store| store.forget(r.prompt, plan.from, &.{}); // so later prompts do not ask rank 1 for it again
                job.cached = 0;
                break :retry h.eng.generateFrom(r.prompt, 0, plan.marks, r.max_tokens, r.eos, depth, out) catch |e2| {
                    if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
                    fin.message = @errorName(e2);
                    return;
                };
            }
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            fin.message = @errorName(e);
            return;
        };
        if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
        fin.reason = switch (res.reason) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        };
        fin.stats = .{ .rounds = res.rounds, .drafted = res.drafted, .accepted = res.accepted, .min_rows = res.min_rows, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null };
    }
};

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// The prompt cache's copies of the engine's state (snapshot.zig), on the engine's thread.
const Snaps = struct {
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.eng.snap_pool.size(snap.bytes(at)); // a new buffer's size; a free one in the pool may take the state instead
    }
    fn charged(_: *anyopaque, saved: pc.Saved) u64 {
        const k: *Kept = @ptrCast(@alignCast(saved));
        return k.st.cap;
    }
    fn spare(ptr: *anyopaque) u64 {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.eng.snap_pool.spare();
    }
    fn trim(ptr: *anyopaque, room: u64) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.eng.snap_pool.trim(room);
    }
    fn reuses(ptr: *anyopaque, at: u32) bool {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.eng.snap_pool.fits(snap.bytes(at));
    }
    /// A kept state and its name on both Macs (speed-up mode's rank 1 keeps its own under the same name).
    const Kept = struct { st: *snap.State, key: u64 };
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const k = try h.gpa.create(Kept);
        errdefer h.gpa.destroy(k);
        k.* = .{ .st = try snap.save(h.eng, h.gpa, at), .key = fx.keyOf(h.prompt[0 .. at + 1]) };
        if (std.mem.indexOfScalar(u64, h.eng.peer_drops.items, k.key)) |i| _ = h.eng.peer_drops.swapRemove(i); // rank 1's copy is this new one
        return k;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const k: *Kept = @ptrCast(@alignCast(saved));
        try snap.restore(h.eng, k.st);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const k: *Kept = @ptrCast(@alignCast(saved));
        if (h.eng.r.tp != null) h.eng.peer_drops.append(h.gpa, k.key) catch {};
        snap.drop(h.gpa, k.st);
        h.gpa.destroy(k);
    }
    /// --learn: a kept state to its file under `dir`.
    fn write(_: *anyopaque, saved: pc.Saved, dir: [:0]const u8, key: u64) anyerror!void {
        const k: *Kept = @ptrCast(@alignCast(saved));
        var buf: [1100]u8 = undefined;
        try learned.writeFile(k.st, try learned.path(&buf, dir, key));
    }
    /// --learn: learned state `key` read back into a pool buffer, named by the request's tokens as a saved one is.
    fn read(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!pc.Saved {
        const h: *Host = @ptrCast(@alignCast(ptr));
        if (at >= h.prompt.len) return error.SnapshotRead;
        var buf: [1100]u8 = undefined;
        const k = try h.gpa.create(Kept);
        errdefer h.gpa.destroy(k);
        k.* = .{ .st = try learned.readFile(h.eng, h.gpa, at, try learned.path(&buf, dir, key)), .key = fx.keyOf(h.prompt[0 .. at + 1]) };
        return k;
    }
    /// --learn: learned state `key`'s file removed.
    fn forget(_: *anyopaque, dir: [:0]const u8, key: u64) void {
        var buf: [1100]u8 = undefined;
        _ = std.c.unlink(learned.path(&buf, dir, key) catch return);
    }
};

/// The prompt cache's plan (cache_fit.zig); without an OS reading, `gib` or the Metal working set's spare.
fn cacheBudget(eng: *fx.Engine, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !api.PromptCachePlan {
    const rank = if (eng.followsPeer()) " (rank 1 keeps its halves of rank 0's)" else if (eng.r.tp != null) " (rank 1 mirrors them)" else "";
    const dev = eng.r.device;
    return cache_fit.budget(gib, over, @min(dev.maxWorkingSet() -| dev.allocated() -| (8 << 30), 16 << 30), a, why, rank);
}

/// Learned states under `root` by checkpoint, kernel sources and modes, probe bits, OS build and chip.
fn learnedStates(gpa: Allocator, eng: *fx.Engine, root: []const u8, cap: u64) !api.prompt_imprint.Imprint {
    var h = std.hash.Wyhash.init(0x6678);
    const probe = try learned.probe(eng, gpa);
    const modes = [_]u64{ @intFromBool(eng.r.xnew), @intFromBool(eng.r.hc_up), @intFromBool(eng.r.device.tensorUnits()) };
    for ([_]u64{ eng.identity_hash, learned.sourceHash(), modes[0], modes[1], modes[2], probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(eng.r.device.name()));
    const m = try api.prompt_imprint.Imprint.open(gpa, root, h.final(), cap);
    std.log.info("flash next: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ m.dir, m.metas.items.len, m.total() >> 20, cap >> 20 });
    return m;
}

/// The engine for a Flash Next checkpoint: the replay engine on the kernels and packs in `dump`, warmed, served; `speed_up` names this Mac's speed-up mode settings (tp.zig); `cache_gib` the prompt cache's budget (null: what 70% of RAM leaves; `over_cap` lets a larger one through); on error.CacheOverCap `why` (in `a`) says why.
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, dump: ?[]const u8, window: i64, speed_up: ?[]const u8, cache_gib: ?f64, over_cap: bool, learn: ?[]const u8, learn_cap: u64, a: Allocator, why: *[]const u8) !*Host {
    const eng = try fx.Engine.loadWith(gpa, io, dir, dump, speed_up);
    errdefer eng.deinit();
    if (learn != null and eng.r.tp != null) {
        why.* = "--learn is not kept across a Flash Next pair yet: leave it out, or serve on one Mac";
        return error.LearnOnPair;
    }
    eng.warm() catch |err| {
        std.log.err("flash next: warm-up failed: {s}", .{@errorName(err)});
        return err;
    };
    const cache_plan = try cacheBudget(eng, cache_gib, over_cap, a, why);
    const budget = cache_plan.budget_bytes;
    eng.peer_budget = budget; // rank 1: its kept states and free buffers stay inside the same budget
    eng.snap_pool.max = budget;
    const follower: ?std.Thread = if (eng.followsPeer()) try std.Thread.spawn(.{}, follow, .{eng}) else null; // rank 1
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    const limit: i64 = tf.flashnext_replay.CAP - fx.MARGIN;
    h.* = .{ .gpa = gpa, .io = io, .eng = eng, .follower = follower, .info_ = .{ .name = "flashnext-zig", .prompt_cache_plan = cache_plan, .lanes = 1, .greedy_only = true, .context_window = @intCast(if (window > 0) @min(window, limit) else limit) } };
    h.warm = .{ .queue = eng.r.queue };
    if (!eng.followsPeer() and budget > 0) h.cache = pc.Store.init(gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop, .charged = Snaps.charged, .spare = Snaps.spare, .trim = Snaps.trim, .reuses = Snaps.reuses, .write = Snaps.write, .read = Snaps.read, .forget = Snaps.forget } }, .{ .lookahead = 1 }, budget);
    h.info_.prompt_cache = h.cache != null;
    if (learn) |root| {
        h.learned = try learnedStates(gpa, eng, root, learn_cap);
        if (h.cache) |*store| store.imprint = &h.learned.?;
    }
    errdefer if (h.learned) |*m| m.deinit();
    try h.start();
    return h;
}

fn follow(eng: *fx.Engine) void {
    eng.follow() catch |err| std.log.err("speed-up mode: following rank 0 ended: {s}", .{@errorName(err)});
}

pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.stop();
    if (h.follower) |th| { // rank 1: its wait for rank 0's next request ends, then the thread
        h.eng.stopFollowing();
        th.join();
    }
    if (h.cache) |*store| store.deinit();
    if (h.learned) |*m| m.deinit();
    h.saved.deinit(h.gpa);
    h.eng.deinit();
    h.gpa.destroy(h);
}

test "status after a reply finishes reads no freed job" {
    const gpa = std.testing.allocator;
    var h: Host = .{ .gpa = gpa, .io = std.testing.io, .eng = undefined, .info_ = undefined };
    defer h.decoded.deinit(gpa);
    const Reader = struct { // the server's status poll, here inside the finished event and after it
        h: *Host,
        running: u32 = 9,
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            var st: api.Status = .{};
            var toks: [2]u32 = .{ 0, 0 };
            if (e.* == .finished) {
                Host.statusFn(r.h, &st, &toks);
                r.running = st.running;
            }
        }
    };
    var reader: Reader = .{ .h = &h };
    const prompt = [_]u32{ 1, 2, 3 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    const job = try gpa.create(Job);
    job.* = .{ .id = 1, .request = &request, .sink = .{ .ctx = &reader, .event = Reader.event } };
    try job.emitted.append(gpa, 7);
    h.lock();
    h.running = job; // as run() holds it while serve runs
    h.live_prompt = prompt.len;
    h.live_generated = 1;
    h.unlock();
    var st: api.Status = .{};
    var toks: [2]u32 = .{ 0, 0 };
    Host.statusFn(&h, &st, &toks);
    try std.testing.expectEqual(@as(u32, 1), st.running);
    try std.testing.expectEqual(@as(u32, 4), toks[0]);
    h.finish(job, .stop, .{}, ""); // serve's completion: the job is freed, serve has not returned
    try std.testing.expectEqual(@as(u32, 0), reader.running);
    Host.statusFn(&h, &st, &toks); // a poll before run() clears its own pointer
    try std.testing.expectEqual(@as(u32, 0), st.running);
    try std.testing.expectEqual(@as(usize, 0), st.streams);
}
