//! Flash Next behind the native server: one reply at a time on the replay engine (prompt chunks, then GPU-side
//! rounds), greedy only; the engine's thread owns the GPU, requests wait in arrival order (foreground first).
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const Allocator = std.mem.Allocator;

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    emitted: std.ArrayList(u32) = .empty,
    began: i96 = 0,
    prefill_sent: bool = false,
    prefilled: ?i96 = null,
};

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    eng: *fx.Engine,
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
    live_generated: u64 = 0,

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
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn } };
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
        job.* = .{ .id = id, .request = request, .sink = sink };
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
        if (h.running) |job| if (stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(job.request.prompt.len + job.emitted.items.len);
            generation_tokens = h.live_generated;
            n = 1;
        };
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
            h.cancels.clearRetainingCapacity(); // a cancel for an id no longer queued or running
            h.unlock();
            h.serve(job);
            h.lock();
            h.running = null;
            h.live_generated = 0;
            h.unlock();
        }
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
            if (done > c.job.began) h.prefill_rate = @as(f64, @floatFromInt(c.job.request.prompt.len)) / (@as(f64, @floatFromInt(done - c.job.began)) / 1e9);
            h.prefill_at = done;
            c.job.prefilled = done;
            h.unlock();
            c.job.prefill_sent = true;
            emit(c.job, .{ .prefilled = 0 });
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

    fn serve(h: *Host, job: *Job) void {
        const r = job.request;
        job.began = h.now();
        if (r.sampling) |s| if (s.temperature > 0) {
            emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, "the native Flash Next engine decodes greedily only: send temperature 0");
        };
        if (h.eng.followsPeer()) {
            emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, "speed-up mode: this Mac runs rank 0's requests; send requests to rank 0");
        }
        var c: Ctx = .{ .h = h, .job = job };
        const out: fx.Out = .{ .ctx = &c, .prefilled = Ctx.prefilled, .tokens = Ctx.tokens, .cancelled = Ctx.cancelled };
        const depth: ?usize = if (r.drafts) null else 0;
        const res = h.eng.generate(r.prompt, r.max_tokens, r.eos, depth, out) catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, @errorName(e));
        };
        if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
        const reason: api.Reason = switch (res.reason) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        };
        h.finish(job, reason, .{ .rounds = res.rounds, .drafted = res.drafted, .accepted = res.accepted, .min_rows = res.min_rows, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null }, "");
    }
};

/// The engine for a Flash Next checkpoint: the replay engine on the kernels and packs in `dump`, warmed, served; `speed_up` names this Mac's speed-up mode settings (tp.zig).
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, dump: []const u8, window: i64, speed_up: ?[]const u8) !*Host {
    const eng = try fx.Engine.loadWith(gpa, dir, dump, speed_up);
    errdefer eng.deinit();
    try eng.warm();
    const follower: ?std.Thread = if (eng.followsPeer()) try std.Thread.spawn(.{}, follow, .{eng}) else null; // rank 1
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    const limit: i64 = tf.flashnext_replay.CAP - fx.MARGIN;
    h.* = .{ .gpa = gpa, .io = io, .eng = eng, .follower = follower, .info_ = .{ .name = "flashnext-zig", .lanes = 1, .context_window = @intCast(if (window > 0) @min(window, limit) else limit) } };
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
    h.eng.deinit();
    h.gpa.destroy(h);
}
