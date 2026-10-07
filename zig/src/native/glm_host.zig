//! GLM-5.3-Flash behind the native server: one greedy reply at a time, in arrival order; speed-up mode: rank 0 serves, rank 1 follows.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const ge = tf.glm.engine;
const Allocator = std.mem.Allocator;

/// Draft depth of a drafted request (the MTP head's chain a round): the pair's best on prose and code.
const DEPTH = 2;
/// Copy drafts by default (GLM_COPY overrides): a round copies what followed the reply's last 3+ tokens earlier.
const COPY_MIN = 3;

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
    eng: *ge.Engine,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    running: ?*Job = null,
    closing: bool = false,
    thread: ?std.Thread = null,
    follower: ?std.Thread = null, // speed-up mode's rank 1: the thread running rank 0's requests
    decoded: std.ArrayList(Mark) = .empty,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_prompt: u64 = 0, // the running reply's prompt tokens, for status (which never reads the job: finish frees it)
    live_generated: u64 = 0,

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    pub fn start(h: *Host) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

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
        return .{ .ctx = &h.eng.keepalive_target, .tick = mtl.keepalive.Target.tick };
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
            h.cancels.clearRetainingCapacity();
            h.unlock();
            h.serve(job);
            h.lock();
            h.running = null;
            h.live_generated = 0;
            h.unlock();
        }
    }

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
            for (toks) |t| { // stop strings after each token, as the lane core checks them
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
            return h.finish(job, .failed, .{}, "the native GLM-5.3-Flash engine decodes greedily only: send temperature 0");
        };
        if (h.eng.followsPeer()) {
            emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, "speed-up mode: this Mac runs rank 0's requests; send requests to rank 0");
        }
        var c: Ctx = .{ .h = h, .job = job };
        const out: ge.Out = .{ .ctx = &c, .prefilled = Ctx.prefilled, .tokens = Ctx.tokens, .cancelled = Ctx.cancelled };
        const depth: usize = if (r.drafts) DEPTH else 0;
        if (h.eng.ep) |ep| ep.ctl.sendRequest(.{ .max_tokens = r.max_tokens, .depth = depth, .eos = r.eos, .prompt = r.prompt }) catch |e| {
            emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, @errorName(e));
        };
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
        h.finish(job, reason, .{ .rounds = res.rounds, .drafted = res.drafted, .accepted = res.accepted, .min_rows = res.min_rows, .prefill_seconds = res.prompt_seconds }, "");
    }
};

/// The engine for a GLM-5.3-Flash checkpoint, `window` tokens of cache, warmed and served (speed-up settings in `speed_up`).
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: u32, speed_up: ?[]const u8) !*Host {
    const eng = try ge.Engine.loadWith(gpa, dir, window + 64, speed_up);
    errdefer eng.deinit();
    if (std.c.getenv("GLM_COPY") == null) eng.copy_min = COPY_MIN;
    var toks: [96]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
    var dummy: u8 = 0;
    _ = try eng.generate(&toks, 16, &.{}, DEPTH, .{ .ctx = &dummy, .prefilled = ge.Quiet.prefilled, .tokens = ge.Quiet.tokens, .cancelled = ge.Quiet.cancelled });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .eng = eng, .info_ = .{ .name = "glm-zig", .lanes = 1, .context_window = window } };
    try h.start(); // the host thread first: once the follower runs, nothing after it can fail
    errdefer h.stop();
    if (eng.followsPeer()) h.follower = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, follow, .{eng});
    std.log.info("GLM-5.3-Flash loaded in {d:.1} s ({d:.1} GB of weights{s}), context {d} tokens", .{ eng.load_seconds, @as(f64, @floatFromInt(eng.w.bytes)) / 1e9, if (eng.ep == null) "" else if (eng.followsPeer()) ", speed-up rank 1" else ", speed-up rank 0", window });
    return h;
}

fn follow(eng: *ge.Engine) void {
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

test "status after a reply finishes reads no freed job" {
    const gpa = std.testing.allocator;
    var h: Host = .{ .gpa = gpa, .io = std.testing.io, .eng = undefined, .info_ = undefined };
    defer h.decoded.deinit(gpa);
    const Reader = struct { // the server's status poll, inside the finished event
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
    h.finish(job, .stop, .{}, ""); // the job is freed while serve has not returned
    try std.testing.expectEqual(@as(u32, 0), reader.running);
    Host.statusFn(&h, &st, &toks);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    try std.testing.expectEqual(@as(usize, 0), st.streams);
}
