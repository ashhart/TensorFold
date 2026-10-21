//! Native speech over shared frame rounds and one Drama family, on CUDA or Metal.
const std = @import("std");
const api = @import("engine_api");
const d3 = @import("drama3");
const completion = d3.completion;
const decode = d3.speech_decode;
const rounds = @import("core").frame_rounds;
const is_cuda = !@hasDecl(d3, "gpu");
const cuda = if (is_cuda) @import("cuda") else struct {};
const Limit = if (is_cuda) @import("drama_limit.zig").Limit else struct {
    fn init(_: *@This(), _: std.mem.Allocator, _: std.Io, _: u32) !void {}
    fn deinit(_: *@This()) void {}
};
const Warm = if (is_cuda) struct {} else d3.residency.Warm;
const Session = struct { e: *d3.Engine, t: *d3.tts.Tts };
const Job = struct {
    id: api.Id,
    request: *const api.SpeechRequest,
    sink: api.Sink,
    state: d3.kv_state.State = .{},
    samples: usize = 0,
    guard: api.speech_guard.Guard = .{},
    frame_length: usize = 0,

    fn incremental(j: *const Job) bool {
        return j.request.stream or j.guard.limit > 0;
    }
};
pub const Options = struct {
    context: u32 = 8192,
    device: u32 = 0,
    lanes: u32 = 2,
    silence: api.speech_guard.Config = .{},
    max_frames: ?u32 = null,
};

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    limit: Limit,
    warm: Warm = .{},
    queue: rounds.Queue,
    sessions: [rounds.capacity]Session = undefined,
    jobs: [rounds.capacity]?Job = @splat(null),
    info_: api.Info,
    silence: api.speech_guard.Config,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    worker: ?std.Thread = null,
    closing: bool = false,
    failed: completion.State = .{},
    const vtable: api.Engine.VTable = .{
        .info = info,
        .submit = submitText,
        .submit_speech = submit,
        .cancel = cancel,
        .status = status,
        .memory = memory,
        .keepalive = if (is_cuda) null else keepalive,
        .drain = drain,
    };
    fn self(p: *anyopaque) *Host {
        return @ptrCast(@alignCast(p));
    }
    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }
    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }
    fn info(p: *anyopaque) api.Info {
        return self(p).info_;
    }
    fn submitText(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn valid(h: *Host, r: *const api.SpeechRequest) bool {
        const sp = h.info_.speech.?;
        if (r.frames.len > sp.max_prompt_tokens or r.max_frames > sp.max_frames) return false;
        if (r.pcm.len < @as(usize, r.max_frames) * sp.frame_length) return false;
        if (!std.math.isFinite(r.temperature) or r.temperature < 0) return false;
        if (!std.math.isFinite(r.top_p) or r.top_p <= 0 or r.top_p > 1) return false;
        if (r.top_k == 0 or r.top_k > (if (is_cuda) @as(u32, std.math.maxInt(i32)) else 64)) return false;
        d3.model.validate(r.frames, h.sessions[0].e.c.ids, h.info_.context_window, r.max_frames) catch return false;
        return true;
    }
    fn submit(p: *anyopaque, id: api.Id, r: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const h = self(p);
        if (!h.valid(r)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing or h.failed.closed) return error.Closed;
        for (&h.jobs) |*job| if (job.*) |*j| if (j.id == id) return error.InvalidRequest;
        const lane = try h.queue.admit();
        r.result.* = .{};
        h.jobs[lane] = .{
            .id = id,
            .request = r,
            .sink = sink,
            .guard = .init(h.silence, h.info_.speech.?.sample_rate),
            .frame_length = h.info_.speech.?.frame_length,
        };
        h.wake.signal(h.io);
    }
    fn cancel(p: *anyopaque, id: api.Id) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        for (&h.jobs, 0..) |*job, i| if (job.*) |*j| if (j.id == id) {
            h.queue.cancelled[i] = true;
        };
        h.wake.signal(h.io);
    }
    fn status(p: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        out.* = .{};
        for (h.jobs) |j| if (j != null) {
            out.running += 1;
        };
    }
    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        if (!is_cuda) return null;
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .pinned = u.host, .peak = u.peak };
    }
    fn keepalive(p: *anyopaque) ?api.keepalive.Target {
        const h = self(p);
        return .{ .ctx = &h.warm.target, .tick = @import("metal").keepalive.Target.tick };
    }
    fn sync(s: Session) !void {
        if (is_cuda) try s.e.stream.synchronize() else try s.e.flush();
    }
    fn run(h: *Host) void {
        if (is_cuda) h.sessions[0].e.ctx.makeCurrent() catch {
            h.lock();
            h.closing = true;
            h.unlock();
            while (true) {
                h.lock();
                const w = h.queue.next();
                h.unlock();
                if (w) |work| h.finish(work.lane, .failed, "CUDA context could not become current") else break;
            }
            return;
        };
        while (true) {
            h.lock();
            const work = h.queue.next() orelse {
                if (h.closing or h.failed.closed) {
                    h.unlock();
                    return;
                }
                h.wake.waitUncancelable(h.io, &h.mutex);
                h.unlock();
                continue;
            };
            const cancelled = work.cancelled or h.closing or h.failed.closed;
            h.unlock();
            if (cancelled) {
                h.finish(work.lane, if (h.failed.closed) .failed else .cancelled, if (h.failed.closed) "speech engine closed after execution failure" else "");
                continue;
            }
            h.advance(work) catch |err| {
                h.lock();
                h.failed.closed = true;
                h.unlock();
                h.finish(work.lane, .failed, @errorName(err));
            };
        }
    }
    fn advance(h: *Host, work: rounds.Work) !void {
        const i = work.lane;
        const s = h.sessions[i];
        const j = &h.jobs[i].?;
        const r = j.request;
        if (work.stage == .prompt) {
            if (!j.state.started) {
                try s.e.startPrompt(r.frames);
                if (is_cuda) try s.t.c.reset(s.t.cs) else try s.t.c.reset();
                j.state.started = true;
            }
            try s.e.promptChunk(r.frames.len, j.state.prompt, 1, null);
            j.state.prompt += 1;
            if (j.state.prompt == r.frames.len) {
                try s.e.finishPrompt(r.frames);
                h.lock();
                h.queue.stages[i] = .frame;
                h.unlock();
                j.sink.event(j.sink.ctx, j.id, &.{ .prefilled = 0 });
            }
            try sync(s);
            return;
        }
        const smp: d3.Sampling = .{ .temperature = r.temperature, .top_k = @intCast(r.top_k), .top_p = r.top_p, .seed = r.seed };
        try s.e.frameStep(smp, true, null);
        var frame: d3.model.Frame = undefined;
        try s.e.download(s.e.frames_out + j.state.frames * 11 * 4, std.mem.asBytes(&frame));
        if (frame[0] == s.e.c.ids.eos) {
            try decode.flush(s.t, j.incremental(), j.state.frames, j.state.frames, true, @as(d3.tts.Sink, .{ .ctx = j, .audio = copyAudio }));
            return h.finish(i, if (j.guard.tripped) .silence else .stop, "");
        }
        const from = j.state.frames;
        j.state.frames += 1;
        try decode.flush(s.t, j.incremental(), from, j.state.frames, j.state.frames == r.max_frames, @as(d3.tts.Sink, .{ .ctx = j, .audio = copyAudio }));
        if (r.stream) j.sink.event(j.sink.ctx, j.id, &.{ .audio = j.samples });
        if (j.guard.tripped) return h.finish(i, .silence, "");
        if (j.state.frames == r.max_frames) h.finish(i, .length, "");
    }
    fn copyAudio(p: *anyopaque, samples: []const f32) !void {
        const j: *Job = @ptrCast(@alignCast(p));
        if (samples.len > j.request.pcm.len - j.samples) return error.AudioOverflow;
        for (samples) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
        const kept = j.guard.hear(samples, j.frame_length);
        @memcpy(j.request.pcm[j.samples..][0..kept], samples[0..kept]);
        j.samples += kept;
    }
    fn finish(h: *Host, lane: usize, reason: api.Reason, message: []const u8) void {
        var state: completion.State = .{};
        const completed = state.settle(&.{.{ .ctx = &h.sessions[lane], .call = waitSession }});
        const final_reason: api.Reason = if (completed) reason else .failed;
        const final_message = if (completed) message else "speech drain failed: engine closed; storage retained";
        h.lock();
        if (!completed) h.failed.closed = true;
        const j = h.jobs[lane].?;
        if (final_reason == .stop or final_reason == .length or final_reason == .silence)
            j.request.result.samples = j.samples;
        j.request.result.reason = final_reason;
        h.jobs[lane] = null;
        h.queue.release(lane);
        h.unlock();
        j.sink.event(j.sink.ctx, j.id, &.{ .finished = .{ .reason = final_reason, .message = final_message } });
    }
    fn waitSession(ctx: *anyopaque) !void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        if (is_cuda) return s.t.complete();
        // Metal's command failure is reported after its wait completed, including the queued successor.
        s.e.flush() catch |err| if (err != error.CommandBufferFailed) return err;
    }
    fn drain(p: *anyopaque) void {
        const h = self(p);
        h.lock();
        h.closing = true;
        h.wake.signal(h.io);
        h.unlock();
        if (h.worker) |t| t.join();
        h.worker = null;
    }
    fn close(p: *anyopaque) void {
        const h = self(p);
        drain(h);
        var safe = true;
        for (h.sessions[0..h.queue.lanes]) |*s| waitSession(s) catch {
            safe = false;
        };
        if (!safe) {
            std.log.warn("speech drain failed: retaining every session and shared weight owner", .{});
            return;
        }
        if (!is_cuda) h.warm.drop(h.sessions[0].e.g);
        var i = h.queue.lanes;
        while (i > 0) {
            i -= 1;
            h.sessions[i].t.destroy();
            h.sessions[i].e.destroy();
        }
        h.limit.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, _: []const u8, o: Options) !api.Opened {
    if (o.context < 1025 or o.context > 8192) return error.BadDramaContext;
const budget = try @import("drama_budget.zig").budget(o.context, o.max_frames);
    if (!is_cuda) try d3.load_limit.admit(gpa, io, dir, o.context); // refused before a weight is read: the floor's one-Mac limit
    try o.silence.validate();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa,
        .io = io,
        .limit = undefined,
        .queue = try rounds.Queue.init(o.lanes),
        .info_ = .{},
        .silence = o.silence,
    };
    try h.limit.init(gpa, io, o.device);
    errdefer h.limit.deinit();
    var made: usize = 0;
    errdefer while (made > 0) {
        made -= 1;
        h.sessions[made].t.destroy();
        h.sessions[made].e.destroy();
    };
    const codec = try std.fs.path.join(gpa, &.{ dir, "codec.pth" });
    defer gpa.free(codec);
    while (made < o.lanes) : (made += 1) {
        const e = if (made == 0) try d3.Engine.create(gpa, io, dir, .{
            .max_ctx = o.context,
            .max_rows = 1,
            .attn_chunk = 1,
            .max_frames = o.max_frames orelse 1024,
            .warmup = false,
            .device = o.device,
        }) else try h.sessions[0].e.fork();
        errdefer e.destroy();
        const t = if (made == 0) try d3.tts.Tts.create(gpa, io, e, codec, 1) else try h.sessions[0].t.fork(e);
        h.sessions[made] = .{ .e = e, .t = t };
    }
    const frame_cap = budget.frames;
    h.info_ = .{
        .name = "Drama 3",
        .text_generation = false,
        .lanes = o.lanes,
        .context_window = o.context,
        .plain_only = true,
        .prompt_cache = false,
        .speech = .{ .sample_rate = 44100, .frame_length = 2048, .max_frames = frame_cap, .max_prompt_tokens = budget.prompt },
    };
    if (!is_cuda) try h.warm.hold(h.sessions[0].e.g, io);
    errdefer if (!is_cuda) h.warm.drop(h.sessions[0].e.g);
    h.worker = try std.Thread.spawn(.{}, Host.run, .{h});
    if (o.silence.enabled())
        std.log.info("drama3: silence guard on: a reply ends after {d} s below {d} dBFS", .{ o.silence.seconds.?, o.silence.floor_dbfs.? });
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}

test "native silence guard: whole and stream share decode cuts; off preserves PCM" {
    const Fake = struct {
        calls: usize = 0,
        const pcm = [_]f32{ 0.25, -0.25, 0, 0, 0, 0, 0.25, -0.25 };
        pub fn flush(t: *@This(), from: usize, to: usize, sink: d3.tts.Sink) !void {
            t.calls += 1;
            try sink.audio(sink.ctx, pcm[from * 2 .. to * 2]);
        }
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
    };
    for ([_]bool{ false, true }) |enabled| {
        for ([_]bool{ false, true }) |stream| {
            var pcm: [Fake.pcm.len]f32 = undefined;
            var result: api.SpeechResult = .{};
            const r: api.SpeechRequest = .{
                .prompt = &.{},
                .pcm = &pcm,
                .result = &result,
                .seed = 0,
                .stream = stream,
            };
            var t: Fake = .{};
            var j: Job = .{
                .id = 0,
                .request = &r,
                .sink = .{ .ctx = &t, .event = Fake.event },
                .guard = .init(if (enabled) .{ .floor_dbfs = -20, .seconds = 0.5 } else .{}, 8),
                .frame_length = 2,
            };
            const sink: d3.tts.Sink = .{ .ctx = &j, .audio = Host.copyAudio };
            for (0..Fake.pcm.len / 2) |i| {
                try decode.flush(&t, j.incremental(), i, i + 1, false, sink);
                if (j.guard.tripped) break;
            }
            if (!j.guard.tripped)
                try decode.flush(&t, j.incremental(), Fake.pcm.len / 2, Fake.pcm.len / 2, true, sink);
            const kept: usize = if (enabled) 6 else Fake.pcm.len;
            try std.testing.expectEqual(kept, j.samples);
            try std.testing.expectEqual(enabled, j.guard.tripped);
            try std.testing.expectEqualSlices(f32, Fake.pcm[0..kept], pcm[0..j.samples]);
            if (!enabled and !stream) try std.testing.expectEqual(@as(usize, 1), t.calls);
        }
    }
}
