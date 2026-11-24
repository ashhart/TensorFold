//! Native speech from one ZONOS3 Mini family on CUDA: one request at a time, streamed in chunks or decoded at the end.
const std = @import("std");
const api = @import("engine_api");
const z3 = @import("zonos3");
const cuda = @import("cuda");

const cfg = z3.config;
const W = cfg.frame_width;
const C = cfg.codebooks;

pub const default_frames = 2000; // the release's frame budget, 133 s of audio

pub const Options = struct {
    context: u32 = 4096,
    device: u32 = 0,
    max_frames: ?u32 = null,
    admit: ?Admit = null, // null: nothing is asked, as the tool and tests load
    refusal: ?*Refusal = null,
};

/// Who knows the pool: the device bytes a load may still take, on the context given.
pub const Admit = struct { ptr: *anyopaque, room: *const fn (*anyopaque, *const cuda.Context) anyerror!u64 };

/// A refused load's sizes, in bytes.
pub const Refusal = struct { need: u64 = 0, room: u64 = 0 };

/// The pinned release's weights as the device holds them: the backbone's bf16 and the keep2_plain decoder's fp32.
pub const weight_bytes: u64 = 2_531_483_136;
/// The clip encoder's weights, state and workspaces: the acoustic encoder, the SSL teacher and the quantizers.
pub const clip_bytes: u64 = 5_643_510_688;

/// Device bytes a load takes at a context and frame budget: the weights, then each part's own count of its state.
pub fn need(context: u32, frames: u32) u64 {
    return weight_bytes + z3.Engine.stateBytes(.{ .max_ctx = context }) + z3.Codec.stateBytes(codecOptions(frames));
}

/// The codec at a frame budget: its usual pass, or the whole budget when that is shorter.
pub fn codecOptions(frames: u32) z3.codec.Options {
    return .{ .max_frames = frames, .pass_frames = @min(frames, (z3.codec.Options{}).pass_frames) };
}

const Door = struct {
    o: Options,
    need: u64,

    fn check(p: *anyopaque, ctx: *const cuda.Context) anyerror!void {
        const d: *Door = @ptrCast(@alignCast(p));
        const a = d.o.admit orelse return;
        const room = try a.room(a.ptr, ctx);
        if (d.need <= room) return;
        if (d.o.refusal) |r| r.* = .{ .need = d.need, .room = room };
        return error.InsufficientDeviceMemory;
    }
};

/// The frame budget and the prompt rows a context leaves.
pub fn budget(context: u32, max_frames: ?u32) !struct { frames: u32, prompt: u32 } {
    if (context < 1025 or context > cfg.max_seqlen) return error.BadZonosContext;
    const frames = max_frames orelse @min(default_frames, context / 2);
    if (frames < C + 2 or frames >= context) return error.BadZonosFrames;
    codecOptions(frames).validate() catch return error.BadZonosFrames;
    return .{ .frames = frames, .prompt = context - frames };
}

/// What a load opens with, before it touches the device: the budget, the codec's options and the bytes it admits.
pub const Plan = struct { frames: u32, prompt: u32, codec: z3.codec.Options, need: u64 };

pub fn plan(o: Options) !Plan {
    const b = try budget(o.context, o.max_frames);
    return .{ .frames = b.frames, .prompt = b.prompt, .codec = codecOptions(b.frames), .need = need(o.context, b.frames) };
}

/// What a request's prompt rows name: [reference prefix | speaker row] conditions, the text from BOS to EOS, silence.
pub const Layout = struct {
    condition_row: usize,
    prefill: usize,

    pub fn of(rows: []const [W]i32) !Layout {
        var first: ?usize = null;
        var last: ?usize = null;
        for (rows, 0..) |row, i| {
            if (row[C] == z3.prompt.bos and first == null) first = i;
            if (row[C] == z3.prompt.eos) last = i;
        }
        const b = first orelse return error.BadPromptLayout;
        const e = last orelse return error.BadPromptLayout;
        if (b < cfg.conditions or e <= b) return error.BadPromptLayout;
        return .{ .condition_row = b - cfg.conditions, .prefill = rows.len - 1 - e };
    }
};

const Job = struct { id: api.Id, request: *const api.SpeechRequest, sink: api.Sink };

/// The backbone under generation, stopped between rows when the request is cancelled or the host closes.
const Guarded = struct {
    h: *Host,

    pub fn run(g: Guarded, rows: []const [W]i32, replaced: []const z3.Replace, probe: ?z3.Probe) !void {
        g.h.lock();
        const stop = g.h.cancelled or g.h.closing;
        g.h.unlock();
        if (stop) return error.Cancelled;
        try g.h.e.run(rows, replaced, probe);
    }

    pub fn lastLogits(g: Guarded, out: *[cfg.head_rows]f32) !void {
        try g.h.e.lastLogits(out);
    }
};

/// Frames a streamed chunk waits for after `spoken` generated frames went out: 1, 2, 4, 8, then 15 (a second).
pub fn chunkFrames(spoken: usize) usize {
    return @min(15, 1 + spoken);
}

/// A request's audio as its rows arrive: streamed requests decode in growing chunks, the rest once at the end.
const Audio = struct {
    h: *Host,
    job: Job,
    prefill: []const [W]i32,
    decoded: usize = 0, // frames handed to the codec
    samples: usize = 0, // samples in the request's buffer

    pub fn row(s: *Audio, rows: []const [W]i32, eos_frame: ?usize, finished: bool) !void {
        const h = s.h;
        const r = s.job.request;
        if (!finished and !r.stream) return;
        const all = try z3.speech.frames(h.gpa, s.prefill, rows, eos_frame);
        defer h.gpa.free(all);
        const waiting = (s.prefill.len -| s.decoded) + chunkFrames(s.decoded -| s.prefill.len);
        if (!finished and all.len - s.decoded < waiting) return;
        try h.codec.decode(all[s.decoded..], null);
        s.decoded = all.len;
        const pcm = try h.codec.emit(h.gpa, finished, null);
        defer h.gpa.free(pcm);
        if (pcm.len > r.pcm.len - s.samples) return error.AudioOverflow;
        for (pcm) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
        z3.speech.output(pcm, h.rel.headroom_db);
        @memcpy(r.pcm[s.samples..][0..pcm.len], pcm);
        s.samples += pcm.len;
        if (r.stream and pcm.len > 0) s.job.sink.event(s.job.sink.ctx, s.job.id, &.{ .audio = s.samples });
    }
};

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    rel: z3.release.Release,
    e: *z3.Engine,
    codec: *z3.Codec,
    clip: ?*z3.clip.Clip = null, // the clip encoder, loaded by the first request that brings a clip
    admit: ?Admit = null,
    dir: []const u8,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    worker: ?std.Thread = null,
    job: ?Job = null,
    cancelled: bool = false,
    closing: bool = false,
    failed: bool = false,

    const vtable: api.Engine.VTable = .{
        .info = info,
        .submit = submitText,
        .submit_speech = submit,
        .cancel = cancel,
        .status = status,
        .memory = memory,
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
        if (r.frames.len == 0 or r.frames.len > sp.max_prompt_tokens or r.max_frames == 0 or r.max_frames > sp.max_frames) return false;
        if (r.pcm.len < @as(usize, r.max_frames) * sp.frame_length) return false;
        if (r.speaker.len != 0 and r.speaker.len != cfg.speaker_dim) return false;
        for (r.speaker) |x| if (!std.math.isFinite(x)) return false;
        if (r.reference_audio.len != 0) {
            // a clip becomes a reference prefix: a row a frame and nine more, ahead of the request's rows
            const clip_frames = (r.reference_audio.len + z3.codec.frame_samples - 1) / z3.codec.frame_samples;
            if (r.speaker.len != 0 or r.reference_audio.len < z3.clip.min_samples or clip_frames > z3.clip.max_frames) return false;
            if (r.frames.len + clip_frames + C - 1 > sp.max_prompt_tokens) return false;
            for (r.reference_audio) |x| if (!std.math.isFinite(x)) return false;
        }
        if (!std.math.isFinite(r.temperature) or r.temperature < 0) return false;
        if (!std.math.isFinite(r.top_p) or r.top_p <= 0 or r.top_p > 1 or r.top_k == 0) return false;
        if (r.unconditioned.len != 0 or r.cfg_gamma != 1) return false;
        for (r.frames) |row| {
            for (row[0..C]) |code| if (code < 0 or code >= cfg.audio_vocab) return false;
            if (row[C] < 0 or row[C] >= h.e.w.text_rows) return false;
        }
        _ = Layout.of(r.frames) catch return false;
        return true;
    }

    fn submit(p: *anyopaque, id: api.Id, r: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const h = self(p);
        if (!h.valid(r)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing or h.failed) return error.Closed;
        if (h.job != null) return error.Busy;
        r.result.* = .{};
        h.job = .{ .id = id, .request = r, .sink = sink };
        h.cancelled = false;
        h.wake.signal(h.io);
    }

    fn cancel(p: *anyopaque, id: api.Id) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        if (h.job) |j| if (j.id == id) {
            h.cancelled = true;
        };
    }

    fn status(p: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        out.* = .{};
        if (h.job != null) out.running = 1;
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .pinned = u.host, .peak = u.peak };
    }

    fn run(h: *Host) void {
        const current = if (h.e.ctx.makeCurrent()) |_| true else |_| false;
        while (true) {
            h.lock();
            const job = h.job orelse {
                if (h.closing) {
                    h.unlock();
                    return;
                }
                h.wake.waitUncancelable(h.io, &h.mutex);
                h.unlock();
                continue;
            };
            const dead = h.failed or !current;
            h.unlock();
            if (dead) {
                h.finish(job, .failed, 0, "speech engine closed after execution failure");
                continue;
            }
            var samples: usize = 0;
            const reason = h.serve(job, &samples) catch |err| switch (err) {
                error.Cancelled => api.Reason.cancelled,
                // refused before anything ran on the device: the engine is as it was
                error.ClipEncoderNeedsMoreDeviceMemory => {
                    h.finish(job, .failed, 0, "cloning from a clip needs 5.26 GiB of device memory for its encoder, and less than that is free");
                    continue;
                },
                else => {
                    h.lock();
                    h.failed = true;
                    h.unlock();
                    h.e.stream.synchronize() catch {};
                    h.finish(job, .failed, 0, @errorName(err));
                    continue;
                },
            };
            h.finish(job, reason, samples, "");
        }
    }

    /// One request: its prompt rows, then frames to the end of audio or the budget, their audio into its buffer.
    fn serve(h: *Host, job: Job, samples: *usize) !api.Reason {
        const r = job.request;
        const gpa = h.gpa;
        var rows = r.frames;
        var cloned: [][W]i32 = &.{};
        defer gpa.free(cloned);
        if (r.reference_audio.len != 0) {
            if (h.clip == null) {
                if (h.admit) |a| if (clip_bytes > try a.room(a.ptr, &h.e.ctx)) return error.ClipEncoderNeedsMoreDeviceMemory;
                const before = cuda.usage(false).device;
                h.clip = try z3.clip.Clip.create(gpa, h.io, &h.e.driver, h.e.stream, h.dir, h.rel.generator, h.codec.w);
                const took = cuda.usage(false).device - before;
                std.log.info("zonos3: clip encoder loaded, {d} bytes on the device ({d} admitted)", .{ took, clip_bytes });
                if (took > clip_bytes) return error.ClipEncoderOverItsAdmission;
            }
            const codes = try h.clip.?.encode(gpa, r.reference_audio);
            defer gpa.free(codes);
            const prefix = try z3.prompt.referenceRows(gpa, codes);
            defer gpa.free(prefix);
            cloned = try gpa.alloc([W]i32, prefix.len + r.frames.len);
            @memcpy(cloned[0..prefix.len], prefix);
            @memcpy(cloned[prefix.len..], r.frames);
            rows = cloned;
        }
        const layout = try Layout.of(rows);
        var replaced: [1 + cfg.conditions]z3.Replace = undefined;
        var n: usize = 0;
        if (r.speaker.len != 0) {
            replaced[n].row = 0;
            h.e.w.speakerRow(r.speaker[0..cfg.speaker_dim], &replaced[n].vec);
            n += 1;
        }
        // every condition unknown: the last row of its table
        for (0..cfg.conditions) |c| {
            replaced[n].row = layout.condition_row + c;
            @memcpy(&replaced[n].vec, h.e.w.conditions[c][h.e.c.buckets[c] * cfg.dim ..][0..cfg.dim]);
            n += 1;
        }
        const sampling: z3.Sampling = .{ .temperature = r.temperature, .top_k = r.top_k, .top_p = r.top_p, .seed = r.seed };
        try h.e.reset();
        try h.codec.reset();
        var audio: Audio = .{ .h = h, .job = job, .prefill = rows[rows.len - layout.prefill ..] };
        const made = try z3.generation.generateWatched(gpa, Guarded{ .h = h }, rows, @as([]const z3.Replace, replaced[0..n]), sampling, r.max_frames, &audio);
        defer gpa.free(made.rows);
        samples.* = audio.samples;
        return if (made.eos_frame != null) .stop else .length;
    }

    fn finish(h: *Host, job: Job, reason: api.Reason, samples: usize, message: []const u8) void {
        h.lock();
        if (reason == .stop or reason == .length) job.request.result.samples = samples;
        job.request.result.reason = reason;
        h.job = null;
        h.unlock();
        job.sink.event(job.sink.ctx, job.id, &.{ .finished = .{ .reason = reason, .message = message } });
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
        h.e.ctx.makeCurrent() catch {};
        if (h.clip) |c| c.destroy();
        h.codec.destroy();
        h.e.destroy();
        h.rel.deinit();
        h.gpa.free(h.dir);
        h.gpa.destroy(h);
    }
};

/// `dir`: a snapshot of the release's repository.
pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !api.Opened {
    const b = try plan(o);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    var rel = try z3.release.open(gpa, io, dir, null);
    errdefer rel.deinit();
    var door: Door = .{ .o = o, .need = b.need };
    const e = try z3.Engine.create(gpa, io, rel.lm, .{ .device = o.device, .max_ctx = o.context, .gate = .{ .ptr = &door, .check = Door.check } });
    errdefer e.destroy();
    const codec = try z3.Codec.create(gpa, io, &e.driver, e.stream, rel.generator, rel.head, b.codec);
    errdefer codec.destroy();
    if (cuda.usage(false).device > door.need) return error.LoadOverItsAdmission;
    const kept = try gpa.dupe(u8, dir);
    errdefer gpa.free(kept);
    h.* = .{
        .gpa = gpa,
        .io = io,
        .dir = kept,
        .rel = rel,
        .e = e,
        .codec = codec,
        .admit = o.admit,
        .info_ = .{
            .name = "ZONOS3 Mini",
            .text_generation = false,
            .lanes = 1,
            .context_window = o.context,
            .plain_only = true,
            .prompt_cache = false,
            .speech = .{ .sample_rate = z3.codec.sample_rate, .frame_length = z3.codec.frame_samples, .max_frames = b.frames, .max_prompt_tokens = b.prompt },
        },
    };
    h.worker = try std.Thread.spawn(.{}, Host.run, .{h});
    const held = cuda.usage(false).device;
    std.log.info("zonos3: decoder {s}, {d} frames a request, {d} bytes on the device ({d} admitted)", .{ h.rel.decoder, b.frames, held, door.need });
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}

test "need: the weights, then state that grows with the context and with the frame budget" {
    const t = std.testing;
    const base = need(4096, 2000);
    try t.expectEqual(@as(u64, 3_828_406_528), base);
    try t.expect(need(8192, 2000) > base and need(4096, 1000) < base);
    // two attention layers keep a key row and a value row a position
    try t.expectEqual(@as(u64, 2 * 2 * 4096 * cfg.kv_width * 4), need(8192, 2000) - base);
}

test "budget: the release's frame budget where the context holds it, the rest for the prompt" {
    const t = std.testing;
    const d = try budget(4096, null);
    try t.expectEqual(@as(u32, 2000), d.frames);
    try t.expectEqual(@as(u32, 2096), d.prompt);
    try t.expectEqual(@as(u32, 512), (try budget(1025, null)).frames);
    try t.expectError(error.BadZonosContext, budget(1024, null));
    try t.expectError(error.BadZonosContext, budget(32769, null));
    try t.expectError(error.BadZonosFrames, budget(4096, 4096));
    try t.expectError(error.BadZonosFrames, budget(4096, 5));
    try t.expectEqual(@as(u32, 8192), (try budget(32768, 8192)).frames);
    try t.expectError(error.BadZonosFrames, budget(32768, 8193));
}

test "plan: every frame budget a load admits is one the codec opens with, its pass no longer than the budget" {
    const t = std.testing;
    for ([_]u32{ 12, 16, 31, 32 }) |n| {
        const p = try plan(.{ .max_frames = n });
        try t.expectEqual(n, p.frames);
        try p.codec.validate();
        try t.expectEqual(@as(usize, n), p.codec.max_frames);
        try t.expectEqual(@as(usize, @min(n, 32)), p.codec.pass_frames);
        try t.expectEqual(need(4096, n), p.need);
    }
    try t.expectError(error.BadZonosFrames, plan(.{ .max_frames = 11 }));
    const d = try plan(.{});
    try t.expectEqual(@as(usize, 32), d.codec.pass_frames);
    try t.expectEqual(@as(u64, 3_828_406_528), d.need);
}

test "chunkFrames: a streamed reply's chunks double to a second" {
    const t = std.testing;
    var spoken: usize = 0;
    for ([_]usize{ 1, 2, 4, 8, 15, 15 }) |want| {
        try t.expectEqual(want, chunkFrames(spoken));
        spoken += want;
    }
}

test "Layout: the condition rows sit before the text and the silence rows after it" {
    const t = std.testing;
    var p = try z3.prompt.build(t.allocator, &.{ 9, 8 }, .embedding, @splat(1), 3);
    defer p.deinit(t.allocator);
    const l = try Layout.of(p.rows);
    try t.expectEqual(p.condition_row, l.condition_row);
    try t.expectEqual(@as(usize, 3), l.prefill);
    try t.expectError(error.BadPromptLayout, Layout.of(p.rows[0..3]));
}
