//! Native speech from one ZONOS3 Mini family on CUDA: one request at a time, its audio decoded whole at the end.
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
};

/// The frame budget and the prompt rows a context leaves.
pub fn budget(context: u32, max_frames: ?u32) !struct { frames: u32, prompt: u32 } {
    if (context < 1025 or context > cfg.max_seqlen) return error.BadZonosContext;
    const frames = max_frames orelse @min(default_frames, context / 2);
    if (frames < C + 2 or frames >= context) return error.BadZonosFrames;
    return .{ .frames = frames, .prompt = context - frames };
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

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    rel: z3.release.Release,
    e: *z3.Engine,
    codec: *z3.Codec,
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

    /// One request: its prompt rows, frames to the end of audio or the budget, then the whole audio into its buffer.
    fn serve(h: *Host, job: Job, samples: *usize) !api.Reason {
        const r = job.request;
        const gpa = h.gpa;
        const layout = try Layout.of(r.frames);
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
        const made = try z3.generation.generate(gpa, Guarded{ .h = h }, r.frames, @as([]const z3.Replace, replaced[0..n]), sampling, r.max_frames);
        defer gpa.free(made.rows);
        const frames = try z3.speech.frames(gpa, r.frames[r.frames.len - layout.prefill ..], made.rows, made.eos_frame);
        defer gpa.free(frames);
        try h.codec.reset();
        try h.codec.decode(frames, null);
        const pcm = try h.codec.finish(gpa, null);
        defer gpa.free(pcm);
        if (pcm.len > r.pcm.len) return error.AudioOverflow;
        for (pcm) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
        z3.speech.output(pcm, h.rel.headroom_db);
        @memcpy(r.pcm[0..pcm.len], pcm);
        samples.* = pcm.len;
        if (r.stream) job.sink.event(job.sink.ctx, job.id, &.{ .audio = pcm.len });
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
        h.codec.destroy();
        h.e.destroy();
        h.rel.deinit();
        h.gpa.destroy(h);
    }
};

/// `dir`: a snapshot of the release's repository.
pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !api.Opened {
    const b = try budget(o.context, o.max_frames);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    var rel = try z3.release.open(gpa, io, dir, null);
    errdefer rel.deinit();
    const e = try z3.Engine.create(gpa, io, rel.lm, .{ .device = o.device, .max_ctx = o.context });
    errdefer e.destroy();
    const codec = try z3.Codec.create(gpa, io, &e.driver, e.stream, rel.generator, rel.head, .{ .max_frames = b.frames });
    errdefer codec.destroy();
    h.* = .{
        .gpa = gpa,
        .io = io,
        .rel = rel,
        .e = e,
        .codec = codec,
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
    std.log.info("zonos3: decoder {s}, {d} MiB of weights, {d} frames a request", .{ h.rel.decoder, (e.w.bytes + codec.w.bytes) >> 20, b.frames });
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
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
