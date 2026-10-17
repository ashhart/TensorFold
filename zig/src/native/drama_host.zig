//! Drama 3 speech for the native server: one synthesis at a time on a worker thread that owns the CUDA context.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const drama = @import("drama");
const budget = @import("cuda_memory.zig");
const Allocator = std.mem.Allocator;

const Frame = drama.tower.Frame;
const Codes = [10]u32;

/// Backbone and tower rows a prompt pass takes at most.
const chunk_rows = 128;
/// Generation steps a request may take at most, the end-of-speech step included.
const max_frames = 1024;
/// Prompt tokens a request may carry at most, whatever the context.
const prompt_cap = 4096;
const min_context = max_frames + 1;
const max_context = 8192;
/// The tower's hidden width, which every backbone row it reads must have.
const tower_hidden = 2048;
/// Semantic logits a backbone row carries: EOS first, then the 4096 codes.
const logit_cols = drama.backbone.weights.head_cols;
const sample_rate = 44100;
const frame_length = 2048;
/// Frames a streamed request decodes together after its first, which is decoded alone for the earliest audio.
const stream_chunk_frames = 4;

pub const Options = struct {
    context: usize = 8192,
    device: u32 = 0,
};

const Job = struct {
    id: api.Id,
    request: *const api.SpeechRequest,
    sink: api.Sink,
};

const Host = struct {
    gpa: Allocator,
    io: std.Io,
    context: usize,
    driver: cuda.Driver = undefined,
    ctx: cuda.Context = undefined,
    stream: cuda.Stream = undefined,
    backbone: drama.backbone.Model = undefined,
    tower: drama.tower.Model = undefined,
    codec: drama.codec.Model = undefined,
    info_: api.Info = .{},
    startup: []u8 = &.{},
    previous_allocation_limit: ?u64 = null,
    semantic_start: u32 = 0,
    semantic_end: u32 = 0,
    eos: u32 = 0,
    thread: ?std.Thread = null,

    // Worker-thread only: the prompt pass's frames and the generated codes, sized once at open.
    frames: [chunk_rows + 1]Frame = undefined,
    codes: [max_frames]Codes = undefined,

    // Under `mutex`: the one admitted request and what status reports of it.
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    slot: ?Job = null,
    running: bool = false,
    cancelled: bool = false,
    closing: bool = false,
    live_prompt: u64 = 0,
    live_frames: u64 = 0,
    decode_began: ?i96 = null,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,

    const window_ns: i96 = 2 * std.time.ns_per_s;

    const vtable: api.Engine.VTable = .{
        .info = infoFn,
        .submit = submitFn,
        .cancel = cancelFn,
        .status = statusFn,
        .memory = memoryFn,
        .submit_speech = submitSpeechFn,
    };

    fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &vtable };
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

    /// Speech-only: the text completion routes are refused.
    fn submitFn(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Unsupported;
    }

    /// Admits `request` when it fits the advertised limits and no other synthesis is admitted.
    fn submitSpeechFn(ctx: *anyopaque, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const h = self(ctx);
        if (!h.valid(request)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing) return error.Closed;
        if (h.slot != null) return error.Busy;
        h.slot = .{ .id = id, .request = request, .sink = sink };
        h.cancelled = false;
        h.wake.signal(h.io);
    }

    /// Sizes, sampling and prompt tokens a plain built-in-voice synthesis may use (immutable host fields only).
    fn valid(h: *const Host, r: *const api.SpeechRequest) bool {
        const speech = h.info_.speech.?;
        if (r.prompt.len == 0 or r.prompt.len > speech.max_prompt_tokens) return false;
        if (r.max_frames == 0 or r.max_frames > speech.max_frames) return false;
        if (r.pcm.len < @as(usize, r.max_frames) * speech.frame_length) return false;
        // the tower's sampler accepts exactly these (tower.generate's BadSpeechSampling)
        if (!std.math.isFinite(r.temperature) or r.temperature < 0) return false;
        if (!std.math.isFinite(r.top_p) or r.top_p <= 0 or r.top_p > 1) return false;
        if (r.top_k == 0 or r.top_k > std.math.maxInt(i32)) return false;
        for (r.prompt) |t| {
            if (t >= drama.backbone.config.vocab) return false;
            if (t >= h.semantic_start and t <= h.semantic_end) return false; // plain prompts carry no audio rows
        }
        return true;
    }

    /// Ends the admitted request at its next prompt chunk, frame or codec boundary; unknown ids are ignored.
    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const job = h.slot orelse return;
        if (job.id != id) return;
        h.cancelled = true;
        h.wake.signal(h.io);
    }

    /// Host counters only: HTTP threads never touch the CUDA context.
    fn statusFn(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const t = h.now();
        var rate: f64 = 0;
        if (h.running) if (h.decode_began) |began| if (t > began) {
            rate = @as(f64, @floatFromInt(h.live_frames)) / (@as(f64, @floatFromInt(t - began)) / 1e9);
        };
        var streams: usize = 0;
        if (h.running and stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(h.live_prompt + h.live_frames);
            streams = 1;
        }
        out.* = .{
            .running = @intFromBool(h.running),
            .waiting = @intFromBool(h.slot != null and !h.running),
            .decode_tokens_per_second = rate,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = streams,
            .generation_tokens = if (h.running) h.live_frames else 0,
        };
    }

    /// The process's CUDA allocation counters (atomics, not a driver query).
    fn memoryFn(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .cache = 0, .peak = u.peak, .pinned = u.host };
    }

    /// The worker: takes the admitted request, synthesizes it, finishes it; on close, cancels what is admitted.
    fn run(h: *Host) void {
        // CUDA binds per thread; synthesize binds again and reports a failure there
        h.ctx.makeCurrent() catch |e| std.log.err("drama: cuCtxSetCurrent on the worker thread: {s}", .{@errorName(e)});
        while (true) {
            h.lock();
            while (h.slot == null and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
            const job = h.slot orelse {
                h.unlock();
                return;
            };
            h.running = true;
            h.live_prompt = job.request.prompt.len;
            h.live_frames = 0;
            h.decode_began = null;
            h.unlock();
            h.serve(job);
        }
    }

    /// True when the admitted request was cancelled or the engine is closing (read under the lock).
    fn stopped(h: *Host) bool {
        h.lock();
        defer h.unlock();
        return h.cancelled or h.closing;
    }

    fn check(h: *Host) error{Cancelled}!void {
        if (h.stopped()) return error.Cancelled;
    }

    /// One synthesis to its single terminal event; nothing touches the request after `finished`.
    fn serve(h: *Host, job: Job) void {
        const r = job.request;
        var stats: api.Stats = .{};
        const began = h.now();
        const samples = h.synthesize(job, &stats, began) catch |e| {
            h.stream.synchronize() catch {}; // no device work outlives the request it served
            if (e == error.Cancelled) return h.finish(job, .cancelled, stats, "");
            std.log.warn("drama: synthesis {d} failed: {s}", .{ job.id, @errorName(e) });
            return h.finish(job, .failed, stats, words(e));
        };
        r.result.samples = samples; // published only after the full decode and the final cancellation check
        h.finish(job, .stop, stats, "");
    }

    /// Frees the slot (a new request may be admitted while this one's event is out), then sends `finished`.
    fn finish(h: *Host, job: Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        h.slot = null;
        h.running = false;
        h.cancelled = false;
        h.live_frames = 0;
        h.decode_began = null;
        h.unlock();
        const event: api.Event = .{ .finished = .{ .reason = reason, .stats = stats, .message = message } };
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// Prompt pass, frame generation and codec decode into the request's PCM (streamed by `publish`); returns samples.
    fn synthesize(h: *Host, job: Job, stats: *api.Stats, began: i96) !usize {
        const r = job.request;
        try h.check();
        try h.ctx.makeCurrent(); // CUDA binds per thread: the worker's calls need the context current
        const b = &h.backbone;
        const t = &h.tower;
        const s = h.stream;
        try b.reset(s);
        try t.reset(s, r.prompt, r.seed);

        // Prompt pass: chunks staged in the tower, run through the backbone, teacher-forced but the prompt's last row.
        const prompt = r.prompt;
        var last: drama.backbone.Rows = undefined;
        var start: usize = 0;
        while (start < prompt.len) {
            try h.check();
            const end = @min(start + chunk_rows, prompt.len);
            const n = end - start;
            const look = @min(end + 1, prompt.len); // this chunk's frames and the next row's, for its last label
            for (prompt[start..look], h.frames[0 .. look - start]) |token, *frame| frame.* = plainFrame(token);
            const extra = try t.setInput(s, h.frames[0..n]);
            last = try b.run(s, .prefill, prompt[start..end], extra);
            if (last.rows != n) return error.DramaRowsMismatch;
            const taught = look - start - 1; // n inside the prompt; n - 1 in its final chunk
            if (taught > 0) try t.prefill(s, last.hidden, h.frames[1 .. 1 + taught]);
            // the final row's tower pass is generate's: it stages that row alone when prefill consumed the chunk
            if (end == prompt.len and taught > 0) _ = try t.setInput(s, h.frames[n - 1 .. n]);
            start = end;
        }
        const prefilled = h.now();
        stats.prefill_seconds = @as(f64, @floatFromInt(prefilled - began)) / 1e9;
        h.notePrefill(prompt.len, began, prefilled);

        // Generation: each step samples a frame; a non-EOS frame is the next backbone row, max_frames steps at most.
        const sampling: drama.tower.Sampling = .{ .temperature = r.temperature, .top_p = r.top_p, .top_k = r.top_k };
        var hidden = try lastRow(last.hidden, last.rows, b.hidden * 2);
        var logits = try lastRow(last.semantic_logits, 1, logit_cols * 4);
        var count: usize = 0;
        var published: usize = 0; // frames a streamed request's caller owns
        if (r.stream) h.codec.begin();
        while (true) {
            try h.check();
            const frame = try t.generate(s, logits, hidden, sampling);
            stats.rounds += 1;
            if (frame[0] == h.eos) break;
            if (frame[0] < h.semantic_start or frame[0] > h.semantic_end) return error.BadSpeechFrame;
            h.codes[count] = frame[1..11].*;
            count += 1;
            h.noteFrames(count);
            if (count == r.max_frames) return error.SpeechTruncated;
            if (r.stream and (published == 0 or count - published == stream_chunk_frames)) published = try h.publish(job, count);
            h.frames[0] = frame;
            const extra = try t.setInput(s, h.frames[0..1]);
            const rows = try b.run(s, .decode, h.frames[0][0..1], extra);
            if (rows.rows != 1) return error.DramaRowsMismatch;
            hidden = try lastRow(rows.hidden, 1, b.hidden * 2);
            logits = try lastRow(rows.semantic_logits, 1, logit_cols * 4);
        }
        if (count == 0) return error.EmptySpeech;

        try h.check();
        if (r.stream) {
            if (count > published) _ = try h.publish(job, count);
            return count * frame_length;
        }
        const want = count * frame_length;
        const samples = try h.codec.decode(s, h.codes[0..count], r.pcm);
        if (samples != want) return error.CodecSampleCount;
        for (r.pcm[0..samples]) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
        try h.check();
        return samples;
    }

    /// Decodes the frames not yet decoded, checks them, then hands pcm[0..frames * 2048] over with an `audio` event.
    fn publish(h: *Host, job: Job, frames: usize) !usize {
        try h.check();
        const r = job.request;
        const from = h.codec.decoded * frame_length;
        const samples = try h.codec.advance(h.stream, h.codes[0..frames], r.pcm);
        if (samples != frames * frame_length) return error.CodecSampleCount;
        for (r.pcm[from..samples]) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
        try h.check();
        const event: api.Event = .{ .audio = samples };
        job.sink.event(job.sink.ctx, job.id, &event);
        return frames;
    }

    fn notePrefill(h: *Host, tokens: usize, began: i96, done: i96) void {
        h.lock();
        defer h.unlock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(tokens)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        h.decode_began = done;
    }

    fn noteFrames(h: *Host, frames: usize) void {
        h.lock();
        defer h.unlock();
        h.live_frames = frames;
    }
};

/// A plain prompt row's frame: the text token, no codebook values.
fn plainFrame(token: u32) Frame {
    var frame: Frame = @splat(0);
    frame[0] = token;
    return frame;
}

/// A borrowed view of the last row; the backbone's semantic logits contain exactly one row.
fn lastRow(buffer: cuda.DeviceBuffer, rows: usize, row_bytes: usize) !cuda.DeviceBuffer {
    if (rows == 0 or buffer.len != rows * row_bytes) return error.DramaRowsMismatch;
    const at = (rows - 1) * row_bytes;
    return .{ .d = buffer.d, .ptr = buffer.ptr + at, .len = row_bytes, .borrowed = true };
}

/// The failure message a finished event carries (static text: it outlives the sink call).
fn words(err: anyerror) []const u8 {
    return switch (err) {
        error.SpeechTruncated => "no end of speech within the request's max_frames; truncated audio is not returned",
        error.EmptySpeech => "the model ended speech before its first audio frame",
        error.NonFiniteSpeechLogits => "the model produced non-finite speech logits",
        error.NonFiniteAudio => "the codec produced non-finite audio",
        error.BadSpeechFrame => "the tower sampled a token outside the semantic band",
        error.CodecSampleCount => "the codec returned a sample count other than frames x frame_length",
        error.DramaRowsMismatch => "the backbone returned rows other than those it was given",
        error.DramaContextOverflow => "the prompt and frames do not fit the engine's context",
        error.OutOfDeviceMemory => "the GPU ran out of memory during synthesis",
        else => @errorName(err),
    };
}

/// Drama 3 at `checkpoint_dir` with kernels from `kernel_dir`, served on its own worker thread; `close` joins it.
pub fn open(gpa: Allocator, io: std.Io, checkpoint_dir: []const u8, kernel_dir: []const u8, options: Options) !api.Opened {
    if (options.context < min_context or options.context > max_context) return error.BadDramaContext;
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .context = options.context };

    h.driver = try cuda.Driver.open();
    errdefer h.driver.close();
    const count = try h.driver.deviceCount();
    if (count <= 0 or options.device >= @as(u32, @intCast(count))) return error.NoSuchCudaDevice;
    h.ctx = try cuda.Context.init(&h.driver, @intCast(options.device));
    errdefer h.ctx.deinit();
    h.stream = try cuda.Stream.init(&h.driver, true);
    errdefer h.stream.deinit();
    var admission_arena: std.heap.ArenaAllocator = .init(gpa);
    defer admission_arena.deinit();
    var admission_problem: []const u8 = "";
    const plan = try budget.readPool(admission_arena.allocator(), io, &h.ctx, &admission_problem) orelse {
        std.log.err("Drama memory admission: {s}", .{admission_problem});
        return error.DramaMemoryAdmission;
    };
    const used = cuda.usage(false);
    const held = used.device + used.host;
    h.previous_allocation_limit = try cuda.setAllocationLimit(held + plan.room(held));
    errdefer _ = cuda.setAllocationLimit(h.previous_allocation_limit) catch unreachable;

    const held0 = cuda.usage(false).device;
    h.backbone = try drama.backbone.Model.init(gpa, io, &h.ctx, checkpoint_dir, kernel_dir, .{ .context = options.context, .max_rows = chunk_rows });
    errdefer h.backbone.deinit();
    h.tower = try drama.tower.Model.init(gpa, io, &h.ctx, checkpoint_dir, kernel_dir, .{ .context = options.context, .max_rows = chunk_rows });
    errdefer h.tower.deinit();
    h.codec = try drama.codec.Model.init(gpa, io, &h.ctx, checkpoint_dir, kernel_dir, .{ .max_frames = max_frames });
    errdefer h.codec.deinit();
    const model_bytes = cuda.usage(false).device -| held0;

    if (h.backbone.hidden != tower_hidden) return error.DramaHiddenMismatch;
    if (h.codec.sample_rate != sample_rate or h.codec.frame_length != frame_length or h.codec.num_codebooks != 10) return error.UnsupportedDramaCodec;
    h.semantic_start = h.backbone.semantic_start;
    h.semantic_end = h.backbone.semantic_end;
    h.eos = h.backbone.eos;

    var name_buf: [256]u8 = undefined;
    const name = h.ctx.name(&name_buf) catch "GPU";
    const capability = try h.ctx.capability();
    const max_prompt: u32 = @intCast(@min(prompt_cap, options.context - max_frames));
    h.startup = try std.fmt.allocPrint(gpa, "Drama 3 on CUDA sm_{d} device {d} ({s}): models {d:.2} GiB; context {d} tokens, prompts up to {d} tokens in passes of {d} rows, up to {d} frames of {d} samples at {d} Hz; one synthesis at a time", .{
        capability, options.device, name, @as(f64, @floatFromInt(model_bytes)) / (1 << 30), options.context, max_prompt, chunk_rows, max_frames, frame_length, sample_rate,
    });
    errdefer gpa.free(h.startup);
    h.info_ = .{
        .name = "drama-3",
        .text_generation = false,
        .speech = .{ .sample_rate = sample_rate, .frame_length = frame_length, .max_frames = max_frames, .max_prompt_tokens = max_prompt },
        .lanes = 1,
        .context_window = @intCast(options.context),
        .thinking_budget = false,
        .plain_only = true,
        .startup = h.startup,
    };
    h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, Host.run, .{h});
    return .{ .engine = h.engine(), .close = close, .ctx = h };
}

/// Stops admitting, cancels the admitted request (its finished event still arrives), joins the worker, frees all.
fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.lock();
    h.closing = true;
    h.wake.broadcast(h.io);
    h.unlock();
    if (h.thread) |t| t.join();
    h.thread = null;
    h.ctx.makeCurrent() catch {}; // the closing thread frees device memory
    h.stream.synchronize() catch {};
    h.codec.deinit();
    h.tower.deinit();
    h.backbone.deinit();
    h.stream.deinit();
    _ = cuda.setAllocationLimit(h.previous_allocation_limit) catch |e| blk: {
        std.log.err("Drama memory budget restore: {s}", .{@errorName(e)});
        break :blk null;
    };
    h.ctx.deinit();
    h.driver.close();
    h.gpa.free(h.startup);
    h.gpa.destroy(h);
}
