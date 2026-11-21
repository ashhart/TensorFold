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
    parallel: u32 = 1, // requests at once, taking turns a frame at a time
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

/// Device bytes a guided request's second stream takes, allocated by the first such request.
pub fn twinBytes(o: z3.Options) u64 {
    return z3.Engine.laneBytes(o);
}

/// Device bytes a load takes at a context and frame budget: the weights, then each part's own count of its state.
pub fn need(context: u32, frames: u32) u64 {
    return weight_bytes + z3.Engine.stateBytes(.{ .max_ctx = context }) + z3.Codec.stateBytes(.{ .max_frames = frames });
}

/// Device bytes each request at once past the first adds: a backbone lane and a decoder lane.
pub fn laneBytes(context: u32, frames: u32) u64 {
    return z3.Engine.laneBytes(.{ .max_ctx = context }) + z3.Codec.laneBytes(.{ .max_frames = frames });
}

/// Requests at once a server may ask for.
pub const max_parallel = 16;

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
    return .{ .frames = frames, .prompt = context - frames };
}

pub const Layout = @import("zonos_layout.zig").Layout;
pub const chunkFrames = @import("zonos_layout.zig").chunkFrames;

const Job = struct { id: api.Id, request: *const api.SpeechRequest, sink: api.Sink };
const LaciRun = z3.laci_run.Run(z3.Replace);

/// A request's place among those running: its lanes on the backbone and the decoder, and its generation so far.
const Slot = struct {
    index: usize, // its backbone lane and its decoder lane
    twin: ?usize = null, // the backbone lane of its guidance twin, made by its first guided request
    job: ?Job = null,
    cancelled: bool = false,
    started: bool = false,
    run: z3.generation.Run = undefined,
    audio: Audio = undefined,
    cloned: [][W]i32 = &.{}, // a clip's reference prefix and the request's rows
    steering: z3.steering.Plan = .{}, // the request's controls, over its request's own slices
    delta: []f32 = &.{}, // a frame's delta, then scratch of the same length
    steer: []z3.Steer = &.{}, // that delta by bank source
    spans: []z3.steering.Span = &.{},
    twin_steered: bool = false, // a speaker twin takes the deltas too; a controls twin is the prompt without them
    laci: ?LaciRun = null, // a request under LACI: its attempts, in place of `run`
    heads: []z3.laci.Head = &.{},
    short: z3.short_text.Config = undefined, // a short text's cut settings, over the request's weights
    kept: [1 + cfg.conditions]z3.Replace = undefined, // the replaced rows, which a regeneration's prompt takes again

    fn steerFor(ctx: *anyopaque, frame: usize) []const z3.Steer {
        return @as(*Slot, @ptrCast(@alignCast(ctx))).steerAt(frame);
    }

    /// The deltas of the forward that takes generated row `frame`; none when the frame is not steered.
    fn steerAt(s: *Slot, frame: usize) []const z3.Steer {
        if (s.steer.len == 0) return &.{};
        const half = s.delta.len / 2;
        return if (s.steering.at(frame, s.delta[0..half], s.delta[half..])) s.steer else &.{};
    }
};

/// The backbone as one slot's generation sees it: its lane, and its twin's between swaps.
const Guarded = struct {
    h: *Host,
    slot: *Slot,

    pub fn run(g: Guarded, rows: []const [W]i32, replaced: []const z3.Replace, probe: ?z3.Probe) !void {
        try g.h.e.run(rows, replaced, probe);
    }

    pub fn lastLogits(g: Guarded, out: *[cfg.head_rows]f32) !void {
        try g.h.e.lastLogits(out);
    }

    pub fn reset(g: Guarded) !void {
        try g.h.e.reset();
    }

    pub fn keys(g: Guarded, layer: usize, first: usize, out: []f32) !void {
        try g.h.e.keys(layer, first, out);
    }

    pub fn swap(g: Guarded) void {
        g.h.e.use(if (g.h.e.active == g.slot.index) g.slot.twin.? else g.slot.index);
    }
};

/// A request's audio: streamed or whole, it decodes in the same growing chunks (cuBLAS rounds by a call's rows).
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
        // StreamDecoder: a reply that ends before its first generated frame is empty, the silence rows unspoken
        if (finished and s.decoded == 0 and all.len <= s.prefill.len) return;
        const before = s.decoded;
        while (s.decoded < all.len) {
            const chunk = (s.prefill.len -| s.decoded) + chunkFrames(s.decoded -| s.prefill.len);
            if (!finished and all.len - s.decoded < chunk) break;
            const n = @min(chunk, all.len - s.decoded);
            try h.codec.decode(all[s.decoded..][0..n], null);
            s.decoded += n;
        }
        if (s.decoded == before and !finished) return;
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
    slots: []Slot,
    turns: []z3.Turn, // the rows the requests in progress run next, one pass for all: two a slot at most
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
        // guidance: the host makes the twin itself, from a prompt that has a voice to drop
        if (r.unconditioned.len != 0 or !std.math.isFinite(r.cfg_gamma) or r.cfg_gamma < 0) return false;
        if (r.conditions.len != 0 and r.conditions.len != cfg.conditions) return false;
        for (r.conditions, 0..) |bucket, i| if (bucket > h.e.c.buckets[i]) return false;
        for (r.frames) |row| {
            for (row[0..C]) |code| if (code < 0 or code >= cfg.audio_vocab) return false;
            if (row[C] < 0 or row[C] >= h.e.w.textIds()) return false;
        }
        const layout = Layout.of(r.frames) catch return false;
        if (r.cfg_gamma != 1 and layout.condition_row == 0 and r.reference_audio.len == 0) return false;
        if (r.steering) |st| if (!z3.steering.valid(st, r.cfg_gamma != 1)) return false;
        // LACI runs one stream: no guidance twin of either kind
        if (r.laci) |lc| if (!z3.laci.valid(lc, r.cfg_gamma != 1 or (r.steering != null and r.steering.?.guidance != 1))) return false;
        return true;
    }

    fn submit(p: *anyopaque, id: api.Id, r: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const h = self(p);
        if (!h.valid(r)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing or h.failed) return error.Closed;
        const slot = for (h.slots) |*s| {
            if (s.job == null) break s;
        } else return error.Busy;
        r.result.* = .{};
        slot.job = .{ .id = id, .request = r, .sink = sink };
        slot.cancelled = false;
        slot.started = false;
        h.wake.signal(h.io);
    }

    fn cancel(p: *anyopaque, id: api.Id) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        for (h.slots) |*s| if (s.job) |j| if (j.id == id) {
            s.cancelled = true;
        };
    }

    fn status(p: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        out.* = .{};
        for (h.slots) |s| out.running += @intFromBool(s.job != null);
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .pinned = u.host, .peak = u.peak };
    }

    /// The worker: every request in progress takes a turn, a frame each, until none is left.
    fn run(h: *Host) void {
        const current = if (h.e.ctx.makeCurrent()) |_| true else |_| false;
        while (true) {
            h.lock();
            var waiting = false;
            for (h.slots) |s| waiting = waiting or s.job != null;
            if (!waiting) {
                if (h.closing) {
                    h.unlock();
                    return;
                }
                h.wake.waitUncancelable(h.io, &h.mutex);
                h.unlock();
                continue;
            }
            h.unlock();
            var n: usize = 0;
            for (h.slots) |*slot| h.turn(slot, current, &n);
            h.advance(h.turns[0..n]);
        }
    }

    /// The rows the turns left to run, every request's lane and twin lane in one pass.
    fn advance(h: *Host, turns: []const z3.Turn) void {
        h.e.runLanes(turns) catch |err| {
            h.lock();
            h.failed = true;
            h.unlock();
            h.e.stream.synchronize() catch {};
            // every request in progress shared the pass
            for (h.slots) |*slot| if (slot.started) if (slot.job) |job| h.finish(slot, job, .failed, @errorName(err));
        };
    }

    /// One slot's turn: a new request's prompt, or a running one's next frame, whose row joins `turns[n.*..]`.
    fn turn(h: *Host, slot: *Slot, current: bool, n: *usize) void {
        h.lock();
        const job = slot.job orelse return h.unlock();
        const dead = h.failed or !current;
        const stop = slot.cancelled or h.closing;
        h.unlock();
        if (dead) return h.finish(slot, job, .failed, "speech engine closed after execution failure");
        if (stop) return h.finish(slot, job, .cancelled, "");
        h.e.use(slot.index);
        h.codec.use(slot.index);
        const reason = h.step(slot, job, n) catch |err| switch (err) {
            // refused before anything ran on the device: the engine is as it was
            error.ClipEncoderNeedsMoreDeviceMemory => return h.finish(slot, job, .failed, "cloning from a clip needs 5.26 GiB of device memory for its encoder, and less than that is free"),
            error.GuidanceNeedsMoreDeviceMemory => return h.finish(slot, job, .failed, "speaker guidance needs device memory for a second stream's state, and less than that is free"),
            else => {
                h.lock();
                h.failed = true;
                h.unlock();
                h.e.stream.synchronize() catch {};
                return h.finish(slot, job, .failed, @errorName(err));
            },
        } orelse return;
        h.finish(slot, job, reason, "");
    }

    /// The request's prompt rows on its first turn, then a frame a turn; its reason once its last frame is spoken.
    fn step(h: *Host, slot: *Slot, job: Job, n: *usize) !?api.Reason {
        if (!slot.started) {
            try h.start(slot, job);
            return null;
        }
        if (slot.laci) |*lr| {
            // a failed attempt's regeneration runs its prompt inside this turn
            const got = try lr.step(Guarded{ .h = h, .slot = slot });
            try slot.audio.row(got.rows, got.eos_frame, got.final);
            if (got.final) return if (got.eos_frame != null) .stop else .length;
            const next = got.turn orelse return null;
            h.turns[n.*] = .{ .lane = slot.index, .row = next.row, .out = @ptrCast(lr.run.logits), .steer = next.steer, .probe = next.probe };
            n.* += 1;
            return null;
        }
        const finished = try slot.run.sample();
        try slot.audio.row(slot.run.rows.items, slot.run.eos_frame, finished);
        if (finished) return if (slot.run.eos_frame != null) .stop else .length;
        const row = slot.run.lastRow();
        const steer = slot.steerAt(slot.run.rows.items.len);
        h.turns[n.*] = .{ .lane = slot.index, .row = row, .out = @ptrCast(slot.run.logits), .steer = steer };
        n.* += 1;
        if (slot.run.unconditional) |out| {
            h.turns[n.*] = .{ .lane = slot.twin.?, .row = row, .out = out, .steer = if (slot.twin_steered) steer else &.{} };
            n.* += 1;
        }
        return null;
    }

    /// A request's first turn: its clip encoded, its lanes zeroed, its prompt rows (and its twin's) run.
    fn start(h: *Host, slot: *Slot, job: Job) !void {
        const r = job.request;
        const gpa = h.gpa;
        var rows = r.frames;
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
            slot.cloned = try gpa.alloc([W]i32, prefix.len + r.frames.len);
            @memcpy(slot.cloned[0..prefix.len], prefix);
            @memcpy(slot.cloned[prefix.len..], r.frames);
            rows = slot.cloned;
        }
        const layout = try Layout.of(rows);
        var replaced: [1 + cfg.conditions]z3.Replace = undefined;
        var n: usize = 0;
        if (r.speaker.len != 0) {
            replaced[n].row = 0;
            h.e.w.speakerRow(r.speaker[0..cfg.speaker_dim], &replaced[n].vec);
            n += 1;
        }
        // a condition the request leaves out is unknown: the last row of its table
        var unknown: [cfg.conditions]z3.Replace = undefined;
        for (0..cfg.conditions, &unknown) |c, *u| {
            const bucket = if (r.conditions.len != 0) r.conditions[c] else h.e.c.buckets[c];
            replaced[n].row = layout.condition_row + c;
            @memcpy(&replaced[n].vec, h.e.w.conditions[c][bucket * cfg.dim ..][0..cfg.dim]);
            n += 1;
            u.row = c;
            @memcpy(&u.vec, h.e.w.conditions[c][h.e.c.buckets[c] * cfg.dim ..][0..cfg.dim]);
        }
        // a speaker twin: the prompt without its voice, every condition unknown; a controls twin: the prompt, unsteered
        const controls_scale: f32 = if (r.steering) |st| st.guidance else 1;
        const twin: []const [W]i32 = if (r.cfg_gamma != 1) rows[layout.condition_row..] else if (controls_scale != 1) rows else &.{};
        const twin_replaced: []const z3.Replace = if (r.cfg_gamma != 1) &unknown else replaced[0..n];
        const scale = if (r.cfg_gamma != 1) r.cfg_gamma else controls_scale;
        slot.twin_steered = r.cfg_gamma != 1;
        if (r.steering) |st| {
            const width = st.sources.len * cfg.dim;
            slot.delta = try gpa.alloc(f32, 2 * width);
            slot.steer = try gpa.alloc(z3.Steer, st.sources.len);
            for (slot.steer, st.sources, 0..) |*s, source, j| s.* = .{ .source = source, .delta = slot.delta[j * cfg.dim ..][0..cfg.dim] };
            slot.spans = try gpa.alloc(z3.steering.Span, st.spans.len);
            for (slot.spans, st.spans) |*to, from| to.* = .{ .start = from.start, .end = from.end, .ramp = from.ramp, .delta = from.delta };
            slot.steering = .{ .sources = st.sources, .constant = st.constant, .released = st.released, .release = .{ .onset = st.onset, .ramp = st.ramp, .hold = st.hold }, .spans = slot.spans };
        }
        if (twin.len != 0) {
            if (slot.twin == null) {
                if (h.admit) |a| if (twinBytes(h.e.o) > try a.room(a.ptr, &h.e.ctx)) return error.GuidanceNeedsMoreDeviceMemory;
                slot.twin = try h.e.addLane();
            }
            h.e.use(slot.twin.?);
            try h.e.reset();
            h.e.use(slot.index);
        }
        const sampling: z3.Sampling = .{ .temperature = r.temperature, .top_k = r.top_k, .top_p = r.top_p, .seed = r.seed };
        try h.e.reset();
        try h.codec.reset();
        slot.audio = .{ .h = h, .job = job, .prefill = rows[rows.len - layout.prefill ..] };
        if (r.laci) |lc| {
            slot.heads = try gpa.alloc(z3.laci.Head, lc.heads.len);
            for (slot.heads, lc.heads) |*to, from| to.* = .{ .layer = from.layer, .head = from.head, .radius = from.radius };
            slot.kept = replaced;
            const text = layout.condition_row + cfg.conditions;
            if (lc.short) |sh| {
                slot.short = .{ .pause_marker = "", .max_words = 0, .max_ratio = .{ 0, 0 }, .lookback = sh.lookback, .min_voiced = sh.min_voiced, .max_frames = sh.max_frames, .max_retries = sh.max_retries, .threshold = sh.threshold, .weights = sh.weights, .fillers = &.{} };
            }
            const cut: ?z3.short_text.Cut = if (lc.short) |sh| z3.short_text.Cut.init(gpa, &slot.short, sh.filler_token) else null;
            const setup: z3.laci_run.Setup = .{ .short = cut, .heads = .{ .heads = slot.heads, .frames_per_token = lc.frames_per_token }, .text_start = text, .text_len = rows.len - layout.prefill - text, .prefill = layout.prefill, .hold = !r.stream };
            slot.laci = try LaciRun.begin(gpa, Guarded{ .h = h, .slot = slot }, rows, slot.kept[0..n], setup, sampling, r.max_frames);
            if (r.steering != null) slot.laci.?.steer = .{ .ctx = slot, .at = Slot.steerFor, .twin = false };
        } else slot.run = try z3.generation.Run.begin(gpa, Guarded{ .h = h, .slot = slot }, rows, @as([]const z3.Replace, replaced[0..n]), twin, twin_replaced, scale, sampling, r.max_frames);
        slot.started = true;
    }

    fn finish(h: *Host, slot: *Slot, job: Job, reason: api.Reason, message: []const u8) void {
        if (slot.laci) |*lr| lr.deinit() else if (slot.started) slot.run.deinit();
        slot.laci = null;
        h.gpa.free(slot.heads);
        slot.heads = &.{};
        h.gpa.free(slot.cloned);
        h.gpa.free(slot.delta);
        h.gpa.free(slot.steer);
        h.gpa.free(slot.spans);
        slot.spans = &.{};
        slot.delta = &.{};
        slot.steer = &.{};
        slot.steering = .{};
        h.lock();
        if (reason == .stop or reason == .length) job.request.result.samples = slot.audio.samples;
        job.request.result.reason = reason;
        slot.cloned = &.{};
        slot.started = false;
        slot.job = null;
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
        h.gpa.free(h.turns);
        h.gpa.free(h.slots);
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
    if (o.parallel == 0 or o.parallel > max_parallel) return error.BadZonosParallel;
    var door: Door = .{ .o = o, .need = need(o.context, b.frames) + (o.parallel - 1) * laneBytes(o.context, b.frames) };
    const e = try z3.Engine.create(gpa, io, rel.lm, .{ .device = o.device, .max_ctx = o.context, .gate = .{ .ptr = &door, .check = Door.check } });
    errdefer e.destroy();
    const codec = try z3.Codec.create(gpa, io, &e.driver, e.stream, rel.generator, rel.head, .{ .max_frames = b.frames });
    errdefer codec.destroy();
    for (1..o.parallel) |_| {
        _ = try e.addLane();
        _ = try codec.addLane();
    }
    if (cuda.usage(false).device > door.need) return error.LoadOverItsAdmission;
    const slots = try gpa.alloc(Slot, o.parallel);
    errdefer gpa.free(slots);
    for (slots, 0..) |*slot, i| slot.* = .{ .index = i };
    const turns = try gpa.alloc(z3.Turn, 2 * o.parallel);
    errdefer gpa.free(turns);
    const kept = try gpa.dupe(u8, dir);
    errdefer gpa.free(kept);
    h.* = .{
        .gpa = gpa,
        .io = io,
        .dir = kept,
        .rel = rel,
        .e = e,
        .codec = codec,
        .slots = slots,
        .turns = turns,
        .admit = o.admit,
        .info_ = .{
            .name = "ZONOS3 Mini",
            .text_generation = false,
            .lanes = o.parallel,
            .context_window = o.context,
            .plain_only = true,
            .prompt_cache = false,
            .speech = .{ .sample_rate = z3.codec.sample_rate, .frame_length = z3.codec.frame_samples, .max_frames = b.frames, .max_prompt_tokens = b.prompt },
        },
    };
    h.worker = try std.Thread.spawn(.{}, Host.run, .{h});
    const held = cuda.usage(false).device;
    std.log.info("zonos3: decoder {s}, {d} frames a request, {d} at once, {d} bytes on the device ({d} admitted)", .{ h.rel.decoder, b.frames, o.parallel, held, door.need });
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}

test {
    _ = @import("zonos_layout.zig");
}

test "need: the weights, then state that grows with the context and with the frame budget" {
    const t = std.testing;
    const base = need(4096, 2000);
    try t.expectEqual(@as(u64, 3_866_946_048), base);
    try t.expect(need(8192, 2000) > base and need(4096, 1000) < base);
    // two attention layers keep a key row and a value row a position, and a masked head's scratch one of each
    try t.expectEqual(@as(u64, (2 * 2 + 2) * 4096 * cfg.kv_width * 4), need(8192, 2000) - base);
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

test "laneBytes: each request at once past the first adds a backbone lane and a decoder lane, no weights" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 1_236_862_976), laneBytes(4096, 2000));
    try t.expectEqual(@as(u64, 10_913_792), laneBytes(4096, 2000) - z3.Codec.laneBytes(.{ .max_frames = 2000 }));
    try t.expect(laneBytes(4096, 1000) < laneBytes(4096, 2000));
}

test "twinBytes: a guided request's second stream is the caches and recurrent state again, no weights" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 10_913_792), twinBytes(.{ .max_ctx = 4096 }));
    try t.expectEqual(@as(u64, 2 * 2 * 4096 * cfg.kv_width * 4), twinBytes(.{ .max_ctx = 8192 }) - twinBytes(.{ .max_ctx = 4096 }));
}
