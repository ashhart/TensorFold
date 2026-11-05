//! MOSS-Image's serving host on CUDA: the encoder, transformer and vae decoder resident, one picture at a time.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const mi = @import("moss_image");
const tok_mod = @import("tokenizer");
const spec = @import("moss_image_spec.zig");
const Allocator = std.mem.Allocator;
const png = mi.png;

/// The card's negative branch: block 22 is skipped there, the prompt's pass running every block (moss_image_run.zig).
const negative_skip: usize = 22;
/// The transformer plans its scratch for the largest geometry the API allows: 1,572,864 pixels is 96 x 64 latent rows.
const max_width: usize = 1536;
const max_height: usize = 1024;
/// The publisher's chat template around a prompt's ids: the adapter this host adds to the encoder's input.
const template_before = "<|im_start|>user\n";
const template_after = "<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// The host's arrivals: `depth` requests held at once, one picture running, served first come first served.
pub const Queue = struct {
    /// One picture runs and one waits; a third request is refused with 503 until a slot frees.
    pub const depth: usize = 2;
    slots: [depth]?api.Id = @splat(null),
    /// The waiting slots in arrival order, the oldest at `head`.
    order: [depth]u8 = @splat(0),
    head: usize = 0,
    len: usize = 0,

    /// The slot `id` takes, or null when `depth` requests are already held.
    pub fn admit(q: *Queue, id: api.Id) ?usize {
        for (&q.slots, 0..) |*slot, i| if (slot.* == null) {
            slot.* = id;
            q.order[(q.head + q.len) % depth] = @intCast(i);
            q.len += 1;
            return i;
        };
        return null;
    }

    /// The oldest waiting slot: its picture begins, and the slot stays held until it finishes.
    pub fn next(q: *Queue) ?usize {
        if (q.len == 0) return null;
        const slot = q.order[q.head];
        q.head = (q.head + 1) % depth;
        q.len -= 1;
        return slot;
    }

    /// The slot `id` holds, waiting or running.
    pub fn find(q: *const Queue, id: api.Id) ?usize {
        for (q.slots, 0..) |slot, i| if (slot) |waiting| if (waiting == id) return i;
        return null;
    }

    /// The slot's request is over: the next arrival may take it.
    pub fn release(q: *Queue, slot: usize) void {
        q.slots[slot] = null;
    }

    /// How many requests the host holds now.
    pub fn held(q: *const Queue) usize {
        var n: usize = 0;
        for (q.slots) |slot| {
            if (slot != null) n += 1;
        }
        return n;
    }
};

/// A request's latent grid: rows (height / 16) by columns (width / 16).
const Grid = struct { rows: usize, columns: usize };

const Phase = enum { prompt, denoise, decode };

/// One held picture and everything its pass allocates; `free` returns every device and host buffer it holds.
const Job = struct {
    id: api.Id,
    request: *const api.ImageRequest,
    sink: api.Sink,
    phase: Phase = .prompt,
    step: u32 = 0,
    prompt: ?mi.Prompt = null,
    negative: ?mi.Prompt = null,
    schedule: ?mi.schedule.Schedule = null,
    x: []f32 = &.{},
    cond: []f32 = &.{},
    uncond: []f32 = &.{},
    /// The decoder's raw picture [height, width, 3] and its 8-bit rows, which the PNG carries.
    picture: []f32 = &.{},
    rgb: []u8 = &.{},
    /// The prompt encoder's rows per branch, freed once the refiner has taken them.
    embeds: [2][]u16 = .{ &.{}, &.{} },

    fn free(j: *Job, gpa: Allocator) void {
        if (j.prompt) |*p| p.deinit();
        if (j.negative) |*p| p.deinit();
        if (j.schedule) |s| s.deinit(gpa);
        for (j.embeds) |rows| if (rows.len > 0) gpa.free(rows);
        for ([_][]f32{ j.x, j.cond, j.uncond, j.picture }) |buf| if (buf.len > 0) gpa.free(buf);
        if (j.rgb.len > 0) gpa.free(j.rgb);
    }
};

/// The checkpoint's three models and the vae, resident on one device, behind the engine API's image hook.
pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    encoder: *mi.Encoder,
    model: *mi.Model,
    decoder: *mi.Decoder,
    tokenizer: tok_mod.Tokenizer,
    /// The template's ids around the prompt, and the empty negative prompt's rows, fixed once the checkpoint opens.
    before: []u32,
    after: []u32,
    empty: []u16,
    /// The checkpoint's scheduler settings (scheduler/scheduler_config.json).
    scheduler: mi.schedule.Config,
    startup: []u8 = &.{},
    info_: api.Info = .{},
    queue: Queue = .{},
    jobs: [Queue.depth]?Job = @splat(null),
    cancelled: [Queue.depth]bool = @splat(false),
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    worker: ?std.Thread = null,
    closing: bool = false,
    /// A device failure: later requests fail with it rather than starting on a broken device.
    broken: bool = false,
    const vtable: api.Engine.VTable = .{
        .info = info,
        .submit = submitText,
        .submit_image = submitImage,
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

    /// Every limit checked before a slot is taken, so a refused request holds nothing.
    fn valid(h: *const Host, r: *const api.ImageRequest) bool {
        const cap = h.info_.image.?;
        if (r.width == 0 or r.height == 0) return false;
        if (r.width % mi.config.pixels_per_row != 0 or r.height % mi.config.pixels_per_row != 0) return false;
        if (r.width > cap.max_side or r.height > cap.max_side) return false;
        if (@as(u32, r.width) * @as(u32, r.height) > cap.max_pixels) return false;
        if (r.steps < 1 or r.steps > cap.max_steps) return false;
        if (!std.math.isFinite(r.guidance) or r.guidance < 1) return false;
        if (r.prompt.len == 0 or r.prompt.len > cap.max_prompt_tokens) return false;
        if (r.negative.len > cap.max_prompt_tokens) return false;
        // the caller's list is empty until this host appends its one PNG: a picture already there is not one request
        return r.png.items.len == 0;
    }

    fn submitImage(p: *anyopaque, id: api.Id, r: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const h = self(p);
        if (!h.valid(r)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing or h.broken) return error.Closed;
        if (h.queue.find(id) != null) return error.InvalidRequest;
        const slot = h.queue.admit(id) orelse return error.Busy;
        r.result.* = .{};
        h.jobs[slot] = .{ .id = id, .request = r, .sink = sink };
        h.cancelled[slot] = false;
        h.wake.signal(h.io);
    }

    /// Ends a held request: a waiting one is not begun, a running one stops between its stages.
    fn cancel(p: *anyopaque, id: api.Id) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        if (h.queue.find(id)) |slot| h.cancelled[slot] = true;
        h.wake.signal(h.io);
    }

    fn status(p: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        out.* = .{};
        out.running = @intCast(h.queue.held());
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const used = cuda.usage(reset_peak);
        return .{ .active = used.device, .pinned = used.host, .peak = used.peak };
    }

    fn run(h: *Host) void {
        // CUDA binds its context per thread: the host's is made current here, before any buffer or launch
        h.ctx.makeCurrent() catch |e| {
            h.lock();
            h.broken = true;
            h.unlock();
            std.log.err("moss image: the CUDA context is not current on its worker ({s})", .{@errorName(e)});
        };
        while (true) {
            h.lock();
            const slot = h.queue.next() orelse {
                if (h.closing) {
                    h.unlock();
                    return;
                }
                h.wake.waitUncancelable(h.io, &h.mutex);
                h.unlock();
                continue;
            };
            const stop = h.cancelled[slot] or h.closing or h.broken;
            const broken = h.broken;
            h.unlock();
            if (stop) {
                h.finish(slot, if (broken) .failed else .cancelled, if (broken) "the picture engine failed earlier" else "");
                continue;
            }
            while (true) {
                const more = h.advance(slot) catch |e| {
                    h.fail(slot, e);
                    break;
                };
                if (!more) {
                    h.finish(slot, .stop, "");
                    break;
                }
                h.lock();
                const now = h.cancelled[slot] or h.closing or h.broken;
                h.unlock();
                if (now) {
                    h.finish(slot, .cancelled, "");
                    break;
                }
            }
        }
    }

    /// One stage of a picture: prompts, one denoising step, the decode; true while the request still has work.
    fn advance(h: *Host, slot: usize) !bool {
        const j = &h.jobs[slot].?;
        switch (j.phase) {
            .prompt => {
                try h.refine(j);
                j.phase = .denoise;
                return true;
            },
            .denoise => {
                const schedule = &j.schedule.?;
                const at = j.step;
                _ = try h.model.forward(&j.prompt.?, j.x, schedule.timesteps[at], null, j.cond);
                const by = schedule.sigmas[at + 1] - schedule.sigmas[at];
                if (j.request.guidance == 1) {
                    // guidance 1 is the prompt's own velocity: the negative pass would change nothing
                    for (j.x, j.cond) |*x, cond| x.* += cond * by;
                } else {
                    _ = try h.model.forward(&j.negative.?, j.x, schedule.timesteps[at], negative_skip, j.uncond);
                    for (j.x, j.cond, j.uncond) |*x, cond, uncond| x.* += (uncond + j.request.guidance * (cond - uncond)) * by;
                }
                j.step += 1;
                // the last step is consumed: the picture is painted next, so the pass goes on into the decode
                if (j.step >= j.request.steps) j.phase = .decode;
                return true;
            },
            .decode => {
                try h.paint(j);
                return false;
            },
        }
    }

    /// The prompt phase: the request's ids through the encoder and the refiner, its noise, and its schedule.
    fn refine(h: *Host, j: *Job) !void {
        const r = j.request;
        const grid = try mi.config.grid(r.width, r.height);
        try h.model.fit(r.width, r.height);
        const ids = try bracket(h.gpa, h.before, r.prompt, h.after);
        defer h.gpa.free(ids);
        const negative = if (r.negative.len > 0) try bracket(h.gpa, h.before, r.negative, h.after) else null;
        defer if (negative) |n| h.gpa.free(n);
        j.embeds[0] = try h.encode(ids);
        defer gpaFree(h.gpa, &j.embeds[0]);
        j.embeds[1] = if (negative) |n| try h.encode(n) else try h.gpa.dupe(u16, h.empty);
        defer gpaFree(h.gpa, &j.embeds[1]);
        j.prompt = try h.model.prompt(j.embeds[0]);
        j.negative = try h.model.prompt(j.embeds[1]);
        j.schedule = try mi.schedule.timesteps(h.gpa, h.scheduler, r.steps);
        const count = grid.rows * grid.columns * mi.config.channels;
        j.x = try h.gpa.alloc(f32, count);
        try mi.noise.latents(h.gpa, r.seed, grid.rows, grid.columns, j.x);
        j.cond = try h.gpa.alloc(f32, count);
        if (r.guidance != 1) j.uncond = try h.gpa.alloc(f32, count);
        j.rgb = try h.gpa.alloc(u8, @as(usize, r.width) * @as(usize, r.height) * 3);
        j.picture = try h.gpa.alloc(f32, j.rgb.len);
    }

    /// The request's ids through the prompt encoder: its final-norm rows [tokens, 4096] bf16, caller frees.
    fn encode(h: *Host, ids: []const u32) ![]u16 {
        const rows = try h.gpa.alloc(u16, ids.len * mi.encoder_config.hidden);
        errdefer h.gpa.free(rows);
        try h.encoder.rows(ids, rows, null);
        return rows;
    }

    /// The decode stage: the latents to a picture, 8-bit as the pipeline clamps it, and its PNG in the caller's list.
    fn paint(h: *Host, j: *Job) !void {
        const r = j.request;
        _ = try h.decoder.decode(j.x, r.width, r.height, j.picture);
        quantize(j.rgb, j.picture);
        const bytes = try png.encode(r.alloc, j.rgb, r.width, r.height);
        defer r.alloc.free(bytes);
        try r.png.appendSlice(r.alloc, bytes);
        r.result.bytes = r.png.items.len;
    }

    /// The one finished event: the request's buffers go first, so the caller may free its own as the event lands.
    fn finish(h: *Host, slot: usize, reason: api.Reason, message: []const u8) void {
        h.lock();
        const job = h.jobs[slot].?;
        h.jobs[slot] = null;
        h.cancelled[slot] = false;
        h.queue.release(slot);
        h.unlock();
        var held = job;
        held.free(h.gpa);
        job.request.result.reason = reason;
        job.sink.event(job.sink.ctx, job.id, &.{ .finished = .{ .reason = reason, .message = message } });
    }

    /// A stage's error: out of memory fails this picture alone, any other failure takes the engine out of service.
    fn fail(h: *Host, slot: usize, e: anyerror) void {
        if (e != error.OutOfMemory) {
            h.lock();
            h.broken = true;
            h.unlock();
            std.log.err("moss image: a picture's pass failed ({s}); this engine serves no more", .{@errorName(e)});
        }
        h.finish(slot, .failed, @errorName(e));
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

    pub fn close(p: *anyopaque) void {
        const h = self(p);
        drain(h);
        // every held request's finished event went out before the worker returned: nothing here points at a caller
        h.model.deinit();
        h.decoder.deinit();
        h.encoder.deinit();
        h.tokenizer.deinit();
        h.gpa.free(h.before);
        h.gpa.free(h.after);
        h.gpa.free(h.empty);
        if (h.startup.len > 0) h.gpa.free(h.startup);
        h.ctx.deinit();
        h.driver.close();
        h.gpa.destroy(h);
    }
};

/// The publisher's chat template around a request's ids, cut at the encoder's longest prompt as the pipeline cuts it.
pub fn bracket(a: Allocator, before: []const u32, ids: []const u32, after: []const u32) ![]u32 {
    const total = @min(before.len + ids.len + after.len, mi.encoder_config.max_rows);
    const out = try a.alloc(u32, total);
    var at: usize = 0;
    for ([_][]const u32{ before, ids, after }) |part| {
        const take = @min(part.len, total - at);
        @memcpy(out[at..][0..take], part[0..take]);
        at += take;
        if (at == total) break;
    }
    return out;
}

/// The pipeline's 8-bit rule: clamp to [-1, 1], rescale to [0, 1], * 255, then round with ties to even.
pub fn quantize(out: []u8, pixels: []const f32) void {
    for (out, pixels) |*o, v| {
        const unit = std.math.clamp(v, -1, 1) * 0.5 + 0.5;
        o.* = roundEven(std.math.clamp(unit, 0, 1) * 255);
    }
}

/// Metal's `rint` for a value from 0 through 255: the nearest whole number, ties to even.
pub fn roundEven(x: f32) u8 {
    const whole = @trunc(x);
    const at: u32 = @intFromFloat(whole);
    const frac = x - whole;
    const up: u32 = if (frac > 0.5 or (frac == 0.5 and at % 2 == 1)) 1 else 0;
    return @intCast(@min(at + up, 255));
}

fn gpaFree(gpa: Allocator, rows: *[]u16) void {
    if (rows.len == 0) return;
    gpa.free(rows.*);
    rows.* = &.{};
}

fn read(a: Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, name }), a, .limited(1 << 20));
}

/// The engine for a MOSS-Image checkpoint in `o.dir`, or null with `problem` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (spec.refusal(o)) |why| {
        problem.* = why;
        return null;
    }
    return openCheckpoint(gpa, io, o.dir) catch |e| {
        problem.* = switch (e) {
            error.UncheckedGpu => "MOSS-Image runs on sm_120 (an RTX 5090) and sm_121 (a GB10) only, the two GPUs this family was checked on",
            error.FileNotFound => try std.fmt.allocPrint(a, "MOSS-Image needs the checkpoint's own folders under {s}: transformer/, text_encoder/, tokenizer/ (tokenizer.json), vae/ and scheduler/", .{o.dir}),
            error.UnsupportedScheduler => "the checkpoint's scheduler turns on an option this family does not implement (the log line names it)",
            error.UnsupportedGeometry, error.NotQwen35, error.NotMossImage => "the checkpoint is not the released MOSS-Image geometry this family runs (the log line names the setting)",
            else => try std.fmt.allocPrint(a, "the native CUDA engine cannot load the MOSS-Image checkpoint ({s})", .{@errorName(e)}),
        };
        return null;
    };
}

fn openCheckpoint(gpa: Allocator, io: std.Io, dir: []const u8) !api.Opened {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const scheduler = try mi.schedule.parse(a, try read(a, io, dir, "scheduler/scheduler_config.json"));
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa,
        .io = io,
        .driver = undefined,
        .ctx = undefined,
        .encoder = undefined,
        .model = undefined,
        .decoder = undefined,
        .tokenizer = undefined,
        .before = &.{},
        .after = &.{},
        .empty = &.{},
        .scheduler = scheduler,
    };
    h.driver = try cuda.Driver.open();
    errdefer h.driver.close();
    h.ctx = try cuda.Context.init(&h.driver, 0);
    errdefer h.ctx.deinit();
    const sm = try h.ctx.capability();
    if (sm != 120 and sm != 121) return error.UncheckedGpu;
    // the prompt encoder keeps its own context and stream over the same primary context, as its tool does
    h.encoder = try mi.Encoder.load(gpa, io, dir);
    errdefer h.encoder.deinit();
    // the transformer's scratch is planned once for the largest geometry the API allows: a request fits inside it
    h.model = try mi.Model.load(gpa, io, &h.ctx, try std.fs.path.join(a, &.{ dir, "transformer" }), mi.encoder_config.max_rows, max_width, max_height);
    errdefer h.model.deinit();
    h.decoder = try mi.Decoder.load(gpa, io, try std.fs.path.join(a, &.{ dir, "vae" }));
    errdefer h.decoder.deinit();
    h.tokenizer = try tok_mod.loadTokenizer(io, gpa, try std.fs.path.join(a, &.{ dir, "tokenizer", "tokenizer.json" }));
    errdefer h.tokenizer.deinit();
    h.before = try h.tokenizer.encode(gpa, template_before);
    errdefer gpa.free(h.before);
    h.after = try h.tokenizer.encode(gpa, template_after);
    errdefer gpa.free(h.after);
    const none = try bracket(gpa, h.before, &.{}, h.after);
    defer gpa.free(none);
    h.empty = try h.encode(none);
    errdefer gpa.free(h.empty);
    var name_buf: [256]u8 = undefined;
    const device = h.ctx.name(&name_buf) catch "a CUDA GPU";
    h.startup = try std.fmt.allocPrint(gpa, "{s}: MOSS-Image's prompt encoder, transformer and decoder resident; one picture at a time, {d} requests held; up to {d} pixels a request", .{ device, Queue.depth, spec.max_pixels });
    errdefer gpa.free(h.startup);
    h.info_ = .{
        .name = "MOSS-Image",
        .text_generation = false,
        .lanes = 1,
        .plain_only = true,
        .image = spec.limits(),
        .startup = h.startup,
    };
    h.worker = try std.Thread.spawn(.{}, Host.run, .{h});
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}
