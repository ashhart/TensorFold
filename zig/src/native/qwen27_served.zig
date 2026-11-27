//! Qwen3.8-27B on the lane host: DFlash2 drafts up to TF_Q27_DRAFT_SLOTS admitted streams, when loaded.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const pc = api.prompt_cache;
const cache_fit = @import("cache_fit.zig");
const share_fit = @import("qwen27_share_fit.zig");
const slot_state = @import("qwen27_slots.zig");
const Resident = @import("resident.zig").Resident;
const lb = @import("qwen27_lanes.zig");
pub const lanes_backend = lb;
const startup = @import("qwen27_startup.zig");
const reserve = @import("qwen27_reserve.zig");
const StartupLine = startup.StartupLine;
const disk = @import("qwen27_disk.zig");
const images = @import("qwen27_images.zig");
const Mtp = @import("qwen27_mtp.zig").Mtp;
const steer_served = @import("qwen27_steer_served.zig");
const Allocator = std.mem.Allocator;
const helpers = @import("qwen27_served_helpers.zig");
const mem_ctrl = @import("memory_controller.zig");
const qwen27_mem = @import("qwen27_memory.zig");
const room = helpers.room;
const drafterRefusal = helpers.drafterRefusal;
const gibBytes = helpers.gibBytes;

pub const Options = struct {
    context: u32,
    streams: u32, // --parallel: slots at most, fewer when their caches do not fit
    streams_fixed: bool = false, // --parallel named a number: every slot fits, or the server refuses
    drafts: bool = true,
    drafter: ?[]const u8 = null,
    drafter_name: ?[]const u8 = null,
    drafter_bits: u8 = 0,
    mtp_drafts: ?u32 = null, // --mtp-drafts N: the MTP head beside the checkpoint drafts every stream's chains (0: off)
    mtp_confidence: ?f64 = null, // --mtp-confidence: a chain stops where its cumulative chance falls below it
    mtp_dir: ?[]const u8 = null, // SenseNova-Loop: the checkpoint directory its own mtp.safetensors is read from
    cache_gib: ?f64 = null,
    cache_over_cap: bool = false,
    decode_share: ?f64 = null, // --decode-share: the reply's share of a fill step's wall time beside live decoders
    learn: ?disk.Learn = null, // --learn-conversations: kept states written behind to disk and read back
    vision: ?api.VisionOpen = null, // --vision: open loads the image tower, and requests may hold images
    images: ?*images.Vision = null, // the loaded tower attach serves (and the host then owns)
    rope_yarn: ?f64 = null,
    glp: ?api.glp.Glp = null, // the control vector the model steers with (null: none)
    slide: bool = false, // --slide: Folded Memory Weights v3 live learning
    dir: []const u8 = "", // checkpoint directory for living-weights.safetensors
};

pub const Host = struct {
    gpa: Allocator,
    counted: mtl.pass_metrics.Counting,
    model: *q.model.Model,
    owns_model: bool,
    runner: q.decode_round.Runner,
    trace: ?tf.gpu_profile.Trace = null, // TF_Q27_PROFILE: rounds' class GPU times to the log, gpu_seconds in replies
    draft: ?*lb.Draft = null,
    sn_mtp: ?*q.sn_mtp.Mtp = null, // SenseNova-Loop's own MTP head (mtp.safetensors), the core's drafter
    back: *lb.Metal,
    mtp: ?*Mtp = null, // --mtp-drafts: the MTP head, the core's drafter in front of `back`
    wrap: ?lanes.drafted.Drafted = null,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?pc.Store = null,
    resident: ?Resident = null,
    warm: mtl.keepalive.Target,
    disk_stage: ?mtl.Buffer = null, // copies out of private kept states for the disk (qwen27_disk.zig)
    vision: ?*images.Vision = null, // --vision's tower and image rows
    spares: q.snapshot.Spares = .{}, // dropped kept states' buffers, for the next saves
    folded_mem: ?*mem_ctrl.Holder = null, // Folded Memory Weights v3 controller

    pub fn engine(h: *Host) api.Engine {
        return h.host.engine();
    }

    fn of(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }

    fn lone(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        return of(ptr).back.lone(s, hooks);
    }

    /// A slot's first use: whether its memory fits beside everything taken and the margin, kept states freed first.
    fn roomFor(ptr: *anyopaque, bytes: usize, s: *const lanes.Stream) bool {
        const h = of(ptr);
        return takeRoom(if (h.cache) |*c| c else null, room(h.model) -| h.runner.gdn.missingBeforeBytes(), bytes, s.reuse.saved, s.prompt());
    }

    /// A new slot's buffers join the residency set, so the keepalive holds them wired with the rest.
    fn added(ptr: *anyopaque, buffers: []const mtl.Buffer) void {
        const h = of(ptr);
        if (h.resident) |*r| r.addNow(buffers); // its first round reads them: committed now, as before staging
        std.log.info("Qwen3.8-27B: a slot took its memory ({d} of {d} ready)", .{ h.runner.ready, h.runner.slots });
    }

    /// A finished reply's state stands where a prompt pass of prompt and reply would: keep it for the next turn.
    fn keepReply(ptr: *anyopaque, s: *lanes.Stream, slot: u32) void {
        const h = of(ptr);
        if (h.cache) |*store| keepEnd(store, h.gpa, s, h.runner.offsets[slot]);
    }

    /// A fallback never displaces a kept state or spends the room promised to this pass's later marks.
    fn checkpoint(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const store = if (of(ptr).cache) |*c| c else return;
        const key = if (s.key.len > 0) s.key else s.prompt();
        if (@import("qwen27_resume.zig").checkpointFits(store, key, at, s.reuse.marks, s.decode_spans)) {
            if (s.reuse.hook) |k| k.at(k.ptr, s, at);
        }
    }

    /// Info.vision's prepare, on the HTTP thread.
    fn prepareImage(ptr: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const h = of(ptr);
        return images.prepare(h.vision.?, h.clock.io, a, bytes, max_tokens);
    }
};

/// The reply-end snapshot, retaining a matching pre-reply cache state too when both fit.
pub fn keepEnd(store: *pc.Store, gpa: Allocator, s: *lanes.Stream, stands: u32) void {
    const prompt: u32 = @intCast(s.prompt_len);
    const tokens = s.context.items;
    if (tokens.len == prompt or stands + 1 < tokens.len or stands > tokens.len) return;
    const spans = api.cache_modes.reply(gpa, s.decode_spans, prompt, stands) catch return;
    defer gpa.free(spans);
    // an image prompt's rows go by their images (the request's cache key), as its lookups and marks do
    const keyed = if (s.key.len == prompt) gpa.dupe(u32, tokens) catch return else tokens;
    defer if (keyed.ptr != tokens.ptr) gpa.free(keyed);
    if (keyed.ptr != tokens.ptr) @memcpy(keyed[0..prompt], s.key);
    // Keep the history too when both copies fit; otherwise the reply end keeps its ordinary priority.
    const history = if (s.history_len > 0) store.find(keyed[0..@min(keyed.len, @as(usize, s.history_len) + 1)], s.chunks, spans) else null;
    const pin = if (history) |e| if (@import("qwen27_resume.zig").pinReplyHistory(store, e, s.history_len, stands)) e else null else null;
    if (pin) |e| e.borrowers += 1;
    defer if (pin) |e| {
        e.borrowers -= 1;
    };
    _ = store.keepBorn(keyed, prompt, stands, s, s.chunks, spans);
}

/// The host's memory counts: `cache_fit.hostMemory` from the process footprint (no configured load limit).
fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    return cache_fit.hostMemory(0, reset_peak);
}

/// The words for a request the engine refuses.
fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and max_tokens do not fit this server's --context",
        else => null,
    };
}

/// The prompt cache's copies of a stream's slot (snapshot.zig): its DeltaNet states, K/V rows and taps ring.
const Snaps = struct {
    fn host(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h = host(ptr);
        return q.snapshot.Spares.size(q.snapshot.slotBytes(&h.runner, h.back.ringOf(0), at)); // a new buffer's
    }
    fn charged(_: *anyopaque, saved: pc.Saved) u64 {
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
        return if (st.spares != null) st.cap else st.bytes;
    }
    fn spare(ptr: *anyopaque) u64 {
        return host(ptr).spares.spare();
    }
    fn trim(ptr: *anyopaque, left: u64) void {
        host(ptr).spares.trim(left);
    }
    fn reuses(ptr: *anyopaque, at: u32) bool {
        const h = host(ptr);
        return h.spares.fits(q.snapshot.slotBytes(&h.runner, h.back.ringOf(0), at));
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const h = host(ptr);
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const slot = h.back.slotOf(s) orelse return error.UnknownStream;
        // Scratch already occupies its bytes; do not let cache growth spend that same free room again.
        if (h.runner.gdn.marks != null) {
            const n = q.snapshot.slotBytes(&h.runner, h.back.ringOf(slot), at);
            const need: usize = if (h.spares.fits(n)) 0 else q.snapshot.Spares.size(n);
            if (need > room(h.model) -| h.runner.gdn.missingBeforeBytes()) return error.LaneMemory;
        }
        if (h.draft) |d| try d.generation.finishKeep();
        h.spares.follow = h.runner.residency;
        const st = try q.snapshot.saveSlot(h.gpa, &h.runner, slot, h.back.ringOf(slot), at, &h.spares);
        return @ptrCast(st);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.RestoredInPrefill; // the backend restores a stream's kept state in its prompt pass
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const h = host(ptr);
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
        q.snapshot.drop(h.gpa, st);
    }
};

/// The optional share's room after residency; unknown readings refuse it, and guards only reduce the room.
fn shareRoom(h: *const Host, why: *[]const u8) !u64 {
    const guards = share_fit.Guards.read() catch |err| {
        why.* = "TF_QWEN27_FIT_RAM_BYTES and TF_QWEN27_FIT_FOOTPRINT_BYTES must be positive byte counts";
        return err;
    };
    return share_fit.room(cache_fit.ram(), cache_fit.footprint(), try laterFootprint(h), guards);
}

/// TF_QWEN27_WORKING_SET_BYTES: a development cap on the device's reported working set.
const cap = reserve.cap;

/// Whether a new slot's `bytes` fit in `free`, once kept prompt states (never the one `restoring`) give theirs back.
pub fn takeRoom(cache: ?*pc.Store, free: usize, bytes: usize, restoring: ?pc.Saved, prompt: []const u32) bool {
    const held = if (cache) |c| c.held else 0;
    const ok = if (cache) |c| c.giveBack(restoring, prompt, Left{ .room = free, .store = c, .held = held, .bytes = bytes }) else free >= bytes;
    const freed = held - (if (cache) |c| c.held else 0);
    if (!builtin.is_test and freed > 0) std.log.info("Qwen3.8-27B: kept prompt states gave {d:.1} GiB back for a stream's slot", .{cache_fit.gibs(freed)});
    if (!builtin.is_test and !ok) std.log.info("Qwen3.8-27B: a slot needs {d:.1} GiB and {d:.1} GiB is free beside the weights and an 8 GiB margin: the request waits for a stream to end", .{ cache_fit.gibs(bytes), cache_fit.gibs(free + freed) });
    return ok;
}

/// A new slot's bytes against the room it found, plus the kept states' bytes freed since (their buffers' own sizes).
const Left = struct {
    room: usize,
    store: *const pc.Store,
    held: u64,
    bytes: usize,
    pub fn fits(l: Left) bool {
        return l.room + (l.held - l.store.held) >= l.bytes;
    }
};

/// Memory the slots past the ready ones take on first use: their K/V rows and taps rings.
fn laterBytes(h: *const Host) u64 {
    const ring_bytes: usize = if (h.draft) |d| @as(usize, d.model.ring().window) * d.model.ring().stride else if (h.sn_mtp) |m| m.slotBytes() else 0;
    return (h.runner.slots - h.runner.ready) * (h.runner.slotBytes() + ring_bytes) + h.runner.gdn.missingBeforeBytes();
}

pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, o: Options, a: Allocator, why: *[]const u8) !*Host {
    try @import("qwen27_prompt_floor.zig").environment(a, platform.environment.rawText("TF_PROMPT_ROWS"), why);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (o.drafter) |d| _ = q.dflash.config.admit(gpa, io, dir, d) catch |err| {
        why.* = try drafterRefusal(a, dir, d, err);
        return err;
    };
    const model = q.model.Model.loadDeferredOpts(gpa, io, dir, reserve.loaded_rows, .{ .rope_yarn = o.rope_yarn }) catch |err| {
        if (err == error.UnsupportedRopeScaling or err == error.RopeYarnFactor or err == error.RopeYarnConflict) {
            why.* = try std.fmt.allocPrint(a, "--rope-yarn / rope_parameters ({s}): YaRN requires valid factor and original_max_position_embeddings matching max_position_embeddings", .{@errorName(err)});
        }
        return err;
    };
    errdefer model.deinit();
    if (@import("qwen27_mtp_flags.zig").deviceRefusal(o.mtp_drafts, model.device.tensorUnits(), try q.decode_round.pagesOn())) |r| {
        why.* = r.why;
        return r.err;
    }
    if ((o.mtp_drafts orelse 0) > 0) model.loadMtp(io, dir) catch |err| {
        if (err == error.MissingMtpFile) why.* = try std.fmt.allocPrint(a, "--mtp-drafts: Qwen3.8-27B's MTP head is read from {s} beside the checkpoint, and {s} has none", .{ q.gpu_weights.mtp_file, dir });
        return err;
    };
    if (o.glp) |g| try steer_served.apply(gpa, model, dir, g, a, why);
    var with = o; // the tower before the slots: their room and the cache's budget count it as taken
    with.mtp_dir = dir;
    with.dir = dir;
    if (o.vision) |v| with.images = try images.open(gpa, io, model, dir, v, a, why);
    errdefer if (with.images) |v| v.deinit();
    const h = try attach(gpa, io, model, null, with, a, why);
    h.owns_model = true;
    return h;
}

/// A loaded model served as `open` serves one, `drafter` bound in place of `o.drafter`; both stay the caller's.
pub fn attach(parent: Allocator, io: std.Io, model: *q.model.Model, drafter: ?*q.dflash.runtime_model.Model, o: Options, a: Allocator, why: *[]const u8) !*Host {
    const setup = tf.startup_receipt.span(.runtime_setup);
    defer setup.end();
    try @import("qwen27_prompt_floor.zig").environment(a, platform.environment.rawText("TF_PROMPT_ROWS"), why);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const capacity = if (o.context == 0) @min(model.config.max_position, 32768) else o.context;
    if (capacity == 0 or capacity > model.config.max_position) return error.ContextFull;
    const policy = @import("qwen27_window.zig");
    const units = model.device.tensorUnits();
    const window = lb.windowRows(units) catch |err| {
        why.* = policy.problem(err) orelse why.*;
        return err;
    };
    const copies_value = platform.environment.rawText("TF_QWEN27_COPIES");
    _ = policy.copyPolicy(units, window, copies_value) catch |err| {
        why.* = policy.problem(err) orelse why.*;
        return err;
    };
    const lanes_value = platform.environment.rawText("TF_QWEN27_LANES");
    const serial = policy.serial(units, window, lanes_value) catch |err| {
        why.* = policy.problem(err) orelse why.*;
        return err;
    };
    const h = try parent.create(Host);
    errdefer parent.destroy(h);
    h.counted = .{ .parent = parent };
    const profiling = platform.c.getenv("TF_Q27_PASS_PROFILE") != null;
    const gpa = if (profiling) h.counted.allocator() else parent;
    var draft_model = drafter;
    const owns_draft = drafter == null and o.drafter != null;
    if (owns_draft) draft_model = q.dflash.runtime_model.Model.loadSpan(gpa, io, model, o.drafter.?, if (o.drafter_bits == 4) .prepared_q4_reference else .bf16_reference, capacity + window) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the native Qwen3.8-27B drafter cannot load {s} ({s})", .{ o.drafter.?, @errorName(err) });
        return error.DrafterLoad;
    };
    var draft_owned = owns_draft;
    errdefer if (draft_owned) draft_model.?.deinit();
    var sn_w = if (o.drafts and drafter == null and o.drafter == null and (o.mtp_drafts orelse 0) == 0) blk: {
        if (o.mtp_dir) |mdir| if (q.sn_mtp.Mtp.available(io, mdir, model.config)) {
            break :blk try q.sn_mtp.Mtp.loadWeights(gpa, io, model, mdir);
        };
        break :blk null;
    } else null;
    errdefer if (sn_w) |*mw| mw.deinit();
    const sn_on = sn_w != null;
    const memory = q.prompt_memory;
    const mtp_on = model.weights.mtp != null and (o.mtp_drafts orelse 0) > 0;
    const shared = try q.simd_attention.sharedPrefix(model.device.tensorUnits());
    // the MTP head's keys and values: one more attention layer a slot, beside the drafter's ring when there is one
    const head_kv: usize = if (mtp_on) @as(usize, capacity + window) * model.config.kvDim() * 2 * 2 else 0;
    const sn_ring: usize = if (sn_on) q.sn_mtp.Mtp.kvSlotBytes(model.config, capacity + window) else 0; // in a ring's place
    var fit_prompt = reserve.fit(model.config, capacity, window, o.streams, shared, head_kv + if (draft_model) |d| @as(usize, d.ring().window) * d.ring().stride else sn_ring, if (draft_model) |d| d.graph.config.tap_count else if (sn_on) 1 else q.taps.layers.len, model.frame.capacity, room(model));
    if (draft_model) |d| fit_prompt.tap_row = memory.tapRowBytes(model.config.hidden, d.graph.config.tap_count, d.graph.config.variant == .v1);
    const rows = try reserve.rows(fit_prompt, model.device.generation(), window);
    const slots = fit_prompt.streams(rows, o.streams_fixed) catch |err| {
        const dev = model.device;
        const ram: ?usize = if (cache_fit.ram()) |r| @intCast(r) else null;
        const tail = memory.advice(dev.maxWorkingSet(), dev.allocated(), cap(), try fit_prompt.cost(rows, 1), ram);
        why.* = try std.fmt.allocPrint(a, "--parallel {d} needs {d:.1} GiB for its streams and {d}-row prompt scratch at a {d}-token context; {d:.1} GiB remains beside the weights and an 8 GiB margin{f}", .{ o.streams, cache_fit.gibs(try fit_prompt.cost(rows, @max(o.streams, 1))), rows, capacity, cache_fit.gibs(fit_prompt.left), tail });
        return err;
    };
    if (model.frame.capacity != rows) {
        const next = try q.gpu_frame.Frame.init(model.device, model.config, rows);
        model.frame.deinit();
        model.frame = next;
    }
    std.log.info("qwen27: prompt capacity={d} rows, startup stream and scratch reserve={d} bytes", .{ rows, try memory.runnerBytesTaps(model.config, capacity + window, slots, rows, shared, fit_prompt.ring, if (fit_prompt.tap_row > 0) fit_prompt.tap_row else memory.tapRowBytes(model.config.hidden, fit_prompt.planes, false)) });
    h.gpa = gpa;
    h.model = model;
    h.owns_model = false;
    h.draft = null;
    h.mtp = null;
    h.sn_mtp = null;
    h.wrap = null;
    h.cache = null;
    h.resident = null;
    h.disk_stage = null;
    h.vision = o.images; // closed with the host; a failed attach leaves it to the caller
    h.trace = null;
    const paged = try q.decode_round.pagesOn();
    const kv_step = if (paged) null else q.kv_grow.step(@import("sys").environment.rawText("TF_Q27_KV_GROW")) catch |err| {
        why.* = "TF_Q27_KV_GROW must be 0 (off) or a row count that is a multiple of 16";
        return err;
    };
    h.runner = try if (paged) q.decode_round.Runner.initPaged(gpa, model, slots, capacity + window) else q.decode_round.Runner.initGrow(gpa, model, slots, capacity + window, 1, kv_step);
    if (kv_step) |s| std.log.info("qwen27: K/V grows as used: a slot starts at {d} rows and grows by {d} or a quarter, up to {d}", .{ q.kv_grow.first(s, capacity + window), s, capacity + window });
    errdefer h.runner.deinit();
    if (profileEvery()) |every| {
        h.trace = .{ .queue = model.queue, .report_every = every }; // enabled once measure() has priced the windows
        h.runner.profile(&h.trace.?);
    }
    if (draft_model) |d| h.draft = try lb.Draft.attach(gpa, &h.runner, d, owns_draft);
    draft_owned = false;
    errdefer if (h.draft) |d| d.deinit();
    if (sn_w) |mw| {
        h.sn_mtp = try q.sn_mtp.Mtp.initFromWeights(gpa, model, mw, slots, capacity + window);
        sn_w = null;
    }
    errdefer if (h.sn_mtp) |m| m.deinit();
    h.back = try lb.Metal.init(gpa, &h.runner, h.draft);
    errdefer h.back.deinit();
    h.back.serial_copy_match = policy.serialCopyMatch(units, window, lanes_value);
    h.back.chunk = h.runner.promptRows(@min(rows, reserve.requestedFor(model.device.generation(), model.config)));
    h.back.grow = .{ .ctx = h, .room = Host.roomFor, .added = Host.added };
    h.runner.grown = .{ .ctx = h, .added = Host.added };
    h.back.vision = o.images;
    h.warm = .{ .queue = model.queue }; // ready slots wired before the cache budget reads the footprint
    const residency = tf.startup_receipt.span(.residency);
    holdResident(h, io) catch |err| std.log.warn("qwen27: no residency set ({s}); the first request after an idle second re-wires the weights", .{@errorName(err)});
    residency.end();
    errdefer q.residency.drop(h); // decided before cache sizing and the worker starts
    h.back.sn_features = h.sn_mtp != null; // before a wrapper takes the backend's table
    if ((h.sn_mtp != null or !o.drafts) and copies_value == null) h.back.copies = false;
    if (h.sn_mtp) |m| {
        try m.warmup(io);
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.back.backend(), @import("sensenova_mtp_drafter.zig").drafter(m));
    }
    errdefer if (h.wrap) |*w| w.deinit();
    if (h.draft != null or mtp_on or h.sn_mtp != null) try h.back.measure(io);
    if (mtp_on) {
        const x = try Mtp.init(gpa, h.back, .{ .cap = o.mtp_drafts, .floor = o.mtp_confidence });
        h.mtp = x;
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.back.backend(), x.drafter());
        std.log.info("qwen27: the MTP head drafts chains of {d} at most, a step {d:.2} ms", .{ x.depth(), x.step_ms });
    }
    errdefer if (h.mtp) |x| x.deinit();
    h.cfg = try lanes.Config.init(gpa, if (h.wrap) |*w| w.facts(h.back.facts()) else h.back.facts(), window, h.back.nodes);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, if (h.wrap) |*w| w.backend() else h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const offer: ?api.Vision = if (o.images) |v| .{ .ctx = h, .prepare = Host.prepareImage, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.capacity, .max_images = v.max_images, .image_share = if (o.vision) |x| x.image_share else .request, .explain = tf.qwen_vl.geometry.problem } else null;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "qwen27", .draft_sources = .{ .drafter = if (h.draft) |d| .{ .kind = d.model.graph.config.kind(), .name = o.drafter_name orelse o.drafter orelse "" } else if (mtp_on) .{ .kind = "MTP", .name = q.gpu_weights.mtp_file } else null }, .structures = @import("qwen27_masks.zig").supported, .penalties = true, .lanes = slots, .context_window = capacity, .prefill_step = @intCast(h.back.chunk), .decode_share = o.decode_share, .vision = offer });
    h.host.memory = .{ .ctx = h, .read = readMemory };
    h.host.explain = .{ .text = words };
    h.host.decoded_rows = true; // a reply's rows prefill with the bits decoding gave them, with the cache on or off
    // A lone greedy request keeps its long-copy tail on the narrow tensor window; arrivals yield to lanes.
    if (h.draft != null and serial) h.host.lone = .{ .ctx = h, .run = Host.lone };
    // a reply that ends soon goes first beside a long prompt; a share someone set stays as they set it
    if (o.decode_share == null and platform.environment.rawText("TENSORFOLD_PREFILL_SHARE") == null) h.host.srpt = .withDefault(true);
    var loop_text: [160]u8 = undefined;
    std.log.info("qwen27: {s}", .{try policy.describe(&loop_text, serial, window, h.draft != null or h.sn_mtp != null, h.back.copies)});
    const ane = tf.startup_receipt.span(.optional_ane);
    if (model.share_pending) {
        const before = cache_fit.footprint();
        q.residency.finishShare(h, try shareRoom(h, why)) catch |err| {
            why.* = "the optional Neural Engine share could not join the residency owner within its load limit";
            return err;
        };
        if (model.ane) |share| share.footprint_growth = platform.memory_policy.growth(before, cache_fit.footprint());
    }
    ane.end(); // slot 0's history rewind: even two full copies may evict it
    const rewind_room = laterRoom(h);
    if (slot_state.rewindInit(h.back, rewind_room, if (o.cache_gib) |g| gibBytes(g) else cache_fit.defaultBudget(rewind_room))) |buf| if (h.resident) |*r| r.addNow(&.{buf});
    try enableCache(h, o.cache_gib, o.cache_over_cap, a, why);
    errdefer if (h.cache) |*store| store.deinit();
    if (o.learn) |l| try disk.attach(h, io, a, l, if (o.glp) |*g| g else null);
    errdefer if (h.cache) |*store| if (store.disk) |r| r.deinit();
    // a wide prompt pass commits its buffers at once: the probe keeps the sets in use while it runs
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick, .probe = mtl.keepalive.Target.busy };
    h.folded_mem = null;
    if (api.memoryWanted(o.dir, o.slide)) { // a saved bank alone never attaches it: --slide asks for learning
        const adapter = try qwen27_mem.Adapter.init(gpa, h.back);
        errdefer adapter.deinit(gpa);
        const holder = try mem_ctrl.Holder.init(gpa, io, o.dir, adapter.backend(), adapter, qwen27_mem.Adapter.deinitOpaque);
        h.folded_mem = holder;
        h.host.folded_memory = holder.controller();
        h.host.info_.folded_memory = true;
    }
    errdefer if (h.folded_mem) |fm| fm.deinit();
    if (model.idle_release) |owner| @import("qwen27_idle.zig").attach(h, owner);
    if (h.trace) |*t| t.enabled = true; // serving rounds count from here: measure() timed whole, unsplit buffers
    try h.host.start();
    tf.startup_receipt.span(.model_ready).end();
    std.log.info("{f}", .{StartupLine{ .streams = slots, .context = capacity, .draft_slots = if (h.draft != null) h.back.draft_slots else if (h.sn_mtp != null) slots else 0, .drafter = if (h.draft) |d| d.model.graph.config.kind() else "DFlash2", .images = o.images != null }});
    if (o.glp) |g| {
        if (model.steer != null) steer_served.report(g);
    }
    return h;
}

/// TF_Q27_PROFILE's rounds between log reports, or null (off); a value it cannot read leaves profiling off.
fn profileEvery() ?u64 {
    const v = std.mem.span(platform.c.getenv("TF_Q27_PROFILE") orelse return null);
    return tf.gpu_profile.every(v) catch {
        std.log.warn("TF_Q27_PROFILE={s} is not 0, 1 or a round count: profiling stays off", .{v});
        return null;
    };
}

/// The room under the load limit beside everything loaded and reserved, as the share's (unreadable guards: none).
fn laterRoom(h: *const Host) u64 {
    const guards = share_fit.Guards.read() catch return 0;
    return share_fit.room(cache_fit.ram(), cache_fit.footprint(), laterFootprint(h) catch return 0, guards);
}

/// Reserve later buffers and the loaded share's priced bytes that its construction left outside the footprint.
fn laterFootprint(h: *const Host) !u64 {
    const tower: u64 = if (h.back.vision) |v| v.firstUse() else 0;
    const share = try h.model.shareCacheBytes();
    return std.math.add(u64, try std.math.add(u64, laterBytes(h), tower), share);
}

/// The prompt bank may capture marks without splitting; only request starts are guaranteed cold cuts.
pub fn cacheRules(chunk: u32, message_starts: u8) pc.Rules {
    return .{ .warm = .on, .chunk = chunk, .chunk_marks = false, .message_starts = message_starts };
}

/// Prompt reuse from room under the memory cap once slot memory is counted; CacheOverCap explains the refusal.
fn enableCache(h: *Host, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !void {
    const later = try laterFootprint(h);
    const plan = try cache_fit.budgetBeside(gib, over, room(h.model) -| later, later, a, why, "");
    if (plan.budget_bytes == 0) return;
    // a short prompt keeps its history's mark at a chunk end its pass cuts anyway (qwen27_slots.prefill): no extra call
    h.cache = pc.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop, .charged = Snaps.charged, .spare = Snaps.spare, .trim = Snaps.trim, .reuses = Snaps.reuses, .rows = &disk.image } }, cacheRules(@intCast(h.back.chunk), try @import("qwen27_resume.zig").messageStarts(h.model.config.looped)), plan.budget_bytes);
    h.host.cache = &h.cache.?;
    h.back.release_hook = .{ .ctx = h, .kept = Host.keepReply };
    h.back.checkpoint_hook = .{ .ptr = h, .at = Host.checkpoint };
    h.host.info_.warm_turns = true;
    h.host.info_.prompt_cache = true;
    h.host.info_.prompt_cache_plan = plan;
}

/// Weights, the ready slots' state and the drafter join one residency set the idle keepalive uses.
const holdResident = q.residency.hold;

pub fn close(ptr: *anyopaque) void {
    const h = Host.of(ptr);
    h.host.stop();
    if (h.folded_mem) |fm| fm.deinit();
    const rows = if (h.cache) |*store| store.disk else null;
    if (h.cache) |*store| store.deinit(); // its states leave the set as they go
    h.spares.deinit();
    if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
    h.core.deinit();
    q.residency.drop(h);
    if (h.disk_stage) |b| b.deinit();
    h.cfg.deinit(h.gpa);
    if (h.wrap) |*w| w.deinit();
    if (h.mtp) |x| x.deinit();
    h.back.deinit();
    if (h.draft) |d| d.deinit();
    if (h.sn_mtp) |m| m.deinit();
    if (h.trace) |*t| t.reportRest();
    h.runner.deinit();
    if (h.vision) |v| v.deinit();
    if (h.owns_model) h.model.deinit();
    h.gpa.destroy(h);
}

test {
    if (builtin.os.tag == .macos) {
        _ = @import("qwen27_served_startup_test.zig");
        _ = @import("qwen27_draft_block.zig");
        _ = @import("qwen27_share_fit.zig");
        _ = @import("qwen27_served_test.zig");
        _ = @import("qwen27_startup_test.zig");
        _ = @import("qwen27_startup_line.zig");
        _ = @import("qwen27_mtp.zig");
    }
}
