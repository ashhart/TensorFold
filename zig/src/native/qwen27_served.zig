//! Qwen3.8-27B on the lane host: a stream a runner slot in shared rounds, DFlash2 drafts for slot 0's stream.
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
const Resident = @import("resident.zig").Resident;
const lb = @import("qwen27_lanes.zig");
pub const lanes_backend = lb;
const disk = @import("qwen27_disk.zig");
const images = @import("qwen27_images.zig");
const Allocator = std.mem.Allocator;

pub const Options = struct {
    context: u32,
    streams: u32, // --parallel: slots at most, fewer when their caches do not fit
    streams_fixed: bool = false, // --parallel named a number: every slot fits, or the server refuses
    drafter: ?[]const u8 = null,
    drafter_name: ?[]const u8 = null,
    drafter_bits: u8 = 0,
    cache_gib: ?f64 = null,
    cache_over_cap: bool = false,
    learn: ?disk.Learn = null, // --learn-conversations: kept states written behind to disk and read back
    vision: ?api.VisionOpen = null, // --vision: open loads the image tower, and requests may hold images
    images: ?*images.Vision = null, // the loaded tower attach serves (and the host then owns)
};

pub const Host = struct {
    gpa: Allocator,
    model: *q.model.Model,
    owns_model: bool,
    runner: q.decode_round.Runner,
    trace: ?tf.gpu_profile.Trace = null, // TF_Q27_PROFILE: rounds' class GPU times to the log, gpu_seconds in replies
    draft: ?*lb.Draft = null,
    back: *lb.Metal,
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

    /// Info.vision's prepare, on the HTTP thread.
    fn prepareImage(ptr: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const h = of(ptr);
        return images.prepare(h.vision.?, h.clock.io, a, bytes, max_tokens);
    }
};

/// The reply's state at `stands` (its end or one short), born with its request's prompt: that request's marks stay.
fn keepEnd(store: *pc.Store, gpa: Allocator, s: *lanes.Stream, stands: u32) void {
    const prompt: u32 = @intCast(s.prompt_len);
    const tokens = s.context.items;
    if (tokens.len == prompt or stands + 1 < tokens.len or stands > tokens.len) return;
    const spans = api.cache_modes.reply(gpa, s.decode_spans, prompt, stands) catch return;
    defer gpa.free(spans);
    // an image prompt's rows go by their images (the request's cache key), as its lookups and marks do
    const keyed = if (s.key.len == prompt) gpa.dupe(u32, tokens) catch return else tokens;
    defer if (keyed.ptr != tokens.ptr) gpa.free(keyed);
    if (keyed.ptr != tokens.ptr) @memcpy(keyed[0..prompt], s.key);
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

/// Room left beside the weights and an 8 GiB margin.
fn room(model: *const q.model.Model) usize {
    const dev = model.device;
    return q.prompt_memory.room(dev.maxWorkingSet(), dev.allocated(), if (std.c.getenv("TF_QWEN27_WORKING_SET_BYTES")) |v| std.mem.span(v) else null) catch 0;
}

/// Whether a new slot's `bytes` fit in `free`, once kept prompt states (never the one `restoring`) give theirs back.
fn takeRoom(cache: ?*pc.Store, free: usize, bytes: usize, restoring: ?pc.Saved, prompt: []const u32) bool {
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

/// Kept states of a fake family whose bytes grow as the 27B's K/V rows (64 KiB a token); `saved` is never read.
const FakeStates = struct {
    live: usize = 0,
    fn bytes(_: *anyopaque, at: u32) u64 {
        return @as(u64, at) << 16;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *FakeStates = @ptrCast(@alignCast(ptr));
        f.live += 1;
        return @ptrFromInt(@as(usize, at) << 4);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, _: pc.Saved) void {
        const f: *FakeStates = @ptrCast(@alignCast(ptr));
        f.live -= 1;
    }
};

test "--prompt-cache-gib 40: a later 192k-token slot takes its room back from kept states, never the one it restores" {
    const gpa = std.testing.allocator;
    const g: u64 = 1 << 30;
    var fake: FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } }, .{}, 40 * g);
    defer store.deinit();
    const tokens = try gpa.alloc(u32, 200_000);
    defer gpa.free(tokens);
    for ([_]u32{ 100_000, 150_000, 190_000 }, 0..) |at, k| { // three conversations' histories: 6.1, 9.2 and 11.6 GiB
        for (tokens, 0..) |*t, i| t.* = @intCast(k * 1_000_000 + i);
        try std.testing.expect(store.keep(tokens[0 .. at + 1], at, null, &.{}, &.{}));
    }
    const restoring = store.entries.items[2].saved; // the newest conversation's next turn resumes its state
    const slot: u64 = 12_900_000_000; // a --context 196608 slot's K/V rows, taps ring and scratch
    try std.testing.expect(!takeRoom(null, 4 * g, slot, null, tokens)); // without a cache to give back: LaneMemory
    try std.testing.expect(takeRoom(&store, 4 * g, slot, restoring, tokens)); // 4 GiB free + 6.1 + 9.2 freed
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expect(store.entries.items[0].saved == restoring);
    try std.testing.expect(!takeRoom(&store, 0, 2 * slot, restoring, tokens)); // its own state never goes
    try std.testing.expectEqual(@as(usize, 1), fake.live);
}

test "a reply's end is born with its request's prompt: the history mark that request kept outlives the reply's keep" {
    const gpa = std.testing.allocator;
    const at: u32 = 5000; // a history mark at 5000 tokens, then a 7-token generation prompt and a 100-token reply
    var fake: FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } }, .{ .warm = .on }, (3 * at + 100) << 16);
    defer store.deinit();
    var other: [at + 7]u32 = undefined; // another conversation's prompt, its history kept first
    for (&other, 0..) |*t, i| t.* = @intCast(i);
    try std.testing.expect(store.keep(&other, at, null, &.{}, &.{}));
    var tokens: [at + 107]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(1_000_000 + i);
    const prompt = tokens[0 .. at + 7];
    try std.testing.expect(store.keep(prompt, at, null, &.{}, &.{}));
    var s = try lanes.Stream.init(gpa, .{ .id = "reply", .prompt = prompt, .max_new = 100 });
    defer s.deinit(gpa);
    try s.context.appendSlice(gpa, tokens[at + 7 ..]);
    keepEnd(&store, gpa, &s, tokens.len); // three states pass the budget: the least recently used goes
    try std.testing.expect(store.find(&other, &.{}, &.{}) == null);
    const history = store.find(prompt, &.{}, &.{}) orelse return error.HistoryEvicted; // a regenerate resumes it
    try std.testing.expectEqual(at, history.at);
    try std.testing.expectEqual(@as(u32, at + 7), store.entries.items[1].born);
    try std.testing.expectEqual(@as(usize, 2), fake.live);
}

/// Memory the slots past the ready ones take on first use: their K/V rows and taps rings.
fn laterBytes(h: *const Host) u64 {
    const ring_bytes: usize = if (h.draft) |d| @as(usize, d.model.ring().window) * d.model.ring().stride else 0;
    return (h.runner.slots - h.runner.ready) * (h.runner.slotBytes() + ring_bytes) + h.runner.gdn.missingBeforeBytes();
}

pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, o: Options, a: Allocator, why: *[]const u8) !*Host {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.loadDeferred(gpa, io, dir, 128);
    errdefer model.deinit();
    var with = o;
    // the tower before the slots: their room and the cache's budget count it as taken
    if (o.vision) |v| with.images = try images.open(gpa, io, model, dir, v, a, why);
    errdefer if (with.images) |v| v.deinit();
    const h = try attach(gpa, io, model, null, with, a, why);
    h.owns_model = true;
    return h;
}

/// A loaded model served as `open` serves one, `drafter` bound in place of `o.drafter`; both stay the caller's.
pub fn attach(gpa: Allocator, io: std.Io, model: *q.model.Model, drafter: ?*q.dflash.runtime_model.Model, o: Options, a: Allocator, why: *[]const u8) !*Host {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const capacity = if (o.context == 0) @min(model.config.max_position, 32768) else o.context;
    if (capacity == 0 or capacity > model.config.max_position) return error.ContextFull;
    var draft_model = drafter;
    const owns_draft = drafter == null and o.drafter != null;
    if (owns_draft) draft_model = q.dflash.runtime_model.Model.load(gpa, io, model, o.drafter.?, if (o.drafter_bits == 4) .prepared_q4_reference else .bf16_reference) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the native Qwen3.8-27B drafter cannot load {s} ({s})", .{ o.drafter.?, @errorName(err) });
        return error.DrafterLoad;
    };
    var draft_owned = owns_draft;
    errdefer if (draft_owned) draft_model.?.deinit();
    const memory = q.prompt_memory;
    const fit_prompt = memory.Fit{ .config = model.config, .context = capacity + lb.max_rows, .slots = @max(o.streams, 1), .shared = model.device.tensorUnits(), .ring = if (draft_model) |d| @as(usize, d.ring().window) * d.ring().stride else 0, .current_rows = model.frame.capacity, .left = room(model) };
    const value = if (std.c.getenv("TF_PROMPT_ROWS")) |v| std.mem.span(v) else null;
    const row_policy = q.prompt_rows;
    const rows = @max(lb.max_rows, try fit_prompt.choose(row_policy.requested(model.device.generation(), value), row_policy.override(value) != null));
    const slots = fit_prompt.streams(rows, o.streams_fixed) catch |err| {
        why.* = try std.fmt.allocPrint(a, "--parallel {d} needs {d:.1} GiB for its streams and {d}-row prompt scratch at a {d}-token context; {d:.1} GiB remains beside the weights and an 8 GiB margin: lower --parallel or --context", .{ o.streams, cache_fit.gibs(try fit_prompt.cost(rows, @max(o.streams, 1))), rows, capacity, cache_fit.gibs(fit_prompt.left) });
        return err;
    };
    if (model.frame.capacity != rows) {
        const next = try q.gpu_frame.Frame.init(model.device, model.config, rows);
        model.frame.deinit();
        model.frame = next;
    }
    std.log.info("qwen27: prompt capacity={d} rows, startup stream and scratch reserve={d} bytes", .{ rows, try memory.runnerBytes(model.config, capacity + lb.max_rows, slots, rows, model.device.tensorUnits(), fit_prompt.ring) });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.owns_model = false;
    h.draft = null;
    h.cache = null;
    h.resident = null;
    h.disk_stage = null;
    h.vision = o.images; // closed with the host; a failed attach leaves it to the caller
    h.trace = null;
    h.runner = try q.decode_round.Runner.initReady(gpa, model, slots, capacity + lb.max_rows, 1);
    errdefer h.runner.deinit();
    if (profileEvery()) |every| {
        h.trace = .{ .queue = model.queue, .report_every = every }; // enabled once measure() has priced the windows
        h.runner.profile(&h.trace.?);
    }
    if (draft_model) |d| h.draft = try lb.Draft.attach(gpa, &h.runner, d, owns_draft);
    draft_owned = false;
    errdefer if (h.draft) |d| d.deinit();
    h.back = try lb.Metal.init(gpa, &h.runner, h.draft);
    errdefer h.back.deinit();
    h.back.chunk = h.runner.promptRows(@min(rows, row_policy.requested(model.device.generation(), value)));
    h.back.grow = .{ .ctx = h, .room = Host.roomFor, .added = Host.added };
    h.back.vision = o.images;
    if (h.draft != null) try h.back.measure(io);
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), lb.max_rows, lb.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const offer: ?api.Vision = if (o.images) |v| .{ .ctx = h, .prepare = Host.prepareImage, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.capacity, .max_images = v.max_images, .explain = tf.qwen_vl.geometry.problem } else null;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "qwen27", .draft_sources = .{ .drafter = if (h.draft != null) .{ .kind = "DFlash2", .name = o.drafter_name orelse o.drafter orelse "" } else null }, .structures = @import("qwen27_masks.zig").supported, .lanes = slots, .context_window = capacity, .prefill_step = @intCast(h.back.chunk), .vision = offer });
    h.host.memory = .{ .ctx = h, .read = readMemory };
    h.host.explain = .{ .text = words };
    h.host.decoded_rows = true; // a reply's rows prefill with the bits decoding gave them, with the cache on or off
    // a lone greedy request takes the serial loop on tensor units, where it is faster (TF_QWEN27_LANES overrides)
    const units = model.device.tensorUnits();
    const serial = if (platform.c.getenv("TF_QWEN27_LANES")) |v| !std.mem.eql(u8, std.mem.span(v), "1") else units;
    if (h.draft != null and serial) h.host.lone = .{ .ctx = h, .run = Host.lone };
    h.warm = .{ .queue = model.queue };
    // the ready slots are wired before the cache's budget reads the footprint, and the later ones count as taken
    holdResident(h, io) catch |err| std.log.warn("qwen27: no residency set ({s}); the first request after an idle second re-wires the weights", .{@errorName(err)});
    errdefer q.residency.drop(h);
    // Decide before cache sizing and the worker starts, with ready state wired and future buffers reserved.
    if (model.share_pending) {
        const before = cache_fit.footprint();
        q.residency.finishShare(h, try shareRoom(h, why)) catch |err| {
            why.* = "the optional Neural Engine share could not join the residency owner within its load limit";
            return err;
        };
        if (model.ane) |share| share.footprint_growth = platform.memory_policy.growth(before, cache_fit.footprint());
    }
    try enableCache(h, o.cache_gib, o.cache_over_cap, a, why);
    errdefer if (h.cache) |*store| store.deinit();
    if (o.learn) |l| try disk.attach(h, io, a, l);
    errdefer if (h.cache) |*store| if (store.disk) |r| r.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    if (h.trace) |*t| t.enabled = true; // serving rounds count from here: measure() timed whole, unsplit buffers
    try h.host.start();
    std.log.info("Qwen3.8-27B: {d} streams of {d} tokens{s}{s}", .{ slots, capacity, if (h.draft != null) ", DFlash2 drafts for one of them" else "", if (o.images != null) ", images read" else "" });
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

/// Reserve later buffers and the loaded share's priced bytes that its construction left outside the footprint.
fn laterFootprint(h: *const Host) !u64 {
    const tower: u64 = if (h.back.vision) |v| v.firstUse() else 0;
    const share = try h.model.shareCacheBytes();
    return std.math.add(u64, try std.math.add(u64, laterBytes(h), tower), share);
}

/// The prompt bank may capture marks without splitting; only request starts are guaranteed cold cuts.
pub fn cacheRules(chunk: u32) pc.Rules {
    return .{ .warm = .on, .chunk = chunk, .chunk_marks = false };
}

/// Prompt reuse in up to 16 GiB of what 70% of RAM leaves once every slot has its memory; CacheOverCap: `why` says why.
fn enableCache(h: *Host, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !void {
    const later = try laterFootprint(h);
    const plan = try cache_fit.budgetBeside(gib, over, room(h.model) -| later, later, a, why, "");
    if (plan.budget_bytes == 0) return;
    // a short prompt keeps its history's mark at a chunk end its pass cuts anyway (qwen27_slots.prefill): no extra call
    h.cache = pc.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop, .charged = Snaps.charged, .spare = Snaps.spare, .trim = Snaps.trim, .reuses = Snaps.reuses, .rows = &disk.image } }, cacheRules(@intCast(h.back.chunk)), plan.budget_bytes);
    h.host.cache = &h.cache.?;
    h.back.release_hook = .{ .ctx = h, .kept = Host.keepReply };
    h.host.info_.warm_turns = true;
    h.host.info_.prompt_cache = true;
    h.host.info_.prompt_cache_plan = plan;
}

/// Weights, the ready slots' state and the drafter join one residency set the idle keepalive uses.
const holdResident = q.residency.hold;

pub fn close(ptr: *anyopaque) void {
    const h = Host.of(ptr);
    h.host.stop();
    const rows = if (h.cache) |*store| store.disk else null;
    if (h.cache) |*store| store.deinit(); // its states leave the set as they go
    h.spares.deinit();
    if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
    h.core.deinit();
    q.residency.drop(h);
    if (h.disk_stage) |b| b.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    if (h.draft) |d| d.deinit();
    if (h.trace) |*t| t.reportRest();
    h.runner.deinit();
    if (h.vision) |v| v.deinit();
    if (h.owns_model) h.model.deinit();
    h.gpa.destroy(h);
}

test {
    _ = @import("qwen27_share_fit.zig");
}

test "Qwen27 interior state uses a guaranteed cold request cut" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try a.alloc(u32, 128 * 1024 + 64);
    @memset(tokens, 7);
    var fake: FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } }, cacheRules(2048), 32 << 30);
    defer store.deinit();
    var starts: std.ArrayList(u32) = .empty;
    var at: u32 = 2048;
    while (at < tokens.len) : (at += 2048) try starts.append(a, at);
    const p = try store.beginUser(a, tokens, @intCast(tokens.len - 8), 16, @intCast(tokens.len - 8), &.{}, starts.items, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 129024, @intCast(tokens.len - 8) }, p.marks);
    for (p.marks) |m| try std.testing.expect(store.keep(tokens, m, null, starts.items, &.{}));
    tokens[tokens.len - 512] += 1;
    const followup = try store.beginUser(a, tokens, @intCast(tokens.len - 8), 16, @intCast(tokens.len - 8), &.{}, starts.items, null, &.{});
    try std.testing.expectEqual(@as(u32, 129024), followup.from);
    try std.testing.expect(store.held <= store.budget);
}
