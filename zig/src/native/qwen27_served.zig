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
        return q.snapshot.slotBytes(&h.runner, h.back.ringOf(0), at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const h = host(ptr);
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const slot = h.back.slotOf(s) orelse return error.UnknownStream;
        // Scratch already occupies its bytes; do not let cache growth spend that same free room again.
        if (h.runner.gdn.marks != null) {
            const need = q.snapshot.slotBytes(&h.runner, h.back.ringOf(slot), at);
            if (need > room(h.model) -| h.runner.gdn.missingBeforeBytes()) return error.LaneMemory;
        }
        const st = try q.snapshot.saveSlot(h.gpa, &h.runner, slot, h.back.ringOf(slot), at);
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
    return share_fit.room(cache_fit.ram(), cache_fit.footprint(), laterFootprint(h), guards);
}

/// Room left beside the weights and an 8 GiB margin.
fn room(model: *const q.model.Model) usize {
    const dev = model.device;
    return dev.maxWorkingSet() -| dev.allocated() -| (8 << 30);
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

/// One slot's caches at `context` tokens: its K/V rows, its taps ring when drafting, DeltaNet states and scratch.
fn slotCost(model: *const q.model.Model, context: u32, drafting: bool) usize {
    const c = model.config;
    var attention: usize = 0;
    for (0..c.layers) |i| attention += @intFromBool(c.kind(i) == .attention);
    const kv = attention * 2 * @as(usize, context) * c.kvDim() * 2;
    const ring_bytes: usize = if (drafting) 2048 * 5 * c.hidden * 2 else 0;
    return kv + ring_bytes + (256 << 20); // DeltaNet states, conv tails and scratch, generously
}

/// The most streams the server offers: slots whose caches fit `left`, at most `want`; a fixed count fits or refuses.
fn fit(left: usize, per: usize, want: u32, fixed: bool) error{TooManyStreams}!u32 {
    const n: u32 = @intCast(@max(1, @min(@as(usize, @max(want, 1)), left / per)));
    return if (fixed and n < want) error.TooManyStreams else n;
}

test fit {
    const g: usize = 1 << 30;
    try std.testing.expectEqual(@as(u32, 8), try fit(80 * g, 2 * g, 8, false));
    try std.testing.expectEqual(@as(u32, 3), try fit(7 * g, 2 * g, 8, false)); // auto serves what fits
    try std.testing.expectError(error.TooManyStreams, fit(7 * g, 2 * g, 8, true)); // a fixed --parallel refuses
    try std.testing.expectEqual(@as(u32, 3), try fit(7 * g, 2 * g, 3, true));
    try std.testing.expectEqual(@as(u32, 1), try fit(g, 2 * g, 8, false)); // one stream at least
}

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
    const per = slotCost(model, capacity + lb.max_rows, drafter != null or o.drafter != null); // a last window's rows past prompt + max_tokens
    const left = room(model);
    const slots = fit(left, per, o.streams, o.streams_fixed) catch |err| {
        why.* = try std.fmt.allocPrint(a, "--parallel {d} needs {d:.1} GiB for its streams' caches at a {d}-token context, and this Mac leaves {d:.1} GiB beside the weights and an 8 GiB margin: serve --parallel {d}, or lower --context", .{ o.streams, cache_fit.gibs(per * o.streams), capacity, cache_fit.gibs(left), @max(1, left / per) });
        return err;
    };
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
    h.runner = try q.decode_round.Runner.initReady(gpa, model, slots, capacity + lb.max_rows, 1);
    errdefer h.runner.deinit();
    if (drafter) |d| h.draft = try lb.Draft.attach(gpa, &h.runner, d, false) else if (o.drafter) |d| h.draft = lb.Draft.open(gpa, io, model, &h.runner, d, if (o.drafter_bits == 4) .prepared_q4_reference else .bf16_reference) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the native Qwen3.8-27B drafter cannot load {s} ({s})", .{ d, @errorName(err) });
        return error.DrafterLoad;
    };
    errdefer if (h.draft) |d| d.deinit();
    h.back = try lb.Metal.init(gpa, &h.runner, h.draft);
    errdefer h.back.deinit();
    h.back.chunk = h.runner.promptRows(q.model.promptRows());
    h.back.grow = .{ .ctx = h, .room = Host.roomFor, .added = Host.added };
    h.back.vision = o.images;
    if (h.draft != null) try h.back.measure(io);
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), lb.max_rows, lb.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const offer: ?api.Vision = if (o.images) |v| .{ .ctx = h, .prepare = Host.prepareImage, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.capacity, .max_images = v.max_images, .explain = tf.qwen_vl.geometry.problem } else null;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "qwen27", .structures = @import("qwen27_masks.zig").supported, .lanes = slots, .context_window = capacity, .prefill_step = @intCast(h.back.chunk), .vision = offer });
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
    if (model.share_pending) q.residency.finishShare(h, try shareRoom(h, why)) catch |err| {
        why.* = "the optional Neural Engine share could not join the residency owner within its load limit";
        return err;
    };
    try enableCache(h, o.cache_gib, o.cache_over_cap, a, why);
    errdefer if (h.cache) |*store| store.deinit();
    if (o.learn) |l| try disk.attach(h, io, a, l);
    errdefer if (h.cache) |*store| if (store.disk) |r| r.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    std.log.info("Qwen3.8-27B: {d} streams of {d} tokens{s}{s}", .{ slots, capacity, if (h.draft != null) ", DFlash2 drafts for one of them" else "", if (o.images != null) ", images read" else "" });
    return h;
}

/// Buffers absent from the footprint at ready: later slots and the image tower's first-use scratch and rows.
fn laterFootprint(h: *const Host) u64 {
    const tower: u64 = if (h.back.vision) |v| v.firstUse() else 0;
    return laterBytes(h) +| tower;
}

/// Prompt reuse in up to 16 GiB of what 70% of RAM leaves once every slot has its memory; CacheOverCap: `why` says why.
fn enableCache(h: *Host, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !void {
    const later = laterBytes(h);
    const plan = try cache_fit.budgetBeside(gib, over, room(h.model) -| later, laterFootprint(h), a, why, "");
    if (plan.budget_bytes == 0) return;
    // a short prompt keeps its history's mark at a chunk end its pass cuts anyway (qwen27_slots.prefill): no extra call
    h.cache = pc.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop, .rows = &disk.image } }, .{ .warm = .on, .chunk = @intCast(h.back.chunk) }, plan.budget_bytes);
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
    if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
    h.core.deinit();
    q.residency.drop(h);
    if (h.disk_stage) |b| b.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    if (h.draft) |d| d.deinit();
    h.runner.deinit();
    if (h.vision) |v| v.deinit();
    if (h.owns_model) h.model.deinit();
    h.gpa.destroy(h);
}

test {
    _ = @import("qwen27_share_fit.zig");
}
