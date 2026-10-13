//! Qwen3.8-27B on the lane host: a stream a runner slot in shared rounds, DFlash2 drafts for slot 0's stream.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const pc = api.prompt_cache;
const cache_fit = @import("cache_fit.zig");
const Resident = @import("resident.zig").Resident;
const lb = @import("qwen27_lanes.zig");
pub const lanes_backend = lb;
const Allocator = std.mem.Allocator;

pub const Options = struct {
    context: u32,
    streams: u32, // --parallel: slots at most, fewer when their caches do not fit
    streams_fixed: bool = false, // --parallel named a number: every slot fits, or the server refuses
    drafter: ?[]const u8 = null,
    drafter_bits: u8 = 0,
    cache_gib: ?f64 = null,
    cache_over_cap: bool = false,
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
        return takeRoom(if (h.cache) |*c| c else null, room(h.model), bytes, s.reuse.saved, s.prompt());
    }

    /// A new slot's buffers join the residency set, so the keepalive holds them wired with the rest.
    fn added(ptr: *anyopaque, buffers: []const mtl.Buffer) void {
        const h = of(ptr);
        if (h.resident) |*r| r.add(buffers);
        std.log.info("Qwen3.8-27B: a slot took its memory ({d} of {d} ready)", .{ h.runner.ready, h.runner.slots });
    }

    /// A finished reply's state stands where a prompt pass of prompt and reply would: keep it for the next turn.
    fn keepReply(ptr: *anyopaque, s: *lanes.Stream, slot: u32) void {
        const h = of(ptr);
        const store = if (h.cache) |*c| c else return;
        const prompt: u32 = @intCast(s.prompt_len);
        const tokens = s.context.items;
        const stands = h.runner.offsets[slot];
        if (tokens.len == prompt or stands + 1 < tokens.len or stands > tokens.len) return;
        const spans = api.cache_modes.reply(h.gpa, s.decode_spans, prompt, stands) catch return;
        defer h.gpa.free(spans);
        _ = store.keep(tokens, stands, s, s.chunks, spans);
    }
};

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
        return @ptrCast(try q.snapshot.saveSlot(h.gpa, &h.runner, slot, h.back.ringOf(slot), at));
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.RestoredInPrefill; // the backend restores a stream's kept state in its prompt pass
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        q.snapshot.drop(host(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};

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

/// Memory the slots past the ready ones take on first use: their K/V rows and taps rings.
fn laterBytes(h: *const Host) u64 {
    const ring_bytes: usize = if (h.draft) |d| @as(usize, d.model.ring().window) * d.model.ring().stride else 0;
    return (h.runner.slots - h.runner.ready) * (h.runner.slotBytes() + ring_bytes);
}

pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, o: Options, a: Allocator, why: *[]const u8) !*Host {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.load(gpa, io, dir, 128);
    errdefer model.deinit();
    const h = try attach(gpa, io, model, null, o, a, why);
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
    h.runner = try q.decode_round.Runner.initReady(gpa, model, slots, capacity + lb.max_rows, 1);
    errdefer h.runner.deinit();
    if (drafter) |d| h.draft = try lb.Draft.attach(gpa, &h.runner, d, false) else if (o.drafter) |d| h.draft = lb.Draft.open(gpa, io, model, &h.runner, d, if (o.drafter_bits == 4) .prepared_q4_reference else .bf16_reference) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the native Qwen3.8-27B drafter cannot load {s} ({s})", .{ d, @errorName(err) });
        return error.DrafterLoad;
    };
    errdefer if (h.draft) |d| d.deinit();
    h.back = try lb.Metal.init(gpa, &h.runner, h.draft);
    errdefer h.back.deinit();
    h.back.grow = .{ .ctx = h, .room = Host.roomFor, .added = Host.added };
    if (h.draft != null) try h.back.measure(io);
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), lb.max_rows, lb.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "qwen27", .lanes = slots, .context_window = capacity, .prefill_step = 128 });
    h.host.explain = .{ .text = words };
    h.host.decoded_rows = true; // a reply's rows prefill with the bits decoding gave them, with the cache on or off
    // a lone greedy request takes the serial loop on tensor units, where it is faster (TF_QWEN27_LANES overrides)
    const units = model.device.tensorUnits();
    const serial = if (std.c.getenv("TF_QWEN27_LANES")) |v| !std.mem.eql(u8, std.mem.span(v), "1") else units;
    if (h.draft != null and serial) h.host.lone = .{ .ctx = h, .run = Host.lone };
    h.warm = .{ .queue = model.queue };
    // the ready slots are wired before the cache's budget reads the footprint, and the later ones count as taken
    holdResident(h) catch |err| std.log.warn("qwen27: no residency set ({s}); the first request after an idle second re-wires the weights", .{@errorName(err)});
    errdefer if (h.resident) |*r| r.deinit();
    try enableCache(h, o.cache_gib, o.cache_over_cap, a, why);
    errdefer if (h.cache) |*store| store.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    std.log.info("Qwen3.8-27B: {d} streams of {d} tokens{s}", .{ slots, capacity, if (h.draft != null) ", DFlash2 drafts for one of them" else "" });
    return h;
}

/// Prompt reuse in up to 16 GiB of what 70% of RAM leaves once every slot has its memory; CacheOverCap: `why` says why.
fn enableCache(h: *Host, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !void {
    const later = laterBytes(h);
    const plan = try cache_fit.budgetBeside(gib, over, room(h.model) -| later, later, a, why, "");
    if (plan.budget_bytes == 0) return;
    // a short prompt keeps its history's mark at a chunk end its pass cuts anyway (qwen27_slots.prefill): no extra call
    h.cache = pc.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .warm = true, .chunk = @intCast(h.back.chunk) }, plan.budget_bytes);
    h.host.cache = &h.cache.?;
    h.back.release_hook = .{ .ctx = h, .kept = Host.keepReply };
    h.host.info_.warm_turns = true;
    h.host.info_.prompt_cache = true;
    h.host.info_.prompt_cache_plan = plan;
}

/// Weights, the ready slots' state and the drafter join one residency set the idle keepalive uses.
fn holdResident(h: *Host) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    for (h.model.checkpoint.shards.items) |s| try list.append(h.gpa, s.buffer);
    try list.appendSlice(h.gpa, h.model.weights.owned.items);
    try list.appendSlice(h.gpa, &h.model.frame.buffers);
    try list.appendSlice(h.gpa, &h.runner.gdn.buffers);
    try h.runner.readyBuffers(h.gpa, &list);
    for (h.back.rings) |r| if (r) |x| try list.append(h.gpa, x.buf);
    if (h.draft) |d| {
        for (d.model.checkpoint.shards.items) |s| try list.append(h.gpa, s.buffer);
        try list.append(h.gpa, d.model.pending);
    }
    h.resident = try Resident.init(h.model.device, h.model.queue, list.items);
    h.warm = (&h.resident.?).target();
}

pub fn close(ptr: *anyopaque) void {
    const h = Host.of(ptr);
    h.host.stop();
    if (h.resident) |*r| r.deinit();
    if (h.cache) |*store| store.deinit();
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    if (h.draft) |d| d.deinit();
    h.runner.deinit();
    if (h.owns_model) h.model.deinit();
    h.gpa.destroy(h);
}
