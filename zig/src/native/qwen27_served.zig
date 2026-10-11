//! Qwen3.8-27B on the lane host: a stream a runner slot in shared rounds, DFlash2 drafts for slot 0's stream, a lone greedy
//! request on the DFlash2 serial loop.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const pc = api.prompt_cache;
const cache_fit = @import("cache_fit.zig");
const Resident = @import("resident.zig").Resident;
const lb = @import("qwen27_lanes.zig");
const Allocator = std.mem.Allocator;

pub const Options = struct {
    context: u32,
    streams: u32, // --parallel: slots at most, fewer when their caches do not fit
    drafter: ?[]const u8 = null,
    drafter_bits: u8 = 0,
    cache_gib: ?f64 = null,
    cache_over_cap: bool = false,
};

pub const Host = struct {
    gpa: Allocator,
    model: *q.model.Model,
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

    /// A finished reply's state stands where a prompt pass of prompt and reply would: keep it for the next turn.
    /// A slot's first use: whether its memory fits beside everything taken so far and the margin.
    fn roomFor(ptr: *anyopaque, bytes: usize) bool {
        return room(of(ptr).model) >= bytes;
    }

    /// A new slot's buffers join the residency set, so the keepalive holds them wired with the rest.
    fn added(ptr: *anyopaque, buffers: []const mtl.Buffer) void {
        const h = of(ptr);
        if (h.resident) |*r| r.add(buffers);
        std.log.info("Qwen3.8-27B: a slot took its memory ({d} of {d} ready)", .{ h.runner.ready, h.runner.slots });
    }

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

/// Slots whose caches would fit beside the weights and an 8 GiB margin, at most `want`, at least one: the most
/// streams the server offers. Slots past the first take their memory when first used (Host.grow).
fn fit(model: *const q.model.Model, context: u32, want: u32, drafting: bool) u32 {
    const c = model.config;
    var attention: usize = 0;
    for (0..c.layers) |i| attention += @intFromBool(c.kind(i) == .attention);
    const kv = attention * 2 * @as(usize, context) * c.kvDim() * 2;
    const ring_bytes: usize = if (drafting) 2048 * 5 * c.hidden * 2 else 0;
    const per = kv + ring_bytes + (256 << 20); // DeltaNet states, conv tails and scratch, generously
    return @intCast(@max(1, @min(@as(usize, @max(want, 1)), room(model) / per)));
}

pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, o: Options, a: Allocator, why: *[]const u8) !*Host {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.load(gpa, io, dir, 128);
    errdefer model.deinit();
    const capacity = if (o.context == 0) @min(model.config.max_position, 32768) else o.context;
    if (capacity == 0 or capacity > model.config.max_position) return error.ContextFull;
    const slots = fit(model, capacity, o.streams, o.drafter != null);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.draft = null;
    h.cache = null;
    h.resident = null;
    h.runner = try q.decode_round.Runner.initReady(gpa, model, slots, capacity, 1);
    errdefer h.runner.deinit();
    if (o.drafter) |d| h.draft = lb.Draft.open(gpa, io, model, &h.runner, d, if (o.drafter_bits == 4) .prepared_q4_reference else .bf16_reference) catch |err| {
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
    // a lone greedy request takes the serial loop where it is the faster one: on tensor units (TF_QWEN27_LANES overrides)
    const units = model.device.tensorUnits();
    const serial = if (std.c.getenv("TF_QWEN27_LANES")) |v| !std.mem.eql(u8, std.mem.span(v), "1") else units;
    if (h.draft != null and serial) h.host.lone = .{ .ctx = h, .run = Host.lone };
    try enableCache(h, o.cache_gib, o.cache_over_cap, a, why);
    errdefer if (h.cache) |*store| store.deinit();
    h.warm = .{ .queue = model.queue };
    holdResident(h) catch |err| std.log.warn("qwen27: no residency set ({s}); the first request after an idle second re-wires the weights", .{@errorName(err)});
    errdefer if (h.resident) |*r| r.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    std.log.info("Qwen3.8-27B: {d} streams of {d} tokens{s}", .{ slots, capacity, if (h.draft != null) ", DFlash2 drafts for one of them" else "" });
    return h;
}

/// Prompt reuse between turns and requests; on error.CacheOverCap `why` (in `a`) says why.
fn enableCache(h: *Host, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !void {
    const dev = h.model.device;
    const plan = try cache_fit.budget(gib, over, @min(dev.maxWorkingSet() -| dev.allocated() -| (8 << 30), 16 << 30), a, why, "");
    if (plan.budget_bytes == 0) return;
    h.cache = pc.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .warm = true }, plan.budget_bytes);
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
    h.model.deinit();
    h.gpa.destroy(h);
}
