//! Opt-in single-Mac Flash Next serving through the lane core, sharing each target forward.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const Backend = tf.flashnext_backend.Backend;
const Session = tf.flashnext_session.Session;
const fx = tf.flashnext_engine;
const fit = @import("cache_fit.zig");
const budgets = @import("flashnext_batch_budget.zig");
const Resident = @import("resident.zig").Resident;
const images = @import("flashnext_images.zig");
const VisPrompt = tf.flashnext_replay.vis_prompt.VisPrompt;

const Host = struct {
    gpa: std.mem.Allocator,
    eng: *fx.Engine,
    back: Backend,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null,
    warm: mtl.keepalive.Target,
    resident: ?Resident,
    load_limit: u64,
    vis: ?*images.Vision, // --vision: the image tower
};

/// The whole-footprint ceiling: 70% of RAM or FLASHNEXT_LOAD_LIMIT_GB, capped by the GPU working set.
fn ceiling(total: u64, working_set: u64) error{BadLoadLimit}!u64 {
    const text: ?[]const u8 = if (std.c.getenv("FLASHNEXT_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const c = fit.ceiling(total, working_set, text) catch |err| {
        std.log.err("flash next: FLASHNEXT_LOAD_LIMIT_GB={s} is not a load limit: give a number of GB above 0, or unset it for 70% of RAM", .{text.?});
        return err;
    };
    if (c.capped) std.log.info("flash next: FLASHNEXT_LOAD_LIMIT_GB={s} passes this GPU's recommended working set; the limit is {d:.1} GB", .{ text.?, fit.gibs(c.bytes) * 1.073741824 });
    if (c.bytes > c.default) std.log.warn("flash next: FLASHNEXT_LOAD_LIMIT_GB lets the server (sessions and prompt cache) take this Mac to {d:.1} GiB, past the default {d:.1} GiB (70% of RAM): a load that wires most of the RAM can stall macOS until its watchdog kills the server", .{ fit.gibs(c.bytes), fit.gibs(c.default) });
    return c.bytes;
}

/// The sessions past the engine's own from FLASHNEXT_SLOT_CONTEXTS (as "131072,65536x2,16384x8"); null when unset.
fn slotContexts(gpa: std.mem.Allocator, capacity: usize) !?[]usize {
    const text = std.mem.trim(u8, std.mem.span(std.c.getenv("FLASHNEXT_SLOT_CONTEXTS") orelse return null), " \t");
    if (text.len == 0) return null;
    return try tf.flashnext_slots.parse(gpa, text, capacity, @intCast(fx.MARGIN), tf.flashnext_replay.MAXR - 1);
}

/// Admission requires FLASHNEXT_SLOT_GROW reply tokens (default 8192); longer replies can move.
fn slotGrow() usize {
    const v = std.c.getenv("FLASHNEXT_SLOT_GROW") orelse return 8192;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 8192;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, dump: ?[]const u8, window: i64, streams: u32, streams_fixed: bool, cache_gib: ?f64, over: bool, vision: ?api.VisionOpen, a: std.mem.Allocator, why: *[]const u8) !api.Opened {
    return openConfigured(gpa, io, dir, dump, window, streams, streams_fixed, cache_gib, over, vision, a, why, try tf.flashnext_replay.window_config.request(window, tf.flashnext_replay.TRAINED, fx.MARGIN, std.c.getenv("TF_FLASHNEXT_TP") != null, why));
}

pub fn openConfigured(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, dump: ?[]const u8, window: i64, streams: u32, streams_fixed: bool, cache_gib: ?f64, over: bool, vision: ?api.VisionOpen, a: std.mem.Allocator, why: *[]const u8, settings: tf.flashnext_replay.WindowSettings) !api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const capacity: usize = @intCast(@min(@as(i64, @intCast(settings.capacity)), window + fx.MARGIN));
    const tiers = slotContexts(gpa, capacity) catch |err| {
        std.log.err("flash next: FLASHNEXT_SLOT_CONTEXTS is not a list of sessions: token counts of 1024 or more, CONTEXTxN for N alike, at most {d} sessions past the engine's own", .{tf.flashnext_replay.MAXR - 1});
        return err;
    };
    defer if (tiers) |t| gpa.free(t);
    var budget_settings = try @import("flashnext_memory").withCache(settings, cache_gib);
    var more: u64 = 0;
    if (tiers) |ts| {
        for (ts) |c| more = try std.math.add(u64, more, Session.bytes(c));
    } else if (streams_fixed and streams > 1) more = try std.math.mul(u64, streams - 1, Session.bytes(capacity));
    if (more > 0) budget_settings.startup_extra = try std.math.add(u64, budget_settings.startup_extra, try std.math.add(u64, more, Session.overhead(settings.capacity) + tf.flashnext_replay.MAXR * tf.flashnext_replay.WIDE * 2));
    budget_settings = try images.budgetSettings(io, dir, vision, std.c.getenv("TF_FLASHNEXT_TP") != null, a, why, budget_settings);
    const eng = try fx.Engine.loadConfigured(gpa, io, dir, dump, null, budget_settings);
    errdefer eng.deinit();
    if (eng.r.tp != null) return error.BatchSpeedUpUnsupported;
    try eng.warm();
    const vis: ?*images.Vision = if (vision) |o| try images.open(gpa, io, eng, dir, o, a, why) else null; // before the footprint: the sessions fit beside it
    errdefer if (vis) |v| v.deinit();
    const ready = fit.footprint() orelse return error.NoMemoryReading;
    const total = fit.ram() orelse return error.NoMemoryReading;
    const cap = @min(eng.r.load_limit, try ceiling(total, eng.r.device.maxWorkingSet()));
    var plan = try fit.budgetUnder(cap, cache_gib, over, 0, a, why, "");
    const sizing = budgets.reserve(cap, ready, fit.MARGIN, plan.budget_bytes, cache_gib != null, over);
    const overhead = Session.overhead(eng.r.window.capacity) + tf.flashnext_replay.MAXR * tf.flashnext_replay.WIDE * 2;
    const available = sizing.sessions;
    if (available < overhead) {
        if (!streams_fixed and tiers == null) return single(gpa, io, eng, window, cache_gib, over, cap, vis, a, why);
        return error.OverMemoryLimit;
    }
    const room = available - overhead;
    var caps: std.ArrayList(usize) = .empty;
    defer caps.deinit(gpa);
    try caps.append(gpa, capacity); // the engine's own session: the whole window
    if (tiers) |t| {
        var bytes: u64 = 0;
        for (t) |c| bytes += Session.bytes(c);
        if (bytes > room) {
            std.log.err("flash next: the sessions FLASHNEXT_SLOT_CONTEXTS lists take {d:.1} GiB; {d:.1} GiB fits under the limit beside the prompt cache", .{ fit.gibs(bytes), fit.gibs(room) });
            return error.OverMemoryLimit;
        }
        try caps.appendSlice(gpa, t);
    } else {
        // past the trained window only the engine's own session holds it (YaRN's); the others hold the trained window
        const each = @min(capacity, tf.flashnext_replay.TRAINED + @as(usize, @intCast(fx.MARGIN)));
        const n = try tf.flashnext_batch_meta.fit(streams, streams_fixed, room, Session.bytes(each));
        if (n < 2) return single(gpa, io, eng, window, cache_gib, over, cap, vis, a, why);
        for (1..n) |_| try caps.append(gpa, each);
    }
    eng.hostMode();
    const slots: u32 = @intCast(caps.items.len);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.eng = eng;
    h.vis = vis;
    h.back = try Backend.init(gpa, eng, caps.items);
    errdefer h.back.deinit();
    if (vis != null) h.back.images = .{ .ctx = h, .build = buildImages, .release = releaseImages };
    if (tiers != null or capacity > tf.flashnext_replay.TRAINED + @as(usize, @intCast(fx.MARGIN))) { // sessions of two sizes
        h.back.sized = true;
        h.back.grow = slotGrow();
    }
    try h.back.timeCosts();
    const allocated = fit.footprint() orelse return error.NoMemoryReading;
    const final = try budgets.afterSessions(sizing, cap, allocated, fit.MARGIN);
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), tf.flashnext_replay.MAXR, tf.flashnext_replay.MAXR - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    plan.ready_footprint_bytes = allocated;
    plan.budget_bytes = final.cache;
    plan.room_bytes = final.room;
    plan.over_cap = final.over_cap;
    eng.snap_pool.max = plan.budget_bytes;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "flashnext-zig", .lanes = slots, .context_window = @intCast(window), .prefill_step = @intCast(eng.pr.step), .prompt_cache_plan = plan, .structures = true, .admit = if (h.back.sized) .{ .ctx = &h.back, .fits = Backend.fits } else null });
    // FLASHNEXT_NO_LONE=1 keeps lone requests on shared rounds; image streams always (engine rounds: slot positions).
    if (eng.r.gdn_kept == null and std.c.getenv("FLASHNEXT_NO_LONE") == null) h.host.lone = .{ .ctx = h, .run = loneRun, .images = false };
    // The standard frame (buildImages): an image moves no row before it, so the cache keys carry no frame.
    if (vis) |v| h.host.info_.vision = images.offer(v, h, prepareImage, prepareVideo);
    h.cache = null;
    if (plan.budget_bytes > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = Backend.snapBytes, .save = Backend.snapSave, .restore = Backend.snapRestore, .drop = Backend.snapDrop, .charged = Backend.snapCharged, .spare = Backend.snapSpare, .trim = Backend.snapTrim, .reuses = Backend.snapReuses } }, .{ .lookahead = 1, .planned = true, .warm = .on }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.host.info_.warm_turns = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.host.explain = .{ .text = words };
    h.warm = .{ .queue = eng.r.queue };
    h.resident = null;
    h.load_limit = cap;
    if (std.c.getenv("FLASHNEXT_NO_RESIDENCY") == null) holdResident(h) catch |err|
        std.log.warn("flash next shared: no residency set ({s})", .{@errorName(err)});
    errdefer dropResident(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    h.host.memory = .{ .ctx = h, .read = memory };
    var listed: [tf.flashnext_replay.MAXR * 8]u8 = undefined;
    var w: std.Io.Writer = .fixed(&listed);
    for (caps.items, 0..) |c, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", c }) catch break;
    var line: [640]u8 = undefined;
    var startup: std.Io.Writer = .fixed(&line);
    try budgets.startup(&startup, slots, w.buffered(), sizing, h.back.serial_drafts, h.host.lone != null);
    try h.host.start();
    std.log.info("{s}", .{startup.buffered()});
    return .{ .engine = h.host.engine(), .close = close, .ctx = h };
}

fn buildImages(ctx: *anyopaque, a: std.mem.Allocator, imgs: []const lanes.stream.Image, prompt_len: usize, from: usize) anyerror!*VisPrompt {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return images.build(h.vis.?, h.eng, a, imgs, prompt_len, from, true); // the standard frame
}

fn releaseImages(_: *anyopaque, p: *VisPrompt) void {
    p.rows.deinit();
}

/// Info.vision's video prepare: on the HTTP thread.
fn prepareVideo(ctx: *anyopaque, a: std.mem.Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return images.prepareVideo(h.vis.?, a, src, max_tokens, max_frames);
}

/// Info.vision's prepare: on the HTTP thread.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return images.prepare(h.vis.?, h.host.io, a, bytes, max_tokens);
}

fn loneRun(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return h.back.lone(s, hooks);
}

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        else => null,
    };
}

fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    if (h.cache) |*store| store.deinit();
    dropResident(h);
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    if (h.vis) |v| v.deinit();
    h.eng.deinit();
    h.gpa.destroy(h);
}

fn memory(ctx: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const h: *Host = @ptrCast(@alignCast(ctx.?));
    return fit.hostMemory(h.load_limit, reset_peak);
}

/// Session-owned buffers stay allocated until close, beside the runner's buffers and shared hidden rows.
const holdResident = @import("flashnext_residency.zig").holdShared;
const dropResident = @import("flashnext_residency.zig").dropShared;

/// Auto with room for only one session transfers the loaded engine, preserving the chosen footprint limit.
fn single(gpa: std.mem.Allocator, io: std.Io, eng: *fx.Engine, window: i64, cache_gib: ?f64, over: bool, cap: u64, vis: ?*images.Vision, a: std.mem.Allocator, why: *[]const u8) !api.Opened {
    const host = @import("flashnext_host.zig");
    const h = try host.openWarmed(gpa, io, eng, window, cache_gib, over, null, cap, vis, a, why);
    return .{ .engine = h.engine(), .close = host.close, .ctx = h };
}
