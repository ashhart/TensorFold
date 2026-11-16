//! Qwen3.8 Flash Next (`qwen4_exp`) on the XPU server: TP={1,2,4} over Level Zero P2P + Host RAM PLE + MTP drafting.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const lane = @import("xpu_lane.zig");
const admit = @import("xpu_admit.zig");
const fn_xpu = @import("flashnext_xpu");
const Allocator = std.mem.Allocator;

const default_context: i64 = 16384;
const max_context: i64 = 262144;

const Spec = struct {
    pub const model_type = "qwen4_exp";
    pub const name = "flashnext-xpu";
    pub const Engine = fn_xpu.Engine;
    pub const cache_rules: api.prompt_cache.Rules = .{ .planned = true, .grid = 64 };

    pub fn promptRows(_: *const Engine) u32 {
        return 64;
    }
};

var stopping = std.atomic.Value(bool).init(false);
var prev_int: std.posix.Sigaction = undefined;
var prev_term: std.posix.Sigaction = undefined;

fn onStopSignal(sig: std.posix.SIG) callconv(.c) void {
    stopping.store(true, .release);
    const prev = if (sig == .INT) prev_int else prev_term;
    const f = prev.handler.handler orelse return;
    if (f != std.posix.SIG.DFL and f != std.posix.SIG.IGN) f(sig);
}

fn chainStopHandlers() void {
    const act: std.posix.Sigaction = .{ .handler = .{ .handler = onStopSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &act, &prev_int);
    std.posix.sigaction(.TERM, &act, &prev_term);
}

const Hosted = struct {
    gpa: Allocator,
    engine: *fn_xpu.Engine,
    backend: lane.Xpu(Spec),
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},
    cache: ?api.prompt_cache.Store = null,
    peak_bytes: u64 = 0,

    fn readMem(ptr: ?*anyopaque, reset_peak: bool) ?api.Memory {
        const h: *Hosted = @ptrCast(@alignCast(ptr orelse return null));
        const active: u64 = @as(u64, h.engine.allocatedBytesPerRank()) * @as(u64, h.engine.r.tp);
        if (active > h.peak_bytes) h.peak_bytes = active;
        const m: api.Memory = .{ .active = active, .cache = 0, .peak = h.peak_bytes };
        if (reset_peak) h.peak_bytes = active;
        return m;
    }

    fn close(p: *anyopaque) void {
        const h: *Hosted = @ptrCast(@alignCast(p));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.backend.deinit();
        h.engine.deinit();
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
};

pub fn resolveTp(o: api.Open) !usize {
    if (try @import("sys").environment.read(?u32, "XPU_TP", null)) |tp| return tp;
    return o.tp;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const settings = @import("sys").environment;
    if (try settings.refusal(a, settings.Process{}, .{ .xpu = true })) |why| {
        problem.* = why;
        return null;
    }
    const tp = try resolveTp(o);
    if (tp != 1 and tp != 2 and tp != 4) {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine supports --tp 1, 2 or 4 for Qwen3.8 Flash Next, not {d}", .{tp});
        return null;
    }
    const window: i64 = o.context orelse default_context;
    if (window <= 0 or window > max_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside Qwen3.8 Flash Next's {d}-token window", .{ window, max_context });
        return null;
    }
    const mtp_env = try settings.read(?bool, "XPU_MTP", null);
    const want_mtp = if (mtp_env == false) false else o.drafts;

    const engine = fn_xpu.Engine.init(gpa, io, o.dir, .{
        .context = @intCast(window),
        .tp = tp,
        .mtp = want_mtp,
    }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load Qwen3.8 Flash Next from {s} at tp={d}: {s}", .{ o.dir, tp, @errorName(err) });
        return null;
    };
    errdefer engine.deinit();

    // Compute free VRAM on Rank 0 to size prompt cache & admit concurrent streams
    const mem_opt = engine.r.devs[0].g.memory() catch null;
    const mem_free: u64 = if (mem_opt) |m| m.free else 0;
    const mem_size: u64 = if (mem_opt) |m| m.size else 0;
    const want_cache: u64 = std.math.lossyCast(u64, (o.prompt_cache_gib orelse 2.0) * (1 << 30));
    const margin: u64 = 2 << 30;
    const per_seq = engine.r.seqBytesPerRank(@intCast(window));
    // First sequence (`own`) is already allocated inside `engine.r.devs[0].allocated`!
    const ask_extra: u32 = if (o.lanes > 1) @min(o.lanes - 1, 31) else 0;
    const extra = admit.streamsThatFit(
        .{ .free = mem_free, .size = mem_size, .allocated = engine.r.devs[0].allocated },
        margin,
        want_cache,
        per_seq,
        ask_extra,
        31,
    );
    const streams_n: u32 = 1 + extra;

    const h = try gpa.create(Hosted);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.engine = engine;
    h.cache = null;
    h.startup = &.{};
    h.peak_bytes = @as(u64, engine.allocatedBytesPerRank()) * @as(u64, tp);

    h.backend = try lane.Xpu(Spec).init(gpa, engine);
    errdefer h.backend.deinit();
    h.backend.halt = &stopping;
    h.backend.started = chainStopHandlers;
    try h.backend.measure();
    if (h.backend.facts().exact_width > 1) h.backend.streams = streams_n;

    const facts = h.backend.facts();
    h.cfg = try lanes.Config.init(gpa, facts, facts.exact_width, facts.exact_width - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.backend.backend(), h.clock.clock());
    errdefer h.core.deinit();

    const drafted = facts.exact_width > 1;
    h.startup = try std.fmt.allocPrint(
        gpa,
        "Intel GPU engine: Qwen3.8 Flash Next (tp={d}, {d:.2} GiB/GPU, {d:.1} GiB PLE in host RAM, context {d}{s}; {d} stream(s))",
        .{
            tp,
            @as(f64, @floatFromInt(engine.allocatedBytesPerRank())) / (1 << 30),
            @as(f64, @floatFromInt(engine.r.ngram.total_bytes)) / (1 << 30),
            window,
            if (drafted) ", MTP drafts on" else ", plain decode",
            h.backend.streams,
        },
    );

    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .name = Spec.name,
        .lanes = h.backend.streams,
        .context_window = @intCast(window),
        .context_fitted = true,
        .startup = h.startup,
        .prefill_step = if (drafted) Spec.promptRows(engine) else 0,
    });

    const left_after_streams = mem_free -| (@as(u64, extra) * per_seq);
    const cache_budget: u64 = if (drafted and left_after_streams > margin)
        @min(want_cache, left_after_streams - margin) * tp
    else
        0;
    if (drafted and cache_budget > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, h.backend.snapshots(), Spec.cache_rules, cache_budget);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    h.host.memory = .{ .ctx = h, .read = Hosted.readMem };
    h.host.explain = .{ .text = lane.explain };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Hosted.close, .ctx = h };
}

test "Flash Next XPU keeps explicit --tp 1 and refuses invalid --tp before device access" {
    try std.testing.expectEqual(@as(usize, 1), try resolveTp(.{ .tp = 1 }));
    try std.testing.expectEqual(@as(usize, 4), try resolveTp(.{ .tp = 4 }));
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    try std.testing.expect(try open(arena.allocator(), std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = "qwen4_exp", .tp = 3 }, &problem) == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "--tp 1, 2 or 4") != null);
}
