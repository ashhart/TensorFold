//! Inkling 2 Small EXL3 on Metal: text-only serving with exact prompt reuse and budgeted stream state.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const lanes = @import("lanes");
const exl3_pack = @import("../families/inkling2/exl3_pack.zig");
const Model = @import("../families/inkling2/metal/small_model.zig").Model;
const st = @import("../families/inkling2/metal/small_state.zig");
const backend = @import("../families/inkling2/metal/small_backend.zig");

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *Model,
    metal: *backend.Metal,
    warm: mtl.keepalive.Target,
    clock: lanes.backend.WallClock,
    config: lanes.Config = undefined,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null,
    cache_cap: u64 = 0,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn refusal(o: api.Open) ?[]const u8 {
    if (exl3_pack.refusal(.{
        .vision = o.vision != null,
        .cfg = o.cfg_enabled or o.cfg_batched,
        .loops = o.loops,
        .precision = if (o.precision) |p| @tagName(p) else null,
    })) |why| return why;
    if (o.mtp_drafts != null or o.mtp_confidence != null) return "inkling2: external MTP draft controls are unsupported";
    if (o.drafter != null) return "inkling2: Small EXL3 DSpark drafter is not wired yet; serve without --drafter";
    if (o.speed_up != null) return "inkling2: this Metal engine serves one Mac; --speed-up is unsupported";
    if (o.learn != null or o.learn_conversations) return "inkling2: learned disk states are not implemented; serve without --learn and --learn-conversations";
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0 or g > 1024 * 1024) return "inkling2: --prompt-cache-gib must be a finite, nonnegative budget";
    return null;
}

fn contextProblem(window: i64, fits: usize) ?[]const u8 {
    if (window < 1) return "inkling2: --context must be at least one token, and the selected context must fit the Metal state budget";
    if (window > fits) return "inkling2: --context exceeds the largest window this Mac can afford beside the Small EXL3 checkpoint";
    return null;
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusal(o)) |why| {
        problem.* = why;
        return null;
    }
    if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Small EXL3 engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();

    var ram: u64 = 0;
    if (@import("builtin").os.tag.isDarwin()) {
        var len: usize = @sizeOf(u64);
        if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    }
    const scratch: u64 = st.scratchBytes(st.batch_rows) + st.scratchBytes(st.prompt_rows);
    const resident = model.residentBytes() + scratch;
    const budget = cacheBudget(ram, model.device.maxWorkingSet(), @import("sys").getenv("INKLING2_LOAD_LIMIT_GB"), resident) catch {
        model.deinit();
        problem.* = "INKLING2_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 75% of RAM";
        return null;
    };
    const streams: u64 = @max(o.lanes, 1);
    const fits = longest(budget, model.config.text.max_context);
    const window: i64 = o.context orelse @intCast(fits);
    if (contextProblem(window, fits)) |why| {
        model.deinit();
        problem.* = why;
        return null;
    }
    std.log.info(
        "inkling2: {d:.1} GiB for stream and prompt state, context up to {d} tokens; {d} streams ({d:.2} GiB weights and scratch, {d:.2} GiB per stream)",
        .{ inGib(budget), window, streams, inGib(resident), inGib(st.cacheBytes(@intCast(window))) },
    );

    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.metal = try backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = @max(o.lanes, 1), .budget = budget });
    errdefer h.metal.deinit();
    try h.metal.warm();
    h.warm = .{ .queue = model.queue };
    const facts = h.metal.facts();
    h.config = try lanes.Config.init(gpa, facts, st.window_rows, st.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .lanes = o.lanes,
        .context_window = @intCast(window),
        .prefill_step = st.prompt_rows,
        .decode_share = o.decode_share,
        .structures = true,
        .logprobs = true,
    });
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else budget;
    if (h.cache_cap > 0) {
        const M = backend.Metal;
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = st.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{
            .ptr = h.metal,
            .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop },
        }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

fn inGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

pub fn cacheBudget(ram: u64, working_set: u64, text: ?[]const u8, resident: u64) error{BadLoadLimit}!u64 {
    var limit: u64 = @intFromFloat(@as(f64, @floatFromInt(ram)) * 0.75);
    const value = std.mem.trim(u8, text orelse "", " \t");
    if (value.len > 0) {
        const gb = std.fmt.parseFloat(f64, value) catch return error.BadLoadLimit;
        if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
        const cap = @min(ram, working_set);
        const bytes = gb * 1e9;
        limit = if (bytes >= @as(f64, @floatFromInt(cap))) cap else @intFromFloat(bytes);
    }
    const margin: u64 = 2 << 30;
    return if (limit > resident + margin) limit - resident - margin else 0;
}

pub fn longest(bytes: u64, most: u32) usize {
    if (st.cacheBytes(1) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (st.cacheBytes(mid) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = if (h.cache) |*owned| owned else return;
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

test "inkling2 small: native admission refuses unsupported flags before load" {
    var o: api.Open = .{ .dir = "checkpoint", .model_type = "inkling2" };
    try std.testing.expect(refusal(o) == null);
    o.vision = .{};
    try std.testing.expectEqualStrings("inkling2 EXL3 pack is text-only; serve without --vision", refusal(o).?);
    o.vision = null;
    o.cfg_enabled = true;
    try std.testing.expect(refusal(o) != null);
    o.cfg_enabled = false;
    o.loops = 3;
    try std.testing.expect(refusal(o) != null);
    o.loops = null;
    o.precision = .full;
    try std.testing.expect(refusal(o) != null);
    o.precision = null;
    o.drafter = "mtp";
    try std.testing.expect(refusal(o) != null);
    o.drafter = null;
    o.speed_up = "pair.json";
    try std.testing.expect(refusal(o) != null);
    o.speed_up = null;
    o.learn = "disk";
    try std.testing.expect(refusal(o) != null);
    o.learn = null;
    var why: []const u8 = "";
    o.vision = .{};
    try std.testing.expect((try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &why)) == null);
    try std.testing.expectEqualStrings("inkling2 EXL3 pack is text-only; serve without --vision", why);
}

test "inkling2 small: cache budget and longest context bounds" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(220_000_000_000 - 192 * gib, try cacheBudget(256 * gib, 240 * gib, "220", 190 * gib));
    try std.testing.expectError(error.BadLoadLimit, cacheBudget(256 * gib, 240 * gib, "0", 190 * gib));
    try std.testing.expectEqual(@as(usize, 0), longest(st.cacheBytes(1) - 1, 1024));
    try std.testing.expectEqual(@as(usize, 64), longest(st.cacheBytes(64), 1024));
}
