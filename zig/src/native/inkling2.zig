//! Text-only Inkling 2 Mini on Metal, with exact prompt reuse and budgeted stream state.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const lanes = @import("lanes");
const Model = @import("../families/inkling2/metal/model.zig").Model;
const st = @import("../families/inkling2/metal/state.zig");
const backend = @import("../families/inkling2/metal/backend.zig");
const dspark = @import("../families/inkling2/metal/dspark.zig");

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *Model,
    metal: *backend.Metal,
    drafter: ?*dspark.Drafter = null,
    drafted: ?lanes.drafted.Drafted = null,
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
        if (h.drafted) |*d| d.deinit();
        if (h.drafter) |d| d.deinit();
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.vision != null or o.cfg_enabled or o.cfg_batched) return "inkling2_loop: this Metal engine serves text only; images and audio are unsupported";
    if (o.loops != null) return "inkling2_loop: the model has exactly three core passes; --loops cannot override them";
    if (o.mtp_drafts != null or o.mtp_confidence != null) return "inkling2_loop: use --drafter for the checkpoint's DSpark head; external draft controls remain unsupported";
    if (o.precision != null) return "inkling2_loop: the checkpoint's NVFP4, MXFP8 and BF16 arithmetic is fixed; serve without --precision";
    if (o.speed_up != null) return "inkling2_loop: this Metal engine serves one Mac; --speed-up is unsupported";
    if (o.learn != null or o.learn_conversations) return "inkling2_loop: learned disk states are not implemented; serve without --learn and --learn-conversations";
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0 or g > 1024 * 1024) return "inkling2_loop: --prompt-cache-gib must be a finite, nonnegative budget";
    return null;
}

fn drafterDirectory(o: api.Open) ?[]const u8 {
    return if (o.drafts) o.drafter else null;
}

fn contextProblem(window: i64, fits: usize) ?[]const u8 {
    if (window < 1) return "inkling2_loop: --context must be at least one token, and the selected context must fit the Metal state budget";
    if (window > fits) return "inkling2_loop: --context exceeds the largest window this Mac can afford beside the checkpoint and DSpark state";
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
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    const drafter_dir = drafterDirectory(o);
    errdefer model.deinit();
    const drafter = if (drafter_dir) |dir| dspark.Drafter.init(gpa, io, model.device, model, dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "--drafter: Inkling 2 Mini DSpark cannot load {s} ({s})", .{ dir, @errorName(err) });
        model.deinit();
        return null;
    } else null;
    errdefer if (drafter) |d| d.deinit();
    var ram: u64 = 0;
    if (@import("builtin").os.tag.isDarwin()) {
        var len: usize = @sizeOf(u64);
        if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    }
    const scratch: u64 = st.scratchBytes(st.batch_rows) + st.scratchBytes(st.prompt_rows);
    const resident = model.checkpoint.residentBytes() + scratch + if (drafter) |d| d.fixedBytes() + dspark.Drafter.scoreBytes(0) else 0;
    const budget = cacheBudget(ram, model.device.maxWorkingSet(), @import("sys").getenv("INKLING2_LOAD_LIMIT_GB"), resident) catch {
        if (drafter) |d| d.deinit();
        model.deinit();
        problem.* = "INKLING2_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 75% of RAM";
        return null;
    };
    const streams: u64 = @max(o.lanes, 1);
    const draft_fixed = if (drafter != null) dspark.Drafter.streamBytes(0) + 127 * (dspark.Drafter.streamBytes(1) - dspark.Drafter.streamBytes(0)) else 0;
    const draft_token = if (drafter) |_| dspark.Drafter.streamBytes(1) - dspark.Drafter.streamBytes(0) else 0;
    const draft_shared_token = if (drafter) |_| dspark.Drafter.scoreBytes(1) - dspark.Drafter.scoreBytes(0) else 0;
    const fits = longest(budget, model.config.text.max_context, streams, draft_fixed, draft_token, draft_shared_token);
    const window: i64 = o.context orelse @intCast(fits);
    if (contextProblem(window, fits)) |why| {
        if (drafter) |d| d.deinit();
        model.deinit();
        problem.* = why;
        return null;
    }
    std.log.info("inkling2_loop: {d:.1} GiB for stream and prompt state, context up to {d} tokens", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), window });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.drafter = drafter;
    const reserved = if (drafter != null) streams * dspark.Drafter.streamBytes(@intCast(window)) + dspark.Drafter.scoreBytes(@intCast(window)) - dspark.Drafter.scoreBytes(0) else 0;
    if (reserved > budget) return error.DSparkReservationExceeded;
    h.metal = try backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = @max(o.lanes, 1), .budget = budget - reserved });
    errdefer h.metal.deinit();
    try h.metal.warm();
    h.drafted = null;
    h.warm = .{ .queue = model.queue };
    if (drafter) |d| {
        try d.setContext(@intCast(window));
        try d.measure();
        h.drafted = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), d.drafter());
        h.drafted.?.first_rows = true;
    }
    errdefer if (h.drafted) |*d| d.deinit();
    const facts = if (h.drafted) |*d| d.facts(h.metal.facts()) else h.metal.facts();
    h.config = try lanes.Config.init(gpa, facts, st.window_rows, st.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.drafted) |*d| d.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = st.prompt_rows, .decode_share = o.decode_share, .structures = true, .logprobs = true, .draft_sources = .{ .drafter = if (drafter_dir) |dir| .{ .kind = "DSpark", .name = o.drafter_name orelse dir } else null } });
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else budget;
    if (h.cache_cap > 0) {
        const M = backend.Metal;
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = st.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop } }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

fn cacheBudget(ram: u64, working_set: u64, text: ?[]const u8, resident: u64) error{BadLoadLimit}!u64 {
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

fn longest(bytes: u64, most: u32, streams: u64, draft_fixed: u64, draft_token: u64, draft_shared_token: u64) usize {
    const one: u64 = st.cacheBytes(1) + streams * (draft_fixed + draft_token) + draft_shared_token;
    if (one > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        const need = st.cacheBytes(mid) + streams * (draft_fixed + draft_token * @as(u64, @intCast(mid))) + draft_shared_token * @as(u64, @intCast(mid));
        if (need <= bytes) lo = mid else hi = mid - 1;
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

test "Inkling native admission keeps DSpark opt-in and refuses learned modes" {
    var o: api.Open = .{ .dir = "checkpoint", .model_type = "inkling2_loop" };
    try std.testing.expect(refusal(o) == null);
    try std.testing.expect(drafterDirectory(o) == null);
    o.drafter = "checkpoint";
    try std.testing.expectEqualStrings("checkpoint", drafterDirectory(o).?);
    o.drafts = false;
    try std.testing.expect(drafterDirectory(o) == null);
    o.speed_up = "distributed";
    try std.testing.expect(refusal(o) != null);
    o.speed_up = null;
    o.learn = "learned";
    try std.testing.expect(refusal(o) != null);
    try std.testing.expect(contextProblem(0, 10) != null);
    try std.testing.expect(contextProblem(-1, 10) != null);
    try std.testing.expect(contextProblem(11, 10) != null);
    try std.testing.expect(contextProblem(10, 10) == null);
}

test "Inkling native open publishes the exact admission refusal" {
    var problem: []const u8 = "not published";
    const o: api.Open = .{ .dir = "unused", .model_type = "inkling2_loop", .learn_conversations = true };
    try std.testing.expect((try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem)) == null);
    try std.testing.expectEqualStrings("inkling2_loop: learned disk states are not implemented; serve without --learn and --learn-conversations", problem);
}

test "Inkling native open releases DSpark when the later context admission refuses (TF_INKLING2_METAL_DIR, TF_INKLING2_DSPARK_DIR)" {
    const dir = @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse return error.SkipZigTest;
    const drafter_dir = @import("sys").getenv("TF_INKLING2_DSPARK_DIR") orelse return error.SkipZigTest;
    var problem: []const u8 = "not published";
    var o: api.Open = .{ .dir = dir, .model_type = "inkling2_loop", .context = 0, .drafter = drafter_dir };
    try std.testing.expect((try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem)) == null);
    try std.testing.expectEqualStrings("inkling2_loop: --context must be at least one token, and the selected context must fit the Metal state budget", problem);
    o.context = std.math.maxInt(i64);
    problem = "not published";
    try std.testing.expect((try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem)) == null);
    try std.testing.expectEqualStrings("inkling2_loop: --context exceeds the largest window this Mac can afford beside the checkpoint and DSpark state", problem);
    // Null skips errdefer; omitting explicit deinit leaks allocations caught by std.testing.allocator.
}

test "Inkling native open unwinds DSpark when the later Host allocation fails (TF_INKLING2_METAL_DIR, TF_INKLING2_DSPARK_DIR)" {
    const Fault = struct {
        drafter_allocated: bool = false,
        host_failed: bool = false,

        fn allocator(f: *@This()) std.mem.Allocator {
            return .{ .ptr = f, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
        }
        fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            if (f.drafter_allocated and len == @sizeOf(Host)) {
                f.host_failed = true;
                return null;
            }
            const memory = std.testing.allocator.rawAlloc(len, alignment, ra) orelse return null;
            if (len == @sizeOf(dspark.Drafter)) f.drafter_allocated = true;
            return memory;
        }
        fn resize(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) bool {
            return std.testing.allocator.rawResize(memory, alignment, len, ra);
        }
        fn remap(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
            return std.testing.allocator.rawRemap(memory, alignment, len, ra);
        }
        fn free(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
            std.testing.allocator.rawFree(memory, alignment, ra);
        }
    };
    const dir = @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse return error.SkipZigTest;
    const drafter_dir = @import("sys").getenv("TF_INKLING2_DSPARK_DIR") orelse return error.SkipZigTest;
    var fault: Fault = .{};
    var problem: []const u8 = "";
    const o: api.Open = .{ .dir = dir, .model_type = "inkling2_loop", .context = 1, .lanes = 1, .drafter = drafter_dir };
    try std.testing.expectError(error.OutOfMemory, open(std.testing.allocator, fault.allocator(), std.testing.io, o, &problem));
    try std.testing.expect(fault.drafter_allocated and fault.host_failed);
    // Removing errdefer leaks DSpark and its weights in std.testing.allocator.
}

test "Inkling native load limit clamps before integer conversion" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(8_000_000_000 - 5 * gib, try cacheBudget(16 * gib, 14 * gib, "8", 3 * gib));
    try std.testing.expectEqual(9 * gib, try cacheBudget(16 * gib, 14 * gib, "1e30", 3 * gib));
    try std.testing.expectEqual(0, try cacheBudget(4 * gib, 4 * gib, null, 3 * gib));
    for ([_][]const u8{ "0", "-1", "nan", "inf", "bad" }) |text| {
        try std.testing.expectError(error.BadLoadLimit, cacheBudget(16 * gib, 14 * gib, text, 3 * gib));
    }
}

test "Inkling native longest context accounts target, streams and shared draft scores" {
    const budget = st.cacheBytes(7) + 2 * (11 + 7 * 13) + 7 * 17;
    try std.testing.expectEqual(@as(usize, 7), longest(budget, 8, 2, 11, 13, 17));
    try std.testing.expectEqual(@as(usize, 0), longest(st.cacheBytes(1) - 1, 8, 1, 0, 0, 0));
    const per_token = dspark.Drafter.streamBytes(1) - dspark.Drafter.streamBytes(0);
    const conservative = dspark.Drafter.streamBytes(0) + 127 * per_token + 129 * per_token;
    try std.testing.expect(conservative >= dspark.Drafter.streamBytes(129));
}
