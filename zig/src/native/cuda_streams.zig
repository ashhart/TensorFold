//! How many streams the CUDA host serves at load: the page plan's lanes, or the whole windows its room holds.
const std = @import("std");
const api = @import("engine_api");
const kv_pool = @import("core").kv_pool;
const budget = @import("cuda_memory.zig");
const kv_pages = @import("cuda_kv_pool.zig");
const toGib = @import("cuda_family_memory.zig").toGib;
const ledger = @import("cuda_ledger.zig");
const Allocator = std.mem.Allocator;

/// The admit's refusals: no stream fits, or --parallel asks for more than the budget holds.
const no_stream = "the CUDA memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, " ++
    "and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve{s}; " ++
    "lower --context, or free device memory";
const too_many = "--parallel {d} needs {d:.2} GiB for its streams at a {d}-token window, " ++
    "and the CUDA memory budget leaves {d:.2} GiB{s}: serve --parallel {d}, or lower --context";

/// What load knows when it counts streams.
pub const Budget = struct {
    room: u64,
    model: u64,
    reserve: u64,
    low: []const u8, // names the rank whose room bounds the streams, when it is not this one
    window: i64, // the context window (cuda_window.zig)
    stream_bytes: u64,
    host_cache: bool,
    host_budget: u64, // a host prompt cache's bytes (0: none)
    share: bool, // TF_KV_SHARE: the prompt cache's kept states are pool pages
};

/// The streams served, the page plan and layout or the KV ledger plan, and TF_KV_SHARE's count when states take slots.
pub const Count = struct { streams: u32, paged: ?kv_pages.Plan, layout: ?kv_pool.Layout, ledger: ?ledger.Plan = null, slots: ?usize = null, planned: ?kv_pool.Budget = null };

pub const CacheRoom = struct { room: u64, streams: u64 };

/// A host cache sits beside every admitted full window, including a request ledger's declared scratch.
pub fn cacheRoom(room: u64, stream_bytes: u64, free: u32, counted: Count) CacheRoom {
    if (counted.ledger) |p| return .{
        .room = room -| p.cost.each.scratch,
        .streams = p.one_full *| (counted.streams -| p.standing.streams),
    };
    return .{ .room = room, .streams = stream_bytes *| (counted.streams -| free) };
}

/// TF_KV_SHARE: the cache's part made pool pages, all of it unless the family keeps its states' rest elsewhere (GLM).
pub fn keptPpm(comptime F: type, share: bool) u32 {
    if (!share) return 0;
    return if (@hasDecl(F, "share_kept_ppm")) F.share_kept_ppm else 1_000_000;
}

/// TF_KV_SHARE's store opens over a cache to make pages of, or for a family keeping copies (GLM) when it keeps any.
pub fn opensStore(comptime F: type, share: bool, cache: u64, limit: usize) bool {
    return share and if (keptPpm(F, true) == 0) limit > 0 else cache > 0;
}

/// TF_KV_SHARE's count when its states take the plan's slots: share on, a layout slot, and the cache on the device.
pub fn shareSlots(share: bool, snap_bytes: u64, host_cache: bool) !?usize {
    return if (share and snap_bytes > 0 and !host_cache) try @import("core").kv_prompts.sharing() else null;
}

/// The family's streams in `b.room`; null with `problem` set when none fit or --parallel asks for more than fit.
pub fn count(comptime F: type, a: Allocator, loaded: anytype, o: api.Open, b: Budget, problem: *[]const u8) !?Count {
    const free = loaded.free_streams;
    const grow = budget.growRoom(b.room, !@hasField(F.Options, "lanes"));
    const wrapped = kv_pages.wrapped(F, loaded, free, b.stream_bytes, b.room);
    if (wrapped.layout) |l| {
        const slots = try shareSlots(b.share, l.snap_bytes, b.host_cache); // GLM: a slot a state
        const paged = try kv_pages.plan(F, a, l, wrapped.room, o, @intCast(b.window), problem, b.host_cache, kv_pages.enginePaged(loaded), keptPpm(F, b.share), slots) orelse return null;
        const planned = kv_pages.budgetOf(F, l, wrapped.room, o, @intCast(b.window), b.host_cache, kv_pages.enginePaged(loaded), keptPpm(F, b.share), slots);
        return .{ .streams = paged.lanes, .paged = paged, .layout = l, .slots = slots, .planned = if (slots != null) planned else null };
    }
    // --parallel caps the streams; each request reserves its own bytes (a family that allocates per request)
    if (ledger.wanted(F, loaded, b.stream_bytes)) {
        var short: ?ledger.Short = null;
        const p = ledger.plan(F, loaded, o, .{ .room = b.room, .context = b.window, .host_budget = b.host_budget, .host_cache = b.host_cache }, &short) orelse {
            problem.* = try std.fmt.allocPrint(a, no_stream, .{ b.window, toGib(short.?.one_full +| short.?.scratch), toGib(b.room), toGib(b.model), toGib(b.reserve), b.low });
            return null;
        };
        return .{ .streams = o.lanes, .paged = null, .layout = null, .ledger = p };
    }
    const streams = budget.admit(grow, b.stream_bytes, free, o.lanes, o.lanes_fixed) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, no_stream, .{
                b.window,
                toGib(b.stream_bytes),
                toGib(b.room),
                toGib(b.model),
                toGib(b.reserve),
                b.low,
            }),
            error.TooMany => try std.fmt.allocPrint(a, too_many, .{
                o.lanes,
                toGib(b.stream_bytes * (o.lanes -| free)),
                b.window,
                toGib(b.room),
                b.low,
                free + grow / b.stream_bytes,
            }),
        };
        return null;
    };
    return .{ .streams = streams, .paged = null, .layout = null };
}

test "TF_KV_SHARE makes the whole cache pool pages and opens on a cache, unless the family keeps copies in slots (GLM)" {
    const Paged = struct {};
    const Copies = struct {
        pub const share_kept_ppm: u32 = 0;
    };
    try std.testing.expectEqual(@as(u32, 1_000_000), keptPpm(Paged, true));
    try std.testing.expectEqual(@as(u32, 0), keptPpm(Copies, true));
    try std.testing.expectEqual(@as(u32, 0), keptPpm(Paged, false));
    try std.testing.expect(opensStore(Paged, true, 1, 0) and !opensStore(Paged, true, 0, 0) and !opensStore(Paged, false, 1, 0));
    try std.testing.expect(opensStore(Copies, true, 0, 1) and !opensStore(Copies, true, 1 << 30, 0) and !opensStore(Copies, false, 0, 1));
}

test "TF_KV_SHARE's states take slots only with share on, a layout slot and the cache on the device" {
    try std.testing.expect(try shareSlots(true, 1, false) != null);
    try std.testing.expect(try shareSlots(false, 1, false) == null);
    try std.testing.expect(try shareSlots(true, 0, false) == null);
    try std.testing.expect(try shareSlots(true, 1, true) == null);
}

test "TF_KV_SHARE unset or empty plans as before: no slot, and GLM's copies take what the windows leave" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const env = @import("sys").env;
    const name = "TF_KV_SHARE";
    const before = if (std.c.getenv(name)) |v| try std.testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (before) |v| {
            _ = env.setenv(name, v, 1);
            std.testing.allocator.free(v);
        } else _ = env.unsetenv(name);
    }
    const Glm = struct {
        pub const share_kept_ppm: u32 = 0;
        pub const window_rows = 8;
    };
    const mib: u64 = 1 << 20;
    // GLM at TP=3 without the share (327 MiB fixed, no slot), --parallel 8, a 1M window, Jammed XSFP4's 42.0 GiB
    const l: kv_pool.Layout = .{ .fixed_bytes = 327 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 8, .lanes_fixed = true };
    for ([_]?[:0]const u8{ null, "", " " }) |v| {
        _ = if (v) |s| env.setenv(name, s, 1) else env.unsetenv(name);
        const share = try @import("core").kv_prompts.sharing() > 0;
        const slots = try shareSlots(share, l.snap_bytes, false);
        try std.testing.expectEqual(@as(?usize, null), slots);
        const b = kv_pages.budgetOf(Glm, l, (41 << 30) + 992 * mib, o, 1_048_576, false, true, keptPpm(Glm, share), slots);
        try std.testing.expect(!b.slots_only);
        try std.testing.expectEqual(kv_pool.Plan{ .lanes = 8, .snaps = 0, .copied = 1572732928, .pages = 6147 }, try kv_pool.plan(l, b));
    }
}

test "CUDA cache default uses the declared request ledger windows and scratch" {
    const GiB: u64 = 1 << 30;
    var counted: Count = .{ .streams = 8, .paged = null, .layout = null };
    const whole = cacheRoom(88 * GiB, 5 * GiB, 1, counted);
    try std.testing.expectEqual(@as(u64, 35 * GiB), whole.streams);
    counted.ledger = .{
        .total = 88 * GiB,
        .kept = .{ .snaps = 0, .copied = 0, .snap_bytes = 0 },
        .one_full = 6 * GiB,
        .standing = .{ .streams = 1, .each = 6 * GiB },
        .cost = .{ .context = 1, .each = .{ .layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 1, .snap_bytes = 0 }, .scratch = 2 * GiB } },
    };
    const actual = cacheRoom(88 * GiB, 5 * GiB, 1, counted);
    try std.testing.expectEqual(@as(u64, 86 * GiB), actual.room);
    try std.testing.expectEqual(@as(u64, 42 * GiB), actual.streams);
    try std.testing.expectEqual(@as(u64, 22 * GiB), budget.hostCacheBudget(null, 96 * GiB, actual.room, actual.streams, true));
}
