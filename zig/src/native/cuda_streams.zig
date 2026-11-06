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
pub const Count = struct { streams: u32, paged: ?kv_pages.Plan, layout: ?kv_pool.Layout, ledger: ?ledger.Plan = null, slots: ?usize = null };

/// TF_KV_SHARE: the cache's part made pool pages, all of it unless the family keeps its states' rest elsewhere (GLM).
pub fn keptPpm(comptime F: type, share: bool) u32 {
    if (!share) return 0;
    return if (@hasDecl(F, "share_kept_ppm")) F.share_kept_ppm else 1_000_000;
}

/// TF_KV_SHARE's store opens over a cache to make pages of, or for a family keeping copies (GLM) when it keeps any.
pub fn opensStore(comptime F: type, share: bool, cache: u64, limit: usize) bool {
    return share and if (keptPpm(F, true) == 0) limit > 0 else cache > 0;
}

/// The family's streams in `b.room`; null with `problem` set when none fit or --parallel asks for more than fit.
pub fn count(comptime F: type, a: Allocator, loaded: anytype, o: api.Open, b: Budget, problem: *[]const u8) !?Count {
    const free = loaded.free_streams;
    const grow = budget.growRoom(b.room, !@hasField(F.Options, "lanes"));
    const wrapped = kv_pages.wrapped(F, loaded, free, b.stream_bytes, b.room);
    if (wrapped.layout) |l| {
        const slots: ?usize = if (b.share and l.snap_bytes > 0 and !b.host_cache) try @import("core").kv_prompts.sharing() else null; // GLM: a slot a state
        const paged = try kv_pages.plan(F, a, l, wrapped.room, o, @intCast(b.window), problem, b.host_cache, kv_pages.enginePaged(loaded), keptPpm(F, b.share), slots) orelse return null;
        return .{ .streams = paged.lanes, .paged = paged, .layout = l, .slots = slots };
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
