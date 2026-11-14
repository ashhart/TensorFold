//! The CUDA host's KV ledger for a family whose streams allocate per request (`Loaded.per_request`).
const std = @import("std");
const api = @import("engine_api");
const PerRequest = @import("core").kv_pool.PerRequest;
const Standing = api.kv_ledger.Standing;
const kv_pages = @import("cuda_kv_pool.zig");
const toGib = @import("cuda_family_memory.zig").toGib;
const Allocator = std.mem.Allocator;

/// What one request costs: fixed bytes plus a row's bytes for its prompt, its reply and the lookahead.
pub const Cost = struct {
    each: PerRequest,
    context: u64, // the request bound the server enforces (prompt + reply)

    pub fn of(c: *const Cost, prompt: u32, max_tokens: u32) u64 {
        const rows = @min(@as(u64, prompt) + max_tokens, c.context) + c.each.lookahead;
        return c.each.layout.fixed_bytes +| c.each.layout.token_bytes *| rows;
    }

    fn call(ctx: *anyopaque, prompt: u32, max_tokens: u32) u64 {
        const c: *const Cost = @ptrCast(@alignCast(ctx));
        return c.of(prompt, max_tokens);
    }
};

/// The ledger's total (its standing streams in it), the cache beside it, a full-window request's cost, any one's.
pub const Plan = struct { total: u64, kept: kv_pages.Kept, one_full: u64, cost: Cost, standing: Standing = .{} };

/// What the budget needs from load beyond the family.
pub const Inputs = struct { room: u64, context: i64, host_budget: u64, host_cache: bool };

/// Why no request fits: a full-window request's cost and the scratch it sits beside.
pub const Short = struct { one_full: u64, scratch: u64 };

/// The family's per-request declaration, or null: its streams take whole windows.
pub fn declared(loaded: anytype) ?PerRequest {
    return if (comptime @hasField(@TypeOf(loaded), "per_request")) loaded.per_request else null;
}

/// Whether this load serves from the ledger: the family allocates per request and nothing it loaded escapes the cost.
pub fn wanted(comptime F: type, loaded: anytype, served_stream_bytes: u64) bool {
    if (declared(loaded) == null) return false;
    // no path yet routes a warm pass's or a lone driver's LaneMemory back to the queue
    if (comptime @hasDecl(F, "warm_quiet")) return false;
    const L = @TypeOf(loaded);
    if (comptime @hasField(L, "lone")) if (loaded.lone != null) return false;
    if (comptime @hasField(L, "reserve")) return false; // streams made at load are whole windows already
    if (kv_pages.layout(loaded) != null) return false; // pages: the family admits through its own pool
    return served_stream_bytes == loaded.stream_bytes; // a drafter wrap adds bytes the declaration cannot see
}

/// The streams paid at load (`free_streams`: the engine's own sequence, in the model's bytes), up to --parallel.
fn prepaid(loaded: anytype, lanes: u32) u32 {
    return if (comptime @hasField(@TypeOf(loaded), "free_streams")) @min(loaded.free_streams, lanes) else 0;
}

/// The budget, or null with `short` set when one full-window request and the scratch do not fit the room.
pub fn plan(comptime F: type, loaded: anytype, o: api.Open, in: Inputs, short: *?Short) ?Plan {
    const each = declared(loaded).?;
    const cost: Cost = .{ .each = each, .context = @intCast(in.context) };
    const one_full = cost.of(@intCast(in.context), 0);
    // as the window plan's `held`: the cache beside --parallel full windows but the prepaid, each a standing lease
    const standing: Standing = .{ .each = one_full, .streams = prepaid(loaded, o.lanes) };
    if (each.scratch > in.room or (standing.streams == 0 and one_full > in.room - each.scratch)) {
        short.* = .{ .one_full = one_full, .scratch = each.scratch };
        return null;
    }
    const held = @as(u64, o.lanes - standing.streams) *| one_full;
    const kept = kv_pages.kept(F, loaded, null, o, in.room - each.scratch, held, in.host_budget, in.host_cache);
    const device: u64 = if (kept.host) 0 else kept.snaps * kept.snap_bytes + kept.copied;
    const total = (in.room - each.scratch - device) +| standing.each *| standing.streams;
    return .{ .total = total, .kept = kept, .one_full = one_full, .cost = cost, .standing = standing };
}

/// The lane host's hook over a plan that outlives the host (the CUDA Host keeps it).
pub fn info(p: ?*const Plan) ?api.Ledger {
    const x = p orelse return null;
    return .{ .ctx = @constCast(&x.cost), .total = x.total, .cost = Cost.call, .standing = x.standing };
}

/// The startup line's streams: each takes what its request needs from the budget.
pub fn text(a: Allocator, p: Plan, context: i64) ![]const u8 {
    const own = p.standing.each *| p.standing.streams;
    const beside = if (p.standing.streams > 0) " beside the engine's own window" else "";
    return std.fmt.allocPrint(a, "each reserving its prompt and reply from a {d:.2} GiB KV budget{s} ({d:.2} GiB for one at a {d}-token window)", .{ toGib(p.total - own), beside, toGib(p.one_full), context });
}

const Cached = struct {
    pub const snap_save = 1;
};
const sample: PerRequest = .{ .layout = .{ .fixed_bytes = 1000, .token_bytes = 10, .page_tokens = 0, .snap_bytes = 0 }, .lookahead = 8, .scratch = 400 };
var unused: u8 = 0;

fn fixture(per_request: ?PerRequest) struct { ctx: *anyopaque, stream_bytes: u64, per_request: ?PerRequest } {
    return .{ .ctx = &unused, .stream_bytes = 7, .per_request = per_request };
}

test "kv ledger cost: prompt and reply rows within the context, plus the lookahead" {
    const c: Cost = .{ .each = sample, .context = 4096 };
    try std.testing.expectEqual(@as(u64, 1000 + (100 + 8) * 10), c.of(100, 0));
    try std.testing.expectEqual(@as(u64, 1000 + (100 + 900 + 8) * 10), c.of(100, 900));
    try std.testing.expectEqual(@as(u64, 1000 + (4096 + 8) * 10), c.of(100, 100_000)); // past the context: capped
    try std.testing.expectEqual(c.of(4096, 0), c.of(1, 4095)); // every full-window request costs the same
}

test "kv ledger plan: the prompt cache takes only what --parallel full-window requests leave" {
    const o: api.Open = .{ .dir = "", .model_type = "f", .prompt_cache_gib = 1, .lanes = 4 };
    var short: ?Short = null;
    const full: u64 = 1000 + (4096 + 8) * 10;
    // a room past four full windows: the cache takes the rest, the ledger keeps all four
    const big = plan(Cached, fixture(sample), o, .{ .room = 1 << 30, .context = 4096, .host_budget = 0, .host_cache = false }, &short).?;
    try std.testing.expectEqual(full, big.one_full);
    try std.testing.expectEqual(@as(u64, (1 << 30) - 400 - 4 * full), big.kept.copied);
    try std.testing.expectEqual(4 * full, big.total);
    const l = info(&big).?;
    try std.testing.expectEqual(full, l.cost(l.ctx, 4096, 0));
    // a small room that holds two full windows: no cache, the ledger keeps the whole room
    const small = plan(Cached, fixture(sample), o, .{ .room = 400 + 2 * full + 5, .context = 4096, .host_budget = 0, .host_cache = false }, &short).?;
    try std.testing.expectEqual(@as(u64, 0), small.kept.copied);
    try std.testing.expectEqual(2 * full + 5, small.total);
}

test "kv ledger plan: a host prompt cache leaves the device budget whole" {
    const o: api.Open = .{ .dir = "", .model_type = "f", .prompt_cache_gib = 1, .lanes = 4 };
    var short: ?Short = null;
    const p = plan(Cached, fixture(sample), o, .{ .room = 1 << 20, .context = 4096, .host_budget = 1 << 30, .host_cache = true }, &short).?;
    try std.testing.expect(p.kept.host);
    try std.testing.expectEqual(@as(u64, (1 << 20) - 400), p.total);
}

test "kv ledger plan: a room that holds no full-window request beside the scratch is refused" {
    const o: api.Open = .{ .dir = "", .model_type = "f", .prompt_cache_gib = 0, .lanes = 4 };
    var short: ?Short = null;
    const full: u64 = 1000 + (4096 + 8) * 10;
    try std.testing.expectEqual(@as(?Plan, null), plan(Cached, fixture(sample), o, .{ .room = 400 + full - 1, .context = 4096, .host_budget = 0, .host_cache = false }, &short));
    try std.testing.expectEqual(full, short.?.one_full);
    try std.testing.expect(plan(Cached, fixture(sample), o, .{ .room = 400 + full, .context = 4096, .host_budget = 0, .host_cache = false }, &short) != null);
}

test "kv ledger: only a per-request family whose load nothing escapes serves from the ledger" {
    const None = struct {};
    try std.testing.expect(wanted(None, fixture(sample), 7));
    try std.testing.expect(!wanted(None, fixture(sample), 8)); // a drafter wrap's extra bytes
    try std.testing.expect(!wanted(None, fixture(null), 7)); // undeclared: whole windows
    try std.testing.expect(!wanted(None, .{ .ctx = @as(*anyopaque, &unused), .stream_bytes = @as(u64, 7) }, 7));
    const lone = .{ .ctx = @as(*anyopaque, &unused), .stream_bytes = @as(u64, 7), .per_request = @as(?PerRequest, sample), .lone = @as(?*anyopaque, &unused) };
    try std.testing.expect(!wanted(None, lone, 7));
    try std.testing.expect(!wanted(struct {
        pub const warm_quiet = 1;
    }, fixture(sample), 7));
}

test "kv ledger plan: a cost past u64 saturates and is refused, not wrapped" {
    const o: api.Open = .{ .dir = "", .model_type = "f", .prompt_cache_gib = 0, .lanes = 4 };
    const huge: PerRequest = .{ .layout = .{ .fixed_bytes = 1, .token_bytes = std.math.maxInt(u64) / 2, .page_tokens = 0, .snap_bytes = 0 }, .scratch = 400 };
    var short: ?Short = null;
    try std.testing.expectEqual(@as(?Plan, null), plan(Cached, fixture(huge), o, .{ .room = std.math.maxInt(u64), .context = 4096, .host_budget = 0, .host_cache = false }, &short));
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), short.?.one_full);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), short.?.one_full +| short.?.scratch); // as the refusal sums it
}

test "kv ledger plan: the prepaid stream is a standing lease, so the ledger counts the window plan's streams" {
    const o: api.Open = .{ .dir = "", .model_type = "f", .prompt_cache_gib = 1, .lanes = 4 };
    const full: u64 = 1000 + (4096 + 8) * 10;
    const Prepaid = struct { ctx: *anyopaque, stream_bytes: u64, per_request: ?PerRequest, free_streams: u32 };
    const own: Prepaid = .{ .ctx = &unused, .stream_bytes = 7, .per_request = sample, .free_streams = 1 };
    var short: ?Short = null;
    // a big room: the cache sits beside three full windows, the window plan's (streams - free), not four
    const big = plan(Cached, own, o, .{ .room = 1 << 30, .context = 4096, .host_budget = 0, .host_cache = false }, &short).?;
    try std.testing.expectEqual(@as(u64, (1 << 30) - 400 - 3 * full), big.kept.copied);
    try std.testing.expectEqual(Standing{ .each = full, .streams = 1 }, big.standing);
    try std.testing.expectEqual(4 * full, big.total);
    try std.testing.expectEqual(big.standing, info(&big).?.standing);
    // two full windows of room: the ledger holds three, as the window plan admits 1 + 2
    const room = 400 + 2 * full + 5;
    const small = plan(Cached, own, o, .{ .room = room, .context = 4096, .host_budget = 0, .host_cache = false }, &short).?;
    try std.testing.expectEqual(@as(u64, 3 * full + 5), small.total);
    try std.testing.expectEqual(@as(u32, 3), try @import("cuda_memory.zig").admit(room - 400, full, 1, 4, false));
    // less than one full window of room: the prepaid stream still serves one, as the window plan does
    try std.testing.expect(plan(Cached, own, o, .{ .room = 400 + full - 1, .context = 4096, .host_budget = 0, .host_cache = false }, &short) != null);
    var many = own;
    many.free_streams = 9; // never more standing streams than --parallel
    try std.testing.expectEqual(@as(u32, 4), plan(Cached, many, o, .{ .room = room, .context = 4096, .host_budget = 0, .host_cache = false }, &short).?.standing.streams);
}

test {
    _ = @import("cuda_ledger_lease_test.zig");
}
