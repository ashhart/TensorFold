//! The native server's KV page pool: a paging family's plan and pages, and its cache split into slots and copied rows.
const std = @import("std");
const api = @import("engine_api");
const kv_pool = @import("core").kv_pool;
const budget = @import("cuda_memory.zig");
const Allocator = std.mem.Allocator;

pub const Plan = kv_pool.Plan;

/// A family whose other streams take pages (Nemotron's) names their layout; any other allocates a window each (null).
pub fn layout(loaded: anytype) ?kv_pool.Layout {
    return if (comptime @hasField(@TypeOf(loaded), "layout")) loaded.layout else null;
}

/// Whether the engine's own sequence takes pages too (GLM's when it pages): the plan sets a window of them aside first.
pub fn enginePaged(loaded: anytype) bool {
    return if (comptime @hasField(@TypeOf(loaded), "engine_paged")) loaded.engine_paged else false;
}

/// The real native startup room before page planning and cache allocation.
pub fn startupRoom(comptime F: type, a: Allocator, loaded: anytype, o: api.Open, available: u64, host_budget: u64, problem: *[]const u8) !?u64 {
    const memory = @import("cuda_family_memory.zig");
    const left = available -| memory.learnBytes(loaded);
    const staging = stagingBytes(F, loaded, hostBudget(o, host_budget));
    if (staging > left) {
        problem.* = try std.fmt.allocPrint(a, "host snapshot staging needs {d} device bytes but the CUDA memory budget leaves {d}; lower the model/context or disable the prompt cache", .{ staging, left });
        return null;
    }
    return try memory.room(F, loaded.ctx, left - staging);
}

/// Whether prompt states go to host memory: a host-only family's always, a placing one's when the cache won't fit.
pub fn placeHost(comptime F: type, a: Allocator, loaded: anytype, o: api.Open, available: u64, window: u64, stream_bytes: u64, problem: *[]const u8) !?bool {
    if (!host(F, loaded)) return false;
    if (comptime !@hasDecl(F, "snap_place")) return true;
    if (comptime @hasDecl(F, "rankRoom")) @compileError("a second startupRoom would agree the room across ranks twice");
    const room = try startupRoom(F, a, loaded, o, available, 0, problem) orelse return null;
    return !fits(F, loaded, o, room, window, stream_bytes);
}

/// The asked cache (16 GiB unless --prompt-cache-gib says) whole beside the streams, by the startup plan's arithmetic.
fn fits(comptime F: type, loaded: anytype, o: api.Open, room: u64, window: u64, stream_bytes: u64) bool {
    const want = budget.cacheBytes(o.prompt_cache_gib, std.math.maxInt(u64), 0);
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    if (layout(loaded)) |l| return kv_pool.holds(l, .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = named(o), .window = window + rows, .engine_paged = enginePaged(loaded) });
    const n = budget.admit(room, stream_bytes, loaded.free_streams, o.lanes, o.lanes_fixed) catch return false;
    return room -| stream_bytes * (n -| loaded.free_streams) >= want;
}

/// The pool beside the engine's own sequence (in `model`: a window, or on pages its fixed part); null: see `problem`.
pub fn plan(comptime F: type, a: Allocator, l: kv_pool.Layout, room: u64, o: api.Open, window: u64, problem: *[]const u8, host_mode: bool, engine_paged: bool) !?Plan {
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    const want = if (host_mode) 0 else budget.cacheBytes(o.prompt_cache_gib, std.math.maxInt(u64), 0);
    return kv_pool.plan(l, .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = named(o), .window = window + rows, .engine_paged = engine_paged }) catch |e| {
        problem.* = switch (e) {
            error.NoStream => if (engine_paged) try std.fmt.allocPrint(a, "the CUDA memory budget fits no stream beside the model: one at a {d}-token window takes {d:.2} GiB of KV pages, and the budget leaves {d:.2} GiB", .{ window, toGib(l.pages(window + rows) * l.pageBytes()), toGib(room) }) else "the CUDA memory budget fits no stream beside the model",
            error.TooMany => try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its other streams' fixed parts, state runs and a page each, and the CUDA memory budget leaves {d:.2} GiB: serve fewer, or free device memory", .{ o.lanes, toGib((o.lanes - 1) * (l.fixed_bytes + (1 + l.runPages()) * l.pageBytes())), toGib(room) }),
        };
        return null;
    };
}

/// The planned streams' sequences and pages made now; false: `problem` says why they could not be.
pub fn attach(comptime F: type, a: Allocator, ctx: *anyopaque, paged: ?Plan, problem: *[]const u8) !bool {
    if (comptime !@hasDecl(F, "attach")) return true;
    const p = paged orelse return true;
    F.attach(ctx, p.lanes - 1, p.pages) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA engine could not make its {d} streams' pages ({s})", .{ p.lanes - 1, @errorName(e) });
        return false;
    };
    return true;
}

/// The cache's snapshot slots and copied rows' bytes (`snaps` 0: a family without slots keeps its cache whole).
pub const Kept = struct { snaps: usize, copied: u64, snap_bytes: u64, host: bool = false, most: usize = std.math.maxInt(usize) };

pub fn kept(comptime F: type, loaded: anytype, paged: ?Plan, o: api.Open, room: u64, held: u64, host_budget: u64, host_mode: bool) Kept {
    if (host_mode) return .{ .snaps = 0, .copied = hostBudget(o, host_budget), .snap_bytes = stagingBytes(F, loaded, hostBudget(o, host_budget)), .host = true, .most = named(o) orelse std.math.maxInt(usize) };
    // on pages the plan set the cache aside already; `held` counts every stream at a whole window
    if (comptime !@hasDecl(F, "snap_slots")) return .{ .snaps = 0, .copied = if (!@hasDecl(F, "snap_save")) 0 else if (paged) |p| p.copied else budget.cacheBytes(o.prompt_cache_gib, room, held), .snap_bytes = 0 };
    const b: u64 = loaded.snap_bytes;
    const s = if (paged) |p| kv_pool.split(p.snaps * b + p.copied, b, p.snaps) else kv_pool.split(budget.cacheBytes(o.prompt_cache_gib, room, held), b, named(o));
    return .{ .snaps = s.snaps, .copied = s.copied, .snap_bytes = b };
}

/// The store keeps a state a slot, the slots made now in one device buffer; a family without them, its own budget.
pub fn store(comptime F: type, s: *api.prompt_cache.Store, target: anytype, k: Kept) !void {
    if (comptime !@hasDecl(F, "snap_slots")) return;
    if (comptime @hasDecl(F, "snap_place")) F.snap_place(target, k.host);
    s.most = if (k.host) k.most else k.snaps;
    _ = try F.snap_slots(target, k.snaps);
}

/// The startup line's streams and cache.
pub fn text(comptime F: type, a: Allocator, stream_bytes: u64, l: ?kv_pool.Layout, paged: ?Plan, engine_paged: bool, window: u64, k: Kept, kv8: bool) ![2][]const u8 {
    const shape = if (paged) |p| (if (engine_paged) try std.fmt.allocPrint(a, "the engine's own and {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them (enough for one at a {d}-token window), {d:.0} MiB fixed each", .{ p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), window, @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) }) else try std.fmt.allocPrint(a, "the engine's own at a {d}-token window ({d:.2} GiB, in the model's), {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them, {d:.0} MiB fixed each", .{ window, toGib(stream_bytes), p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) })) else try std.fmt.allocPrint(a, "{d:.2} GiB each at a {d}-token window{s}", .{ toGib(stream_bytes), window, if (kv8) " (fp8 caches)" else "" });
    const placed = comptime @hasDecl(F, "snap_place"); // the startup line says where its prompt states went, and why
    const cache = if (k.copied == 0) ", no prompt cache" else if (k.host) try std.fmt.allocPrint(a, ", {s}a {d:.1} GiB prompt cache in host memory with {d:.0} MiB of device staging", .{ if (placed) "prompt states on host, the cache doesn't fit on the card: " else "", toGib(k.copied), @as(f64, @floatFromInt(k.snap_bytes)) / (1 << 20) }) else if (comptime @hasDecl(F, "snap_slots")) try std.fmt.allocPrint(a, ", {s}a prompt cache of {d} snapshot slots ({d:.1} GiB) and {d:.1} GiB of copied rows", .{ if (placed) "prompt states on device: " else "", k.snaps, toGib(k.snaps * k.snap_bytes), toGib(k.copied) }) else try std.fmt.allocPrint(a, ", a {d:.1} GiB prompt cache", .{toGib(k.copied)});
    return .{ shape, cache };
}

fn named(o: api.Open) ?usize {
    return if (o.keep) |k| @intCast(@max(k, 0)) else null;
}

fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

test {
    std.testing.refAllDecls(@This());
}

/// Runtime storage policy from the loaded engine; static host-only families also keep their established policy.
pub fn host(comptime F: type, loaded: anytype) bool {
    if (comptime @hasField(@TypeOf(loaded), "host_snaps")) return loaded.host_snaps;
    return @hasDecl(F, "snap_host");
}
pub fn stagingBytes(comptime F: type, loaded: anytype, host_budget: u64) u64 {
    if (comptime !@hasDecl(F, "snap_slots")) return 0;
    return if (host(F, loaded) and host_budget > 0) loaded.snap_bytes else 0;
}

const Fixture = struct {
    pub const snap_slots = 1;
    pub const snap_save = 1;
};
test "kv merge: a host cache spends no device snapshot slots or copied-row budget" {
    const loaded = .{ .host_snaps = true, .snap_bytes = @as(u64, 8 << 20) };
    const o: api.Open = .{ .dir = "", .model_type = "n", .prompt_cache_gib = 1, .lanes = 4 };
    const k = kept(Fixture, loaded, null, o, 256 << 20, 0, 1 << 30, host(Fixture, loaded));
    try std.testing.expectEqual(@as(usize, 0), k.snaps);
    try std.testing.expectEqual(@as(u64, 1 << 30), k.copied);
}
const Placing = struct {
    pub const snap_slots = 1;
    pub const snap_save = 1;
    pub const snap_place = 1;
};
test "kv placement: a discrete card keeps prompt states when the asked cache fits beside the streams, else the host does" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var ctx: u8 = 0;
    const l: kv_pool.Layout = .{ .fixed_bytes = 64 << 20, .token_bytes = 7 << 10, .page_tokens = 512, .snap_bytes = 48 << 20 };
    const card = .{ .ctx = @as(*anyopaque, &ctx), .host_snaps = true, .snap_bytes = @as(u64, 48 << 20), .layout = @as(?kv_pool.Layout, l), .free_streams = @as(u32, 1) };
    const o: api.Open = .{ .dir = "", .model_type = "n", .lanes = 4 };
    try std.testing.expectEqual(@as(?bool, false), try placeHost(Placing, a, card, o, 70 << 30, 70000, 0, &problem));
    try std.testing.expectEqual(@as(?bool, true), try placeHost(Placing, a, card, o, 12 << 30, 70000, 0, &problem));
    const asked: api.Open = .{ .dir = "", .model_type = "n", .lanes = 4, .prompt_cache_gib = 4 };
    try std.testing.expectEqual(@as(?bool, false), try placeHost(Placing, a, card, asked, 12 << 30, 70000, 0, &problem));
    const shared = .{ .ctx = @as(*anyopaque, &ctx), .host_snaps = false, .snap_bytes = @as(u64, 48 << 20), .layout = @as(?kv_pool.Layout, l), .free_streams = @as(u32, 1) };
    try std.testing.expectEqual(@as(?bool, false), try placeHost(Placing, a, shared, o, 12 << 30, 70000, 0, &problem));
    try std.testing.expectEqual(@as(?bool, true), try placeHost(Fixture, a, card, o, 70 << 30, 70000, 0, &problem));
    const on_host = (try text(Placing, a, 1 << 30, l, null, false, 70000, .{ .snaps = 0, .copied = 8 << 30, .snap_bytes = 48 << 20, .host = true }, false))[1];
    try std.testing.expect(std.mem.startsWith(u8, on_host, ", prompt states on host, the cache doesn't fit on the card: a 8.0 GiB"));
    const on_device = (try text(Placing, a, 1 << 30, l, null, false, 70000, .{ .snaps = 64, .copied = 13 << 30, .snap_bytes = 48 << 20 }, false))[1];
    try std.testing.expect(std.mem.startsWith(u8, on_device, ", prompt states on device: a prompt cache of 64 snapshot slots"));
}
const Copies = struct {
    pub const snap_save = 1;
};
test "kv pages: a family without snapshot slots keeps the paged plan's cache, not what window-sized streams leave" {
    const l: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const p = try kv_pool.plan(l, .{ .room = 40 << 30, .engine = 0, .lanes = 4, .cache = 16 << 30 });
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 4 };
    const k = kept(Copies, .{}, p, o, 40 << 30, 3 * (12 << 30), 0, false); // held: three streams at a 1M window
    try std.testing.expectEqual(@as(u64, 16 << 30), k.copied);
    try std.testing.expectEqual(@as(usize, 0), k.snaps);
    try std.testing.expectEqual(@as(u64, 4 << 30), kept(Copies, .{}, null, o, 40 << 30, 3 * (12 << 30), 0, false).copied);
}
const Owned = struct {
    pub const snap_save = 1;
    pub const window_rows = 8;
};
test "kv pages: an engine whose own rows are pages plans a window of them first, refuses a room under it, and says so" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const l: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 4, .prompt_cache_gib = 4 };
    try std.testing.expect(enginePaged(.{ .layout = @as(?kv_pool.Layout, l), .engine_paged = true }) and !enginePaged(.{ .layout = @as(?kv_pool.Layout, l) }));
    const p = (try plan(Owned, a, l, 40 << 30, o, 8192, &problem, false, true)).?;
    try std.testing.expectEqual(@as(u32, 4), p.lanes);
    try std.testing.expectEqual(@as(usize, 4 * 17), p.pages); // every stream at most 8,192 rows and a verify window: 17 pages
    const floor = 17 * l.pageBytes();
    try std.testing.expectEqual(@as(?Plan, null), try plan(Owned, a, l, floor - 1, o, 8192, &problem, false, true));
    try std.testing.expect(std.mem.indexOf(u8, problem, "8192-token window") != null);
    const shown = (try text(Owned, a, 180 << 20, l, p, true, 8192, .{ .snaps = 0, .copied = 4 << 30, .snap_bytes = 0 }, false))[0];
    try std.testing.expectEqualStrings("the engine's own and 3 more taking 68 pages of 512 rows (0.42 GiB) as they need them (enough for one at a 8192-token window), 180 MiB fixed each", shown);
    // the asked cache holds past the window's pages and a fixed part and page for each other stream, not a byte less
    const each = (180 << 20) + l.pageBytes();
    const loaded = .{ .layout = @as(?kv_pool.Layout, l), .engine_paged = true, .free_streams = @as(u32, 1) };
    try std.testing.expect(fits(Owned, loaded, o, floor + 3 * each + (4 << 30), 8192, 0));
    try std.testing.expect(!fits(Owned, loaded, o, floor + 3 * each + (4 << 30) - 1, 8192, 0));
}
test "kv merge: host armed-state staging is reserved once before planning the page pool" {
    const loaded = .{ .host_snaps = true, .snap_bytes = @as(u64, 8 << 20) };
    try std.testing.expectEqual(@as(u64, 8 << 20), stagingBytes(Fixture, loaded, 1 << 30));
    try std.testing.expectEqual(@as(u64, 0), stagingBytes(Fixture, loaded, 0));
}

/// Close in the host's established order: stop the store before destroying its disk rows.
pub fn closeStore(gpa: Allocator, saved: *api.prompt_cache.Store) void {
    const disk = saved.disk;
    saved.deinit();
    gpa.destroy(saved);
    if (disk) |rows| rows.deinit();
}

test "kv safety: native planning refuses insufficient staging and allocates at or above the room" {
    try @import("cuda_kv_safety_test.zig").run(@This(), false);
}
test "kv safety: keep zero reserves and allocates no staging through native planning" {
    try @import("cuda_kv_safety_test.zig").run(@This(), true);
}

fn hostBudget(o: api.Open, bytes: u64) u64 {
    return if (named(o) == 0) 0 else bytes;
}

/// Startup facts use the actual family planner without changing any budget, store or allocation.
pub fn capacity(comptime F: type, loaded: anytype, k: Kept, window: u64) ?api.PromptCacheCapacity {
    if (comptime !@hasDecl(F, "snap_min_bytes")) return null;
    if (window <= F.snap_grid) return null;
    const minimum = F.snap_min_bytes(loaded.target);
    if (minimum == 0 or minimum == std.math.maxInt(u64)) return null;
    return .{ .budget_bytes = k.copied, .minimum_state_bytes = minimum };
}

pub fn warnCapacity(facts: ?api.PromptCacheCapacity, disabled: bool, comptime log: type) void {
    const c = facts orelse return;
    if (disabled or c.budget_bytes >= c.minimum_state_bytes) return;
    log.warn("GLM CUDA prompt cache budget {d} bytes ({d:.3} GiB) cannot keep one state: the minimum planned state needs {d} bytes ({d:.3} GiB); adjust --context, --parallel, --prompt-cache-gib or TENSORFOLD_MEMORY_RESERVE_GIB to make room", .{ c.budget_bytes, toGib(c.budget_bytes), c.minimum_state_bytes, toGib(c.minimum_state_bytes) });
}

test {
    _ = @import("cuda_cache_capacity_test.zig");
}

test "KV admission: a refusal prices the running states beside the fixed part and KV page" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const l: kv_pool.Layout = .{ .fixed_bytes = 1 << 30, .token_bytes = 1 << 28, .page_tokens = 4, .snap_bytes = 1, .run_bytes = 1 << 30 };
    const o: api.Open = .{ .dir = "", .model_type = "n", .lanes = 2, .lanes_fixed = true, .prompt_cache_gib = 0 };
    try std.testing.expectEqual(@as(?Plan, null), try plan(struct {}, arena.allocator(), l, 2 << 30, o, 8, &problem, false));
    try std.testing.expect(std.mem.indexOf(u8, problem, "needs 3.00 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "state runs") != null);
}
