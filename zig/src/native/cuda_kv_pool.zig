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

/// What the load ledger hears of an engine's KV: its mode and, when paged, the pool it watches.
pub const Watch = struct {
    mode: ?api.load_ledger.Mode = null, // null: the capacity group stays absent
    src: ?kv_pool.Source = null,
    host: ?*api.LaneHost = null,
};

/// A window a stream without a page plan; a paged family's pool when it names one (`kvPool`); otherwise none.
pub fn watchOf(comptime F: type, loaded: anytype, paged: ?Plan) Watch {
    if (paged == null) return .{ .mode = .window };
    if (comptime !@hasDecl(F, "kvPool")) return .{};
    const src = F.kvPool(loaded.ctx) orelse return .{};
    return .{ .mode = .paged, .src = src };
}

/// The pool whose promises the lane host's ledger admits: the watched one, or the one a family names (`ledgerPool`).
pub fn ledgerPool(comptime F: type, loaded: anytype) ?*kv_pool.Pool {
    if (comptime @hasDecl(F, "kvPool")) if (F.kvPool(loaded.ctx)) |src| return src.pool;
    if (comptime @hasDecl(F, "ledgerPool")) return F.ledgerPool(loaded.ctx);
    return null;
}

/// The lane host's capacity source and the pool's first census, before the engine thread owns the pool.
pub fn install(w: *Watch, lane: *api.LaneHost) void {
    const mode = w.mode orelse return;
    lane.capacity = .{ .mode = mode };
    const src = w.src orelse return;
    w.host = lane;
    src.pool.notify = .{ .ctx = w, .call = changed };
    api.LaneHost.Load.poolChanged(lane, census(src));
}

/// The pool stops reporting before its owner closes.
pub fn unwatch(w: *Watch) void {
    if (w.src) |src| src.pool.notify = null;
}

fn changed(ctx: *anyopaque, _: *const kv_pool.Pool) void {
    const w: *Watch = @ptrCast(@alignCast(ctx));
    api.LaneHost.Load.poolChanged(w.host.?, census(w.src.?));
}

/// The pool's counts in tokens at page granularity; `run` only where the family needs a contiguous run.
pub fn census(src: kv_pool.Source) api.load_ledger.Pool {
    const p = src.pool;
    const t: u64 = src.page_tokens;
    return .{
        .capacity = p.total() * t,
        .available = p.available() * t,
        .reclaimable = p.reclaimable() * t,
        .promised = p.promised * t,
        .run = if (src.run_pages > 0) p.findRun(src.run_pages) != null else null,
    };
}

test "a pool's census is in tokens, and a run shows only where the family needs one" {
    var p = try kv_pool.Pool.init(std.testing.allocator, 8, 64);
    defer p.deinit();
    var l: kv_pool.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 3);
    _ = try p.take(&l);
    const plain = census(.{ .pool = &p, .page_tokens = 64 });
    try std.testing.expectEqual(api.load_ledger.Pool{ .capacity = 512, .available = 320, .reclaimable = 0, .promised = 128 }, plain);
    try std.testing.expectEqual(@as(?bool, true), census(.{ .pool = &p, .page_tokens = 64, .run_pages = 4 }).run);
    try std.testing.expectEqual(@as(?bool, false), census(.{ .pool = &p, .page_tokens = 64, .run_pages = 8 }).run);
}

/// The plan's view of a wrapped family: drafter buffers join the fixed part; no free stream pays its window first.
pub fn served(l: ?kv_pool.Layout, free_streams: u32, family_bytes: u64, stream_bytes: u64, room: u64) struct { layout: ?kv_pool.Layout, room: u64 } {
    var out = l orelse return .{ .layout = null, .room = room };
    out.fixed_bytes += stream_bytes -| family_bytes;
    return .{ .layout = out, .room = if (free_streams == 0) room -| stream_bytes else room };
}

test "a wrapped family's plan: drafter buffers a stream, a first window without a free stream" {
    const l: kv_pool.Layout = .{ .fixed_bytes = 10, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 1 };
    const nemotron = served(l, 1, 500, 500, 1000); // its own sequence is in the model's bytes: nothing moves
    try std.testing.expectEqual(@as(u64, 10), nemotron.layout.?.fixed_bytes);
    try std.testing.expectEqual(@as(u64, 1000), nemotron.room);
    const qwen = served(l, 0, 500, 530, 1000); // a drafter's 30 bytes a stream, the first stream's 530 first
    try std.testing.expectEqual(@as(u64, 40), qwen.layout.?.fixed_bytes);
    try std.testing.expectEqual(@as(u64, 470), qwen.room);
    try std.testing.expectEqual(@as(?kv_pool.Layout, null), served(null, 0, 1, 2, 3).layout);
}

/// served() on the family's layout; a family that places a drafter's buffers itself (pagedLayout) plans it again.
pub fn wrapped(comptime F: type, loaded: anytype, free: u32, stream_bytes: u64, room: u64) @TypeOf(served(null, 0, 0, 0, 0)) {
    const own = comptime @hasDecl(F, "pagedLayout");
    const l = layout(loaded);
    const planned = if (own) (if (l != null) F.pagedLayout(loaded.ctx, stream_bytes - loaded.stream_bytes) else null) else l;
    return served(planned, free, if (own) stream_bytes else loaded.stream_bytes, stream_bytes, room);
}

/// Every stream on the pool's pages, the engine's own window idle (Qwen's TF_QWEN_KV_SHARE and TF_QWEN_KV_PAGES_ALL).
pub fn allOnPages(loaded: anytype) bool {
    return if (comptime @hasField(@TypeOf(loaded), "all_on_pages")) loaded.all_on_pages else false;
}

/// Whether the family keeps prompts itself (qwen3_5's TF_QWEN_KV_SHARE): the host makes no prompt cache then.
pub fn ownCache(loaded: anytype) bool {
    return if (comptime @hasField(@TypeOf(loaded), "own_prompt_cache")) loaded.own_prompt_cache else false;
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

/// A family declaring `keeps_no_prompts = true` has no prompt store: --prompt-cache-gib is refused, no cache planned.
pub fn keepsNoPrompts(comptime F: type) bool {
    return @hasDecl(F, "keeps_no_prompts") and F.keeps_no_prompts;
}

/// The cache the plan starts from: --prompt-cache-gib's or the default floor, none for a family that keeps no prompts.
fn cacheWanted(comptime F: type, o: api.Open) u64 {
    return if (comptime keepsNoPrompts(F)) 0 else budget.cacheBytes(o.prompt_cache_gib, std.math.maxInt(u64), 0);
}

/// The asked cache (16 GiB unless --prompt-cache-gib says) whole beside the streams, by the startup plan's arithmetic.
fn fits(comptime F: type, loaded: anytype, o: api.Open, room: u64, window: u64, stream_bytes: u64) bool {
    const want = cacheWanted(F, o);
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    if (layout(loaded)) |l| return kv_pool.holds(l, .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = named(o), .window = window + rows, .engine_paged = enginePaged(loaded), .asked = o.prompt_cache_gib != null });
    const n = budget.admit(room, stream_bytes, loaded.free_streams, o.lanes, o.lanes_fixed) catch return false;
    return room -| stream_bytes * (n -| loaded.free_streams) >= want;
}

/// Plan streams first, then half their remaining room with a 16 GiB floor, unless the cache was explicit.
pub fn budgetOf(comptime F: type, l: kv_pool.Layout, room: u64, o: api.Open, window: u64, host_mode: bool, engine_paged: bool, kept_ppm: u32, slots: ?usize) kv_pool.Budget {
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    const want = if (host_mode) 0 else cacheWanted(F, o);
    var out: kv_pool.Budget = .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = slots orelse named(o), .window = window + rows, .engine_paged = engine_paged, .kept_ppm = if (host_mode) 0 else kept_ppm, .slots_only = slots != null, .asked = o.prompt_cache_gib != null };
    if (out.asked or host_mode or comptime keepsNoPrompts(F)) return out;
    var probe = out;
    probe.cache = std.math.maxInt(u64);
    const all = kv_pool.plan(l, probe) catch return out;
    const left = if (!engine_paged and out.window > 0)
        room -| (@as(u64, all.lanes - 1) *| (l.fixed_bytes +| (l.pages(out.window) + l.runPages()) *| l.pageBytes()))
    else
        all.snaps * l.snap_bytes + all.copied + all.kept;
    out.cache = budget.cacheBytes(null, left, 0);
    return out;
}

/// The pool beside the engine's own sequence (in `model`: a window, or on pages its fixed part); null: see `problem`.
pub fn plan(comptime F: type, a: Allocator, l: kv_pool.Layout, room: u64, o: api.Open, window: u64, problem: *[]const u8, host_mode: bool, engine_paged: bool, kept_ppm: u32, slots: ?usize) !?Plan {
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    return kv_pool.plan(l, budgetOf(F, l, room, o, window, host_mode, engine_paged, kept_ppm, slots)) catch |e| {
        problem.* = switch (e) {
            error.NoStream => if (engine_paged) try std.fmt.allocPrint(a, "the CUDA memory budget fits no stream beside the model: one at a {d}-token window takes {d:.2} GiB of KV pages, and the budget leaves {d:.2} GiB", .{ window, toGib(l.pages(window + rows) * l.pageBytes()), toGib(room) }) else "the CUDA memory budget fits no stream beside the model",
            error.TooMany => try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its other streams' fixed parts, state runs and a page each, and the CUDA memory budget leaves {d:.2} GiB: serve fewer, or free device memory", .{ o.lanes, toGib((o.lanes - 1) * (l.fixed_bytes + (1 + l.runPages()) * l.pageBytes())), toGib(room) }),
        };
        return null;
    };
}

/// The planned streams' sequences and pages made now; false: `problem` says why they could not be.
pub fn attach(comptime F: type, a: Allocator, ctx: *anyopaque, paged: ?Plan, planned: ?kv_pool.Budget, problem: *[]const u8) !bool {
    if (comptime !@hasDecl(F, "attach")) return true;
    const p = paged orelse return true;
    F.attach(ctx, p.lanes - 1, p.pages) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA engine could not make its {d} streams' pages ({s})", .{ p.lanes - 1, @errorName(e) });
        return false;
    };
    if (comptime @hasDecl(F, "share_states")) F.share_states(ctx, p.snaps, p.lanes, planned) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA engine could not ready its kept states ({s})", .{@errorName(e)});
        return false;
    };
    return true;
}

/// The cache's snapshot slots and copied rows' bytes (`snaps` 0: a family without slots keeps its cache whole).
pub const Kept = struct {
    snaps: usize,
    copied: u64,
    snap_bytes: u64,
    host: bool = false,
    device: bool = false, // host budget, states in device memory: an integrated GPU's (`device_snaps`)
    most: usize = std.math.maxInt(usize),
};

pub fn kept(comptime F: type, loaded: anytype, paged: ?Plan, o: api.Open, room: u64, held: u64, host_budget: u64, host_mode: bool) Kept {
    if (host_mode) return .{
        .snaps = 0,
        .copied = hostBudget(o, host_budget),
        .snap_bytes = stagingBytes(F, loaded, hostBudget(o, host_budget)),
        .host = true,
        .device = device(loaded, o),
        .most = named(o) orelse std.math.maxInt(usize),
    };
    // on pages the plan set the cache aside already; `held` counts every stream at a whole window
    if (comptime !@hasDecl(F, "snap_slots")) return .{ .snaps = 0, .copied = if (!@hasDecl(F, "snap_save")) 0 else if (paged) |p| p.copied + p.kept else budget.cacheBytes(o.prompt_cache_gib, room, held), .snap_bytes = 0 }; // kept: the store's budget, held as pool pages
    const b: u64 = loaded.snap_bytes;
    const s = if (paged) |p| kv_pool.split(p.snaps * b + p.copied, b, p.snaps) else kv_pool.split(budget.cacheBytes(o.prompt_cache_gib, room, held), b, named(o));
    return .{ .snaps = s.snaps, .copied = s.copied, .snap_bytes = b };
}

/// The store keeps a state a slot, the slots made now in one device buffer; a family without them, its own budget.
pub fn store(comptime F: type, s: *api.prompt_cache.Store, loaded: anytype, k: Kept) !void {
    s.over = @import("cuda_prompt_pages.zig").hint(F, loaded); // a state past the budget: the setting keeping it
    s.keep_others = api.prompt_cache.others.fromEnv(); // =1: a new state evicts only its own conversation's states
    if (comptime !@hasDecl(F, "snap_slots")) return;
    if (comptime @hasDecl(F, "snap_place")) F.snap_place(loaded.target, k.host);
    s.most = if (k.host) k.most else k.snaps;
    _ = try F.snap_slots(loaded.target, k.snaps);
}

/// The load report of a paged family's groups, a line apart from the startup line.
pub fn report(a: Allocator, l: ?kv_pool.Layout, paged: ?Plan) !void {
    if (paged == null) return;
    std.log.info("{s}", .{try kv_pool.groups.line(a, (l orelse return).shape())});
}

/// The startup line's streams and cache.
pub fn text(comptime F: type, a: Allocator, stream_bytes: u64, l: ?kv_pool.Layout, paged: ?Plan, engine_paged: bool, all_on_pages: bool, window: u64, k: Kept, kv8: bool) ![2][]const u8 {
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0; // the plan's windows: window + rows
    const shape = if (paged) |p| (if (all_on_pages) try std.fmt.allocPrint(a, "all {d} on {d} pages of {d} rows ({d:.2} GiB) as they need them, the engine's own window idle, {d:.0} MiB fixed each", .{ p.lanes, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) }) else if (engine_paged) try std.fmt.allocPrint(a, "the engine's own and {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them (enough for {d} at a {d}-token window), {d:.0} MiB fixed each", .{ p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), p.pages / l.?.pages(window + rows), window, @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) }) else try std.fmt.allocPrint(a, "the engine's own at a {d}-token window ({d:.2} GiB, in the model's), {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them, {d:.0} MiB fixed each", .{ window, toGib(stream_bytes), p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) })) else try std.fmt.allocPrint(a, "{d:.2} GiB each at a {d}-token window{s}", .{ toGib(stream_bytes), window, if (kv8) " (fp8 caches)" else "" });
    const placed = comptime @hasDecl(F, "snap_place"); // the startup line says where its prompt states went, and why
    if (k.copied > 0 and k.device) {
        const line = ", a {d:.1} GiB prompt cache in device memory, budgeted as host memory";
        return .{ shape, try std.fmt.allocPrint(a, line, .{toGib(k.copied)}) };
    }
    const cache = if (k.copied == 0) ", no prompt cache" else if (k.host) try std.fmt.allocPrint(a, ", {s}a {d:.1} GiB prompt cache in host memory with {d:.0} MiB of device staging", .{ if (placed) "prompt states on host, the cache doesn't fit on the card: " else "", toGib(k.copied), @as(f64, @floatFromInt(k.snap_bytes)) / (1 << 20) }) else if (comptime @hasDecl(F, "snap_slots")) try std.fmt.allocPrint(a, ", {s}a prompt cache of {d} snapshot slots ({d:.1} GiB) and {d:.1} GiB of copied rows", .{ if (placed) "prompt states on device: " else "", k.snaps, toGib(k.snaps * k.snap_bytes), toGib(k.copied) }) else try std.fmt.allocPrint(a, ", a {d:.1} GiB prompt cache{s}", .{ toGib(k.copied), if (comptime @hasDecl(F, "snap_reserved")) ", reserved at load" else "" });
    return .{ shape, cache };
}

/// The startup line's cache: the `asked` TF_KV_SHARE slots (GLM), those that fit, or over the pool's pages, or `cache`.
pub fn cacheText(a: Allocator, over_pages: bool, asked: ?usize, paged: ?Plan, l: ?kv_pool.Layout, planned: ?kv_pool.Budget, cache: []const u8) ![]const u8 {
    const n = asked orelse return if (over_pages) ", the prompt cache over the pool's pages" else cache;
    const snaps = paged.?.snaps;
    const slot = l.?.snap_bytes;
    if (snaps > 0 and snaps >= n) return std.fmt.allocPrint(a, ", {d} kept states' slots ({d:.1} GiB) and the prompt cache over the pool's pages", .{ snaps, toGib(snaps * slot) });
    const h = kv_pool.holding(l.?, planned.?, n); // a --prompt-cache-gib named only where asking it keeps all n
    const more = if (h.gib) |gib| try std.fmt.allocPrint(a, "--prompt-cache-gib {d} holds all {d}", .{ gib, n }) else try std.fmt.allocPrint(a, "the room holds {d} at any --prompt-cache-gib", .{h.most});
    if (snaps == 0) return std.fmt.allocPrint(a, ", no prompt cache (none of TF_KV_SHARE={d} kept states' slots fit; {s})", .{ n, more });
    return std.fmt.allocPrint(a, ", {d} of TF_KV_SHARE={d} kept states' slots ({d:.1} GiB; {s}) and the prompt cache over the pool's pages", .{ snaps, n, toGib(snaps * slot), more });
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
/// Kept states in device memory on the host's budget, unless --learn-conversations' disk tier reads them on the host.
fn device(loaded: anytype, o: api.Open) bool {
    if (comptime !@hasField(@TypeOf(loaded), "device_snaps")) return false;
    return loaded.device_snaps and !(o.learn_conversations and o.learn != null);
}
pub fn stagingBytes(comptime F: type, loaded: anytype, host_budget: u64) u64 {
    if (comptime !@hasDecl(F, "snap_slots")) return 0;
    return if (host(F, loaded) and host_budget > 0) loaded.snap_bytes else 0;
}

test "the startup line names TF_KV_SHARE's slots, and when the cache holds fewer, how many of how many and the cache for all" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 64, .snap_bytes = 180 << 20 };
    const p: Plan = .{ .lanes = 2, .snaps = 16, .copied = 0, .pages = 100 };
    const roomy: kv_pool.Budget = .{ .room = 64 << 30, .engine = 0, .lanes = 2, .cache = 16 << 30, .snaps = 16, .slots_only = true };
    try std.testing.expectEqualStrings(", 16 kept states' slots (2.8 GiB) and the prompt cache over the pool's pages", try cacheText(a, true, 16, p, l, roomy, ", own"));
    var some = p;
    some.snaps = 13; // a 2.4 GiB cache: 13 of the 16 slots, 16 x 180 MiB rounding up to 3 GiB
    try std.testing.expectEqualStrings(", 13 of TF_KV_SHARE=16 kept states' slots (2.3 GiB; --prompt-cache-gib 3 holds all 16) and the prompt cache over the pool's pages", try cacheText(a, true, 16, some, l, roomy, ", own"));
    var tight = roomy;
    tight.room = (12 << 30) / 5; // 2.4 GiB: no --prompt-cache-gib keeps more than 13 here, so the line names none
    try std.testing.expectEqualStrings(", 13 of TF_KV_SHARE=16 kept states' slots (2.3 GiB; the room holds 13 at any --prompt-cache-gib) and the prompt cache over the pool's pages", try cacheText(a, true, 16, some, l, tight, ", own"));
    var strip = tight;
    strip.room = 2976 << 20; // the 16 slots fit and 3 GiB doesn't: an asked 3 is clipped to the rest and keeps all 16
    some.snaps = 15;
    try std.testing.expectEqualStrings(", 15 of TF_KV_SHARE=16 kept states' slots (2.6 GiB; --prompt-cache-gib 3 holds all 16) and the prompt cache over the pool's pages", try cacheText(a, true, 16, some, l, strip, ", own"));
    some.snaps = 0;
    try std.testing.expectEqualStrings(", no prompt cache (none of TF_KV_SHARE=16 kept states' slots fit; --prompt-cache-gib 3 holds all 16)", try cacheText(a, true, 16, some, l, roomy, ", own"));
    tight.room = 100 << 20;
    try std.testing.expectEqualStrings(", no prompt cache (none of TF_KV_SHARE=16 kept states' slots fit; the room holds 0 at any --prompt-cache-gib)", try cacheText(a, true, 16, some, l, tight, ", own"));
    try std.testing.expectEqualStrings(", the prompt cache over the pool's pages", try cacheText(a, true, null, p, l, null, ", own"));
    try std.testing.expectEqualStrings(", own", try cacheText(a, false, null, null, null, null, ", own"));
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
    try std.testing.expect(!k.device);
}
test "kv merge: an integrated GPU's host-budget cache names device memory, unless the disk tier keeps it on the host" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const loaded = .{ .host_snaps = true, .device_snaps = true, .snap_bytes = @as(u64, 8 << 20) };
    const o: api.Open = .{ .dir = "", .model_type = "q", .prompt_cache_gib = 8, .lanes = 4 };
    const k = kept(Copies, loaded, null, o, 256 << 20, 0, 8 << 30, true);
    try std.testing.expect(k.host and k.device);
    const shown = (try text(Copies, a, 1 << 30, null, null, false, false, 70000, k, false))[1];
    try std.testing.expectEqualStrings(", a 8.0 GiB prompt cache in device memory, budgeted as host memory", shown);
    var learning = o;
    learning.learn = "/s";
    learning.learn_conversations = true;
    const disk = kept(Copies, loaded, null, learning, 256 << 20, 0, 8 << 30, true);
    try std.testing.expect(disk.host and !disk.device);
    const host_line = (try text(Copies, a, 1 << 30, null, null, false, false, 70000, disk, false))[1];
    const host_text = ", a 8.0 GiB prompt cache in host memory with 0 MiB of device staging";
    try std.testing.expectEqualStrings(host_text, host_line);
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
    const on_host = (try text(Placing, a, 1 << 30, l, null, false, false, 70000, .{ .snaps = 0, .copied = 8 << 30, .snap_bytes = 48 << 20, .host = true }, false))[1];
    try std.testing.expect(std.mem.startsWith(u8, on_host, ", prompt states on host, the cache doesn't fit on the card: a 8.0 GiB"));
    const on_device = (try text(Placing, a, 1 << 30, l, null, false, false, 70000, .{ .snaps = 64, .copied = 13 << 30, .snap_bytes = 48 << 20 }, false))[1];
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
    const p = (try plan(Owned, a, l, 40 << 30, o, 8192, &problem, false, true, 0, null)).?;
    try std.testing.expectEqual(@as(u32, 4), p.lanes);
    try std.testing.expectEqual(@as(usize, 4 * 17), p.pages); // every stream at most 8,192 rows and a verify window: 17 pages
    const floor = 17 * l.pageBytes();
    try std.testing.expectEqual(@as(?Plan, null), try plan(Owned, a, l, floor - 1, o, 8192, &problem, false, true, 0, null));
    try std.testing.expect(std.mem.indexOf(u8, problem, "8192-token window") != null);
    const shown = (try text(Owned, a, 180 << 20, l, p, true, false, 8192, .{ .snaps = 0, .copied = 4 << 30, .snap_bytes = 0 }, false))[0];
    try std.testing.expectEqualStrings("the engine's own and 3 more taking 68 pages of 512 rows (0.42 GiB) as they need them (enough for 4 at a 8192-token window), 180 MiB fixed each", shown);
    // TF_QWEN_KV_SHARE: every stream on the pool's pages, the engine's window idle
    const all = (try text(Owned, a, 180 << 20, l, p, false, true, 8192, .{ .snaps = 0, .copied = 4 << 30, .snap_bytes = 0 }, false))[0];
    try std.testing.expectEqualStrings("all 4 on 68 pages of 512 rows (0.42 GiB) as they need them, the engine's own window idle, 180 MiB fixed each", all);
    try std.testing.expect(allOnPages(.{ .all_on_pages = true }) and !allOnPages(.{ .layout = @as(?kv_pool.Layout, l) }));
    // the unasked cache holds past the four windows' pages this room holds and a fixed part each, not a byte less
    const room = 4 * floor + 3 * (180 << 20) + (16 << 30);
    const loaded = .{ .layout = @as(?kv_pool.Layout, l), .engine_paged = true, .free_streams = @as(u32, 1) };
    const unasked: api.Open = .{ .dir = "", .model_type = "g", .lanes = 4 };
    try std.testing.expect(fits(Owned, loaded, unasked, room, 8192, 0));
    try std.testing.expect(!fits(Owned, loaded, unasked, room - 1, 8192, 0));
    // an asked cache holds past the engine's own window and a fixed part and page for each other one, not a byte less
    const each = (180 << 20) + l.pageBytes();
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

/// `taken`: the cache bytes the page plan's windows past the engine's own took first (Plan.taken; 0 without pages).
pub fn warnCapacity(facts: ?api.PromptCacheCapacity, disabled: bool, taken: u64, comptime log: type) void {
    const c = facts orelse return;
    if (disabled or c.budget_bytes >= c.minimum_state_bytes) return;
    const short = "GLM CUDA prompt cache budget {d} bytes ({d:.3} GiB) cannot keep one state: the minimum planned state needs {d} bytes ({d:.3} GiB); ";
    if (taken > 0) return log.warn(short ++ "the KV pool's windows past the engine's own took {d:.3} GiB of it first, and --prompt-cache-gib N keeps N GiB before them", .{ c.budget_bytes, toGib(c.budget_bytes), c.minimum_state_bytes, toGib(c.minimum_state_bytes), toGib(taken) });
    log.warn(short ++ "lower --context or --parallel, or TENSORFOLD_MEMORY_RESERVE_GIB, to make room (--prompt-cache-gib only caps the budget the weights and streams leave)", .{ c.budget_bytes, toGib(c.budget_bytes), c.minimum_state_bytes, toGib(c.minimum_state_bytes) });
}

test {
    _ = @import("cuda_cache_capacity_test.zig");
    _ = @import("cuda_streams.zig");
    _ = @import("cuda_keep_others_test.zig");
}

test "KV admission: a refusal prices the running states beside the fixed part and KV page" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const l: kv_pool.Layout = .{ .fixed_bytes = 1 << 30, .token_bytes = 1 << 28, .page_tokens = 4, .snap_bytes = 1, .run_bytes = 1 << 30 };
    const o: api.Open = .{ .dir = "", .model_type = "n", .lanes = 2, .lanes_fixed = true, .prompt_cache_gib = 0 };
    try std.testing.expectEqual(@as(?Plan, null), try plan(struct {}, arena.allocator(), l, 2 << 30, o, 8, &problem, false, false, 0, null));
    try std.testing.expect(std.mem.indexOf(u8, problem, "needs 3.00 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "state runs") != null);
}

test "CUDA cache default scales pool copies after full stream windows without cutting lanes" {
    const GiB: u64 = 1 << 30;
    const F = struct {};
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 1 << 20, .page_tokens = 1, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "model", .model_type = "qwen3_5", .lanes = 8 };
    const b = budgetOf(F, l, 88 * GiB, o, 6144, false, true, 0, null);
    const p = try kv_pool.plan(l, b);
    try std.testing.expectEqual(@as(u32, 8), p.lanes);
    try std.testing.expectEqual(@as(u64, 20 * GiB), b.cache);
    try std.testing.expectEqual(@as(u64, 20 * GiB), p.copied);
    var explicit = o;
    explicit.prompt_cache_gib = 8;
    try std.testing.expectEqual(@as(u64, 8 * GiB), budgetOf(F, l, 88 * GiB, explicit, 6144, false, true, 0, null).cache);
    explicit.prompt_cache_gib = 0;
    try std.testing.expectEqual(@as(u64, 0), budgetOf(F, l, 88 * GiB, explicit, 6144, false, true, 0, null).cache);
}

test "CUDA cache default leaves seven full windows beside an unpaged engine" {
    const GiB: u64 = 1 << 30;
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 1 << 20, .page_tokens = 1, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "model", .model_type = "n", .lanes = 8 };
    const b = budgetOf(struct {}, l, 88 * GiB, o, 8192, false, false, 0, null);
    const p = try kv_pool.plan(l, b);
    try std.testing.expectEqual(@as(u32, 8), p.lanes);
    try std.testing.expectEqual(@as(u64, 16 * GiB), b.cache);
    try std.testing.expectEqual(@as(usize, 7 * 8192), p.pages);
    try std.testing.expectEqual(@as(u64, 16 * GiB), p.copied);
    var explicit = o;
    explicit.prompt_cache_gib = 40;
    try std.testing.expectEqual(@as(u64, 40 * GiB), budgetOf(struct {}, l, 88 * GiB, explicit, 8192, false, false, 0, null).cache);
}

test "CUDA cache default uses half of unpaged room when it exceeds the floor" {
    const GiB: u64 = 1 << 30;
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 1 << 20, .page_tokens = 1, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "model", .model_type = "n", .lanes = 8 };
    const b = budgetOf(struct {}, l, 104 * GiB, o, 8192, false, false, 0, null);
    const p = try kv_pool.plan(l, b);
    try std.testing.expectEqual(@as(u64, 24 * GiB), b.cache);
    try std.testing.expectEqual(@as(u32, 8), p.lanes);
    try std.testing.expectEqual(@as(usize, 7 * 8192), p.pages);
    try std.testing.expectEqual(@as(u64, 24 * GiB), p.copied);
}

test "CUDA cache default leaves no copies when unpaged windows use all room" {
    const GiB: u64 = 1 << 30;
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 1 << 20, .page_tokens = 1, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "model", .model_type = "n", .lanes = 8 };
    const b = budgetOf(struct {}, l, 56 * GiB, o, 8192, false, false, 0, null);
    const p = try kv_pool.plan(l, b);
    try std.testing.expectEqual(@as(u64, 0), b.cache);
    try std.testing.expectEqual(@as(u32, 8), p.lanes);
    try std.testing.expectEqual(@as(usize, 7 * 8192), p.pages);
    try std.testing.expectEqual(@as(u64, 0), p.copied);
}

const Promptless = struct {
    pub const keeps_no_prompts = true;
};
test "kv pages: a family that keeps no prompts plans no cache, so its streams take the pages one would have" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const GiB: u64 = 1 << 30;
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 1 << 20, .page_tokens = 1, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "", .model_type = "d", .lanes = 8 };
    // rooms where a family that keeps prompts gets main's scaled default cache, the engine's own on pages or not
    for ([_]struct { bool, u64, u64 }{ .{ true, 88 * GiB, 6144 }, .{ false, 104 * GiB, 8192 } }) |c| {
        try std.testing.expect(budgetOf(struct {}, l, c[1], o, c[2], false, c[0], 0, null).cache > 0);
        try std.testing.expectEqual(@as(u64, 0), budgetOf(Promptless, l, c[1], o, c[2], false, c[0], 0, null).cache);
        const plain = (try plan(struct {}, a, l, c[1], o, c[2], &problem, false, c[0], 0, null)).?;
        const none = (try plan(Promptless, a, l, c[1], o, c[2], &problem, false, c[0], 0, null)).?;
        try std.testing.expect(plain.copied > 0 and none.copied == 0 and none.kept == 0 and none.taken == 0);
        try std.testing.expect(none.pages >= plain.pages);
        try std.testing.expectEqual(plain.lanes, none.lanes);
        const budget_none: kv_pool.Budget = .{ .room = c[1], .engine = 0, .lanes = 8, .window = c[2], .engine_paged = c[0] };
        try std.testing.expectEqual(try kv_pool.plan(l, budget_none), none);
    }
    // a room the default floor's cache does not fit beside the streams: only the family that keeps no prompts fits
    const tight: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    for ([_]bool{ true, false }) |paged| {
        const loaded = .{ .layout = @as(?kv_pool.Layout, tight), .engine_paged = paged, .free_streams = @as(u32, 1) };
        try std.testing.expect(fits(Promptless, loaded, o, 3 << 29, 8192, 0) and !fits(struct {}, loaded, o, 3 << 29, 8192, 0));
    }
}
test "kv pages: LFM2-MoE keeps prompts without snapshot slots: its default cache plans as before, the store holds what the plan set aside, reserved at load" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const lfm = @import("lfm2_moe_cuda").native;
    try std.testing.expect(!keepsNoPrompts(lfm) and @hasDecl(lfm, "snap_save") and !@hasDecl(lfm, "snap_slots") and keepsNoPrompts(Promptless));
    const l: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "", .model_type = "lfm2_moe", .lanes = 8 };
    // as any family that keeps prompts: main's default cache, scaled by the room the streams leave
    const p = (try plan(lfm, arena.allocator(), l, 3 << 29, o, 8192, &problem, false, false, 0, null)).?;
    try std.testing.expectEqual(try kv_pool.plan(l, budgetOf(struct {}, l, 3 << 29, o, 8192, false, false, 0, null)), p);
    // on pages the store holds the plan's budget as pool pages; on fixed pages (TENSORFOLD_LFM2_KV_PAGES=0) the room's
    const q = (try plan(lfm, arena.allocator(), l, 40 << 30, o, 8192, &problem, false, false, 0, null)).?;
    try std.testing.expect(q.copied + q.kept > 0);
    try std.testing.expectEqual(q.copied + q.kept, kept(lfm, .{}, q, o, 40 << 30, 0, 0, false).copied);
    const k = kept(lfm, .{}, null, o, 40 << 30, 0, 0, false);
    try std.testing.expectEqual(budget.cacheBytes(null, 40 << 30, 0), k.copied);
    // the startup line says the cache is reserved at load; a family without `snap_reserved` doesn't
    const want = try std.fmt.allocPrint(arena.allocator(), ", a {d:.1} GiB prompt cache, reserved at load", .{toGib(k.copied)});
    try std.testing.expectEqualStrings(want, (try text(lfm, arena.allocator(), 1 << 30, null, null, false, false, 32768, k, false))[1]);
    try std.testing.expectEqualStrings(want[0 .. want.len - ", reserved at load".len], (try text(struct {}, arena.allocator(), 1 << 30, null, null, false, false, 32768, k, false))[1]);
}
