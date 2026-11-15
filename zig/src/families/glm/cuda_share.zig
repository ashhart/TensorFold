//! TF_KV_SHARE: GLM's side of the shared prompt store (native/cuda_prompt_pages.zig) over rank 0's KV pages.
const std = @import("std");
const kp = @import("core").kv_prompts;
const kv = @import("core").kv_pool;
const engine = @import("cuda_engine.zig");
const page_rows = @import("cuda_state.zig").page_rows;
const Stream = @import("lanes").Stream;

/// Image passes require the server's content key; text without a key retains its token identity.
pub fn cacheKey(s: *const Stream) ?[]const u32 {
    if (s.key.len != 0) return if (s.key.len == s.prompt_len) s.key else null;
    return if (s.images.len == 0) s.prompt() else null;
}

pub fn cacheable(s: *const Stream) bool {
    return s.prompt_logprobs == null and !s.untaught and s.embedding == null and cacheKey(s) != null;
}

/// Below 4 states a stream: that many, the --prompt-cache-gib they need beside TF_KV_SHARE (null: none), and the most.
pub fn advice(states: usize, parallel: usize, l: kv.Layout, planned: ?kv.Budget) ?struct { states: usize, gib: ?u64, most: usize } {
    const want = 4 * parallel;
    if (states >= want) return null;
    const b = planned orelse return .{ .states = want, .gib = kv.holdingGib(want, l.snap_bytes), .most = want };
    var named = b; // TF_KV_SHARE=want: its slots come before the windows past the engine's own (kv_pool.plan)
    named.snaps = want;
    if (!b.asked) named.cache = std.math.maxInt(u64); // and an unasked cache is sized after them (budgetOf)
    if (if (kv.plan(l, named)) |p| p.snaps >= want else |_| false) return .{ .states = want, .gib = null, .most = want };
    const h = kv.holding(l, b, want);
    return .{ .states = want, .gib = h.gib, .most = h.most };
}

/// More kept states than a startup plan sets slots aside for (kv_pool.most_snaps): refused by name at start.
pub fn tooMany(states: usize) bool {
    return states > @import("core").kv_pool.most_snaps;
}

test "TF_KV_SHARE: under 4 planned states a GLM stream, the advice names 4 a stream, and a cache only if they need it" {
    const mib: u64 = 1 << 20;
    const g: u64 = 1 << 30;
    const l: kv.Layout = .{ .fixed_bytes = 340 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 180 * mib };
    try std.testing.expect(advice(16, 4, l, null) == null);
    const a = advice(13, 4, l, null).?; // a 2.4 GiB cache holds 13 of the 16 at --parallel 4; no device plan to check
    try std.testing.expectEqual(@as(usize, 16), a.states);
    try std.testing.expectEqual(@as(?u64, 3), a.gib); // 16 x 180 MiB is 2.8 GiB
    try std.testing.expectEqual(@as(usize, 32), advice(0, 8, l, null).?.states); // a cache holding none
    const tight: kv.Budget = .{ .room = 16 * g + g / 2, .engine = 0, .lanes = 4, .fixed = true, .cache = 16 * g, .snaps = 16, .window = 1_048_576 + 8, .engine_paged = true, .slots_only = true };
    const t = advice(15, 4, l, tight).?; // TP=2 at a 1M window: only the engine's own window comes before the cache
    try std.testing.expectEqual(@as(?u64, null), t.gib); // 3 GiB keeps no more there
    try std.testing.expectEqual(@as(usize, 15), t.most);
    var wide = tight;
    wide.room = 24 * g; // TF_KV_SHARE=16 alone keeps all 16 there: no --prompt-cache-gib is named
    const w = advice(15, 4, l, wide).?;
    try std.testing.expectEqual(@as(?u64, null), w.gib);
    try std.testing.expectEqual(@as(usize, 16), w.most);
    var strip = tight;
    strip.room = 16 * g + 11 * g / 16; // the 16 slots fit past the engine's own window, where TF_KV_SHARE's slots go
    try std.testing.expectEqual(@as(?u64, null), advice(15, 4, l, strip).?.gib);
    // TP=3 at Jammed XSFP4's 42.0 GiB, --parallel 8 (327 MiB fixed, a 168 MiB slot): TF_KV_SHARE=8 keeps its 8
    const now3: kv.Layout = .{ .fixed_bytes = 327 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 168 * mib };
    const eight: kv.Budget = .{ .room = 41 * g + 992 * mib, .engine = 0, .lanes = 8, .fixed = true, .cache = 3 * g / 2, .snaps = 8, .window = 1_048_576 + 8, .engine_paged = true, .slots_only = true };
    const j = advice(8, 8, now3, eight).?; // 32 take TF_KV_SHARE=32 alone, though the cache sized for 8 holds 9 slots
    try std.testing.expectEqual(@as(?u64, null), j.gib);
    try std.testing.expectEqual(@as(usize, 32), j.most);
    var small = eight; // --prompt-cache-gib 2 beside TF_KV_SHARE=32 keeps 12: the advice names both settings
    small.snaps = 32;
    small.asked = true;
    small.cache = 2 * g;
    try std.testing.expectEqual(@as(usize, 12), (try kv.plan(now3, small)).snaps);
    try std.testing.expectEqual(@as(?u64, 6), advice(12, 8, now3, small).?.gib);
}

test "TF_KV_SHARE: up to 64 states is a plan's, 65 is refused" {
    try std.testing.expect(!tooMany(64) and tooMany(65) and !tooMany(0));
}

/// TF_GLM_KV_SHARE is TF_KV_SHARE now: a value that asked for a store is refused; empty and 0 asked for none.
pub fn moved(v: []const u8) bool {
    return v.len != 0 and (std.fmt.parseInt(usize, v, 10) catch 1) != 0;
}

test "TF_GLM_KV_SHARE: a count or anything else is refused; empty and 0 asked for nothing" {
    try std.testing.expect(moved("16"));
    try std.testing.expect(moved("on"));
    try std.testing.expect(!moved(""));
    try std.testing.expect(!moved("0"));
}

/// The store the server attached to rank 0's pages (null: none, or another rank).
pub fn of(e: *const engine.Engine) ?kp.Prompts {
    const p = e.pages orelse return null;
    return p.prompts;
}

/// A resume's state: restored by the prompt pass's start on every rank (engine.Resume), once its pages are mapped.
pub const Owner = struct { pending: ?*engine.Kept = null };

/// Before a prompt pass: the deepest kept state's pages in `adopt` (held for the stream), the state in `owner`.
pub fn begin(e: *engine.Engine, s: *const Stream, owner: *Owner, adopt: *std.ArrayList(u32)) !kp.Plan {
    const prompts = of(e) orelse return error.NoPromptStore;
    const plan = try prompts.begin(.of(cacheKey(s) orelse return error.ImageCacheKeyRequired), s.history_len, s.shared_prefixes, owner, adopt);
    defer prompts.free(plan.marks);
    const mine = e.gpa.dupe(u32, plan.marks) catch |err| { // the lane's own marks, freed by its allocator
        e.pages.?.give(adopt.items);
        adopt.clearRetainingCapacity();
        return err;
    };
    return .{ .from = plan.from, .marks = mine };
}

/// A pass's marks (the engine's Marks.ctx): the mark it stands at, or the last it completed when cancelled, keeps.
pub const Keeps = struct {
    e: *engine.Engine,
    slot: u32,
    prompt: []const u32,

    pub fn at(ctx: *anyopaque, mark: u32) void {
        const k: *Keeps = @ptrCast(@alignCast(ctx));
        k.keep(mark);
    }

    /// Slot `slot`'s pass at mark `mark`: the store takes the pages below it it lacks, and the state (no table swaps).
    pub fn keep(k: *const Keeps, mark: u32) void {
        const prompts = of(k.e) orelse return;
        const p = k.e.pages.?;
        if (!p.paged(k.slot)) return;
        const n = mark / page_rows;
        const ids = p.leases[k.slot].ids.items;
        if (n > ids.len) return;
        const path = k.e.gpa.alloc(u32, n) catch return;
        defer k.e.gpa.free(path);
        var owner: Owner = .{};
        _ = prompts.keep(.of(k.prompt), mark, &owner, ids[0..n], path);
    }
};

/// The same content key names lookup and every completed or interrupted keep.
pub fn marks(e: *engine.Engine, slot: u32, s: *const Stream, at: []const u32, keeps: *Keeps) ?engine.Marks {
    keeps.* = .{ .e = e, .slot = slot, .prompt = cacheKey(s) orelse return null };
    return if (at.len == 0) null else .{ .at = at, .ctx = keeps, .call = Keeps.at, .interrupted = Keeps.at };
}
