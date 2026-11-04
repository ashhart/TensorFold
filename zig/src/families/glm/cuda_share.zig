//! TF_KV_SHARE: GLM's side of the shared prompt store (native/cuda_prompt_pages.zig) over rank 0's KV pages.
const std = @import("std");
const kp = @import("core").kv_prompts;
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

/// Below 4 states a stream, live streams evict each other's before a resume reaches them.
pub fn fewStates(states: usize, parallel: usize) bool {
    return states > 0 and states < 4 * parallel;
}

test "TF_KV_SHARE below 4 states a GLM stream is too few; off is not" {
    try std.testing.expect(fewStates(4, 4));
    try std.testing.expect(fewStates(15, 4));
    try std.testing.expect(!fewStates(16, 4));
    try std.testing.expect(!fewStates(0, 4));
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
    const plan = try prompts.begin(cacheKey(s) orelse return error.ImageCacheKeyRequired, s.history_len, s.shared_prefixes, owner, adopt);
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
        _ = prompts.keep(k.prompt, mark, &owner, ids[0..n], path);
    }
};

/// The same content key names lookup and every completed or interrupted keep.
pub fn marks(e: *engine.Engine, slot: u32, s: *const Stream, at: []const u32, keeps: *Keeps) ?engine.Marks {
    keeps.* = .{ .e = e, .slot = slot, .prompt = cacheKey(s) orelse return null };
    return if (at.len == 0) null else .{ .at = at, .ctx = keeps, .call = Keeps.at, .interrupted = Keeps.at };
}
