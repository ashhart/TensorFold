//! Real cache admission, keeps and restores; hypothetical selectors are isolated in the replay adapter.
const std = @import("std");
const pc = @import("cache");
const trace = @import("cache_replay_trace.zig");
const model = @import("cache_replay_model.zig");
pub const policy = @import("cache_replay_policy.zig");
pub const Replay = struct {
    family: model.Family,
    store: pc.Store,
    selected: policy.Policy,
    adapter_main: bool = false,
    peak_bytes: u64 = 0,
    pub fn init(r: *Replay, a: std.mem.Allocator, selected: policy.Policy, budget: u64, bytes: model.Model) void {
        r.* = .{ .family = .{ .gpa = a, .model = bytes }, .store = undefined, .selected = selected };
        r.store = pc.Store.init(a, r.family.snapshots(), .{ .chunk = 1 }, budget);
    }
    pub fn deinit(r: *Replay) void {
        r.store.deinit();
        std.debug.assert(r.family.live == 0);
    }
    pub fn request(r: *Replay, req: trace.Request) !Observation {
        var own: model.State = .{};
        const before = r.store.counts;
        const a = r.store.gpa;
        const plan = if (r.selected == .main and !r.adapter_main)
            try r.store.begin(a, req.tokens, req.history, &.{}, &.{}, &own, &.{})
        else
            try policy.begin(&r.store, r.selected, a, req.tokens, req.history, &own);
        defer a.free(plan.marks);
        if (own.at != plan.from) return error.RestorePosition;
        for (plan.marks) |at| {
            own.advance(req.tokens, at);
            const live = own;
            if (r.selected != .main or r.adapter_main) {
                policy.reserve(&r.store, r.selected, null, req.tokens, &.{at});
                if (!policy.roomForKeep(&r.store, req.tokens, at)) return error.UnexpectedNativeEviction;
            }
            if (!r.store.keep(req.tokens, at, &own, &.{}, &.{})) return error.KeepRefused;
            if (!std.meta.eql(live, own)) return error.LiveStateEvicted;
        }
        own.advance(req.tokens, @intCast(req.tokens.len));
        if (own.sum != req.hash) return error.ReplayStateMismatch;
        var held: u64 = 0;
        for (r.store.entries.items) |e| {
            held += e.bytes;
            if (e.borrowers != 0) return error.BorrowLeak;
        }
        if (held != r.store.held or held > r.store.budget or r.family.live != r.store.entries.items.len) return error.ByteAccounting;
        r.peak_bytes = @max(r.peak_bytes, held);
        return .{ .from = plan.from, .recompute = @intCast(req.tokens.len - plan.from), .held = held, .evicted = r.store.counts.evicted - before.evicted, .refused = r.store.counts.refused - before.refused, .kept = r.store.counts.kept - before.kept, .fingerprint = fingerprint(&r.store) };
    }
};
pub const Observation = struct { from: u32, recompute: u32, held: u64, evicted: u64, refused: u64, kept: u64, fingerprint: u64 };
pub fn fingerprint(s: *pc.Store) u64 {
    var h: u64 = s.clock;
    for (s.entries.items) |e| {
        for ([_]u64{ e.at, e.bytes, e.born, e.used, trace.hash(e.tokens), trace.hash(e.last), @intFromBool(e.shared), @intFromBool(e.branched), e.ext.items.len }) |v| h = h *% 31 +% v;
    }
    return h;
}
pub const Stats = struct {
    requests: u64 = 0,
    prompt_tokens: u64 = 0,
    recompute_tokens: u64 = 0,
    hits: u64 = 0,
    evicted: u64 = 0,
    refused: u64 = 0,
    costs: std.ArrayList(u32) = .empty,
    pub fn add(s: *Stats, a: std.mem.Allocator, req: trace.Request, got: Observation) !void {
        s.requests += 1;
        s.prompt_tokens += req.tokens.len;
        s.recompute_tokens += got.recompute;
        s.hits += @intFromBool(got.from > 0);
        s.evicted += got.evicted;
        s.refused += got.refused;
        try s.costs.append(a, got.recompute);
    }
    pub fn percentile(s: *Stats, p: u32) u32 {
        std.mem.sort(u32, s.costs.items, {}, std.sort.asc(u32));
        if (s.costs.items.len == 0) return 0;
        const at = (s.costs.items.len * p + 99) / 100;
        return s.costs.items[@max(at, 1) - 1];
    }
    pub fn deinit(s: *Stats, a: std.mem.Allocator) void {
        s.costs.deinit(a);
    }
};
