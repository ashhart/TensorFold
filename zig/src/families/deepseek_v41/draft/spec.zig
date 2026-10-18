//! The next round's DSpark pass started after a verify window (TF_DSV41_SPEC_DRAFT=1): GPU drafts while host commits.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");

pub const decay: f64 = 0.8; // typed: 1 - decay rounds in f64
pub const min_rate: f64 = 0.3;
pub const probe = 8;

pub const Settings = struct {
    on: bool = false, // TF_DSV41_SPEC_DRAFT
    nucleus: bool = false, // TF_DSV41_SPEC_NUCLEUS: rounds with top-p-without-top-k rows speculate too

    pub fn fromEnv() !Settings {
        return parse(env("TF_DSV41_SPEC_DRAFT"), env("TF_DSV41_SPEC_NUCLEUS"));
    }

    /// The knob values: 0 | 1 | on | off | true | false (case-insensitive), empty is 0.
    pub fn parse(spec: ?[]const u8, nucleus: ?[]const u8) !Settings {
        return .{ .on = try flag(spec), .nucleus = try flag(nucleus) };
    }

    fn flag(v: ?[]const u8) !bool {
        const t = std.mem.trim(u8, v orelse return false, " \t\r\n");
        if (t.len == 0) return false;
        var low: [8]u8 = undefined;
        if (t.len > low.len) return error.BadSpecKnob;
        const l = std.ascii.lowerString(&low, t);
        for ([_][]const u8{ "1", "on", "true" }) |y| if (std.mem.eql(u8, l, y)) return true;
        for ([_][]const u8{ "0", "off", "false" }) |n| if (std.mem.eql(u8, l, n)) return false;
        return error.BadSpecKnob;
    }

    fn env(name: [*:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
    }
};

/// A nucleus row: T > 0, top_k 0 and 0 < top_p < 1.
pub fn isNucleus(s: ?lanes.Sampling) bool {
    const x = s orelse return false;
    return x.temperature > 0.0 and x.top_k == 0 and x.top_p > 0.0 and x.top_p < 1.0;
}

/// Why a speculation was dropped.
pub const Why = enum { reset, unused, slots, commit, pending, inputs };

pub const Stats = struct {
    launched: u64 = 0,
    hit: u64 = 0,
    miss: std.EnumArray(Why, u64) = .initFill(0),
    commit_differs: u64 = 0,
    /// commits whose ingest the speculation had done
    ingests_skipped: u64 = 0,
    /// slots speculated over every launch (each ingested its window's rows once)
    launched_slots: u64 = 0,

    pub fn missed(s: Stats) u64 {
        var n: u64 = 0;
        for (s.miss.values) |v| n += v;
        return n;
    }
};

/// A DSpark window of the round, as the speculation sees it.
pub const Candidate = struct {
    slot: u32,
    start: u64, // the window's start (its pending row's position)
    accepted: u32, // drafts the window's choices accept
    bonus: u32, // the choice after them: the next pending token
    sampling: ?lanes.Sampling,
};

const Win = struct { start: u64, accepted: u32, ask: iface.Ask, gen: u64, matched: ?bool = null };

pub const Spec = struct {
    gpa: Allocator,
    settings: Settings,
    block: usize,
    cands: usize, // gathered candidates a draft row (trees)
    rate: []f64,
    skipped: []u32,
    gen: []u64,
    /// the live speculation: its slots' windows (by slot), its asks in launch order
    wins: []?Win,
    asks: []iface.Ask,
    n: usize = 0,
    live: bool = false,
    /// a pass begun on the device and not collected yet (`Pass.begin`)
    begun: bool = false,
    /// its proposals, in launch order (sized for every slot)
    drafts: []u32,
    conf: []f32,
    cand: []i32,
    base: []f32,
    trees: bool = false, // the live pass gathered candidates and base logits
    stats: Stats = .{},

    pub fn init(gpa: Allocator, settings: Settings, slots: usize, block: usize, candidates: usize) !Spec {
        var s: Spec = .{ .gpa = gpa, .settings = settings, .block = block, .cands = candidates, .rate = &.{}, .skipped = &.{}, .gen = &.{}, .wins = &.{}, .asks = &.{}, .drafts = &.{}, .conf = &.{}, .cand = &.{}, .base = &.{} };
        errdefer s.deinit();
        s.rate = try gpa.alloc(f64, slots);
        @memset(s.rate, 1.0);
        s.skipped = try gpa.alloc(u32, slots);
        @memset(s.skipped, 0);
        s.gen = try gpa.alloc(u64, slots);
        @memset(s.gen, 0);
        s.wins = try gpa.alloc(?Win, slots);
        @memset(s.wins, null);
        s.asks = try gpa.alloc(iface.Ask, slots);
        s.drafts = try gpa.alloc(u32, slots * block);
        s.conf = try gpa.alloc(f32, slots * block);
        s.cand = try gpa.alloc(i32, slots * block * candidates);
        s.base = try gpa.alloc(f32, slots * block * candidates);
        return s;
    }

    pub fn deinit(s: *Spec) void {
        const gpa = s.gpa;
        gpa.free(s.rate);
        gpa.free(s.skipped);
        gpa.free(s.gen);
        gpa.free(s.wins);
        gpa.free(s.asks);
        gpa.free(s.drafts);
        gpa.free(s.conf);
        gpa.free(s.cand);
        gpa.free(s.base);
    }

    // -- bookkeeping -------------------------------------------------------------------------------------------------

    /// The slot was reset (a new request, a release): its speculation and rate start over.
    pub fn forget(s: *Spec, slot: u32) void {
        s.gen[slot] += 1;
        s.rate[slot] = 1.0;
        s.skipped[slot] = 0;
        if (s.live and s.wins[slot] != null) s.drop(.reset);
    }

    pub fn wanted(s: *Spec, slot: u32) bool {
        if (s.rate[slot] >= min_rate) return true;
        s.skipped[slot] += 1;
        if (s.skipped[slot] >= probe) {
            s.skipped[slot] = 0;
            return true;
        }
        return false;
    }

    /// `wanted` without counting a skip: whether a launch after this slot's coming window would take it (the arm).
    pub fn wouldWant(s: *const Spec, slot: u32) bool {
        return s.rate[slot] >= min_rate or s.skipped[slot] + 1 >= probe;
    }

    fn score(s: *Spec, hit: bool) void {
        for (s.asks[0..s.n]) |a| s.rate[a.slot] = decay * s.rate[a.slot] + (1.0 - decay) * @as(f64, if (hit) 1.0 else 0.0);
    }

    fn end(s: *Spec) void {
        for (s.asks[0..s.n]) |a| s.wins[a.slot] = null;
        s.n = 0;
        s.live = false;
    }

    fn drop(s: *Spec, why: Why) void {
        s.stats.miss.getPtr(why).* += 1;
        s.score(false);
        s.end();
    }

    /// A window without speculation: an unused one is dropped.
    pub fn clear(s: *Spec) void {
        if (s.live) s.drop(.unused);
    }

    /// Whether the round's windows may speculate at all (not tree rounds; nucleus rows only with the knob).
    pub fn roundAllowed(s: *const Spec, tree_round: bool, any_nucleus: bool) bool {
        return !tree_round and (!any_nucleus or s.settings.nucleus);
    }

    // -- launch (after the window) -----------------------------------------------------------------------------------

    /// Drops an unused speculation, takes the wanted candidates; returns their asks (caller ingests, then `run`s).
    pub fn launch(s: *Spec, cands: []const Candidate) []const iface.Ask {
        if (s.live) s.drop(.unused);
        for (cands) |c| {
            if (!s.wanted(c.slot)) continue;
            if (s.wins[c.slot] != null) continue; // one window a slot
            const next = c.start + c.accepted + 1;
            const ask: iface.Ask = .{ .slot = c.slot, .anchor = c.bonus, .start = next, .params = dspark.Params.of(c.sampling, next) };
            s.asks[s.n] = ask;
            s.wins[c.slot] = .{ .start = c.start, .accepted = c.accepted, .ask = ask, .gen = s.gen[c.slot] };
            s.n += 1;
        }
        if (s.n == 0) return &.{};
        s.live = true;
        s.stats.launched += 1;
        s.stats.launched_slots += s.n;
        return s.asks[0..s.n];
    }

    fn proposals(s: *Spec, out: []iface.Proposal, trees: bool) void {
        const b = s.block;
        const c = s.cands;
        for (out, 0..) |*p, i| p.* = .{
            .drafts = s.drafts[i * b ..][0..b],
            .conf = s.conf[i * b ..][0..b],
            .cand = if (trees) s.cand[i * b * c ..][0 .. b * c] else null,
            .base = if (trees) s.base[i * b * c ..][0 .. b * c] else null,
        };
    }

    /// The pass over the launched asks: begun on device if possible (`settle` collects), else now; `trees`: gather too.
    pub fn run(s: *Spec, pass: iface.Pass, trees: bool) !void {
        if (!s.live) return;
        s.trees = trees;
        if (pass.vtable.begin) |begin| {
            try begin(pass.ptr, s.asks[0..s.n]);
            s.begun = true;
            return;
        }
        var props: [64]iface.Proposal = undefined;
        if (s.n > props.len) return error.TooManySlots;
        s.proposals(props[0..s.n], trees);
        try pass.propose(s.asks[0..s.n], props[0..s.n]);
    }

    /// Collects a begun pass before other pass, target or ingest work (collective order); proposals stay for `take`.
    pub fn settle(s: *Spec, pass: iface.Pass) !void {
        if (!s.begun) return;
        var props: [64]iface.Proposal = undefined;
        s.proposals(props[0..s.n], s.trees);
        try pass.vtable.collect.?(pass.ptr, props[0..s.n]);
        s.begun = false;
    }

    // -- consumers (the next round) ----------------------------------------------------------------------------------

    /// At the commit's ingest: true when the speculation ingested this window's kept path (a chain 0..accepted).
    pub fn ingested(s: *Spec, slot: u32, start: u64, path: []const u32) bool {
        if (!s.live) return false;
        const w = &(s.wins[slot] orelse return false);
        if (w.start != start or w.gen != s.gen[slot]) return false;
        var ok = path.len == w.accepted + 1;
        if (ok) for (path, 0..) |r, i| {
            if (r != i) ok = false;
        };
        w.matched = ok;
        if (ok) s.stats.ingests_skipped += 1 else s.stats.commit_differs += 1;
        return ok;
    }

    /// At the draft: the speculated proposals into `out` (caller's ask order) and true, or false; the speculation ends.
    pub fn take(s: *Spec, asks: []const iface.Ask, out: []iface.Proposal) bool {
        if (!s.live) return false;
        const why: ?Why = check: {
            if (asks.len != s.n) break :check .slots;
            for (asks) |a| if (a.slot >= s.wins.len or s.wins[a.slot] == null) break :check .slots;
            for (asks) |a| {
                const w = s.wins[a.slot].?;
                if (w.matched != true or w.gen != s.gen[a.slot]) break :check .commit;
                if (w.ask.anchor != a.anchor or w.ask.start != a.start) break :check .pending;
                if (!std.meta.eql(w.ask.params, a.params)) break :check .inputs;
            }
            break :check null;
        };
        if (why) |y| {
            s.drop(y);
            return false;
        }
        const b = s.block;
        const c = s.cands;
        for (asks, out) |a, o| {
            const i = for (s.asks[0..s.n], 0..) |x, j| {
                if (x.slot == a.slot) break j;
            } else unreachable;
            @memcpy(o.drafts[0..b], s.drafts[i * b ..][0..b]);
            @memcpy(o.conf[0..b], s.conf[i * b ..][0..b]);
            if (o.cand) |cd| @memcpy(cd[0 .. b * c], s.cand[i * b * c ..][0 .. b * c]);
            if (o.base) |bs| @memcpy(bs[0 .. b * c], s.base[i * b * c ..][0 .. b * c]);
        }
        s.score(true);
        s.stats.hit += 1;
        s.end();
        return true;
    }

    /// "speculative DSpark passes L launched, H used, M dropped (why n, ...)".
    pub fn describe(s: *const Spec, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        const st = s.stats;
        w.print("speculative DSpark passes {d} launched, {d} used, {d} dropped", .{ st.launched, st.hit, st.missed() }) catch return w.buffered();
        var first = true;
        for (std.enums.values(Why)) |y| {
            const v = st.miss.get(y);
            if (v == 0) continue;
            w.print("{s}{t} {d}", .{ if (first) " (" else ", ", y, v }) catch return w.buffered();
            first = false;
        }
        if (!first) w.writeAll(")") catch return w.buffered();
        w.print(", {d} commit ingests skipped", .{st.ingests_skipped}) catch {};
        return w.buffered();
    }
};
