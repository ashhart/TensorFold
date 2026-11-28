//! Draft depth by expected tokens a millisecond (Python family_depth.py DraftDepth and FamilyRounds._head_depth).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
pub const pricedBatched = alloc.pricedBatched;
pub const Constants = @import("depth_constants.zig").Constants;
pub const HoldCost = @import("hold_price.zig").HoldCost;
const holdMs = @import("hold_price.zig").holdMs;
const hold_mod = @import("depth_hold.zig");

/// A stream's acceptance by depth and probe counters (Python `_depth_state[stream]`).
pub const State = struct {
    p: []f64,
    observed: []f64 = &.{}, // borrows p's allocation tail; only the tried prefix has been initialized
    tried: usize = 0, // prefix conditionals with an observed head trial
    hold_capped: bool = false, // an empirical cap stood the hold aside: keep the existing probe backoff live
    rounds: u64 = 0,
    last: ?i64 = null, // the depth this rule last chose: another must beat it by depth_margin
    probe: ?i64 = null, // a probe's depth for the next round (a lone window keeps it)
    plain: ?u64 = null,
    wait: ?u64 = null,
    ms: ?Table = null, // plain_guard: the stream's own round times
    copy_ms: ?Table = null, // opt-in copy prices, independent of the head depth rule
    copy_plain: ?u64 = null,
    copy_wait: ?u64 = null,
    copy_band: ?usize = null,
    copy_rounds: u64 = 0,

    pub fn deinit(st: *State, gpa: Allocator) void {
        gpa.free(st.p.ptr[0 .. st.p.len + st.observed.len]);
        if (st.ms) |*t| t.deinit(gpa);
        if (st.copy_ms) |*t| t.deinit(gpa);
    }
};

/// What the rule reads of a stream when it decides.
pub const Who = struct {
    state: *?State,
    draft_room: i64,
    finished: bool,
    forced: bool,
    granted: ?u32 = null,
    grant_short: bool = false, // the last grant left some held drafts unverified (the width did not bind)
    gain: ?f64 = null, // the drafts its last grant expected to keep (null: none priced)
    prices: []const f64 = &.{}, // fresh reserved prefix's joint chances (empty: only the generic aggregate is known)
    stream: ?*anyopaque = null,
    sampled: bool = false, // keyed draws: host work a row that a greedy round does not have
    row_ms: f64 = 0, // observed host verification work per row; unknown routes retain the optimistic zero model
    hold_context_rows: u32 = 0, // raw context still owed to the active drafter
    head_prior: []const f64 = &.{}, // initial conditional rates only; no grants, probes or context state
};

/// Whether a copied continuation is ahead of a stream (the round loop proposes; Python `_copy_proposal`).
pub const Probe = struct {
    ptr: *anyopaque,
    copy: *const fn (ptr: *anyopaque, who: *const Who) bool,
};

/// A shared round's time past its forward: `ms` without its head steps, `head` the steps' modeled time.
pub const Overhead = struct { streams: u32, ms: f64, head: f64 = 0.0 };

pub const Rule = struct {
    pub const copyRows = @import("copy_rule.zig").rows;
    gpa: Allocator,
    k: Constants = .{},
    prior: []const f64,
    most_drafts: u32,
    shared_cap: u32 = 0, // Model.shared_mtp_cap: most a shared round chains a stream; a lone one chains most_drafts (0: off)
    family_costs: *const Table,
    shared_costs: *const Table,
    mtp_step_ms: f64,
    draft_step_ms: f64 = 0, // a shared round's draft past a stream's first (budgets price it)
    plain_guard: bool,
    copy_guard: bool = false, // copy_rule.zig only; existing head pricing never reads it
    node_probabilities: bool,
    granted_depth: bool = false, // node_probabilities: a stream drafts its last grant plus draft_slack, lone ones too
    confidence_floor: ?f64 = null,
    plain_streams: u32 = 0, // a shared round with this many streams or more drafts none (0: off; a family's switch)
    batch_rows: u32,
    batched_heads: bool = false, // one batch drafts every stream of a shared round (steps: the deepest stream's)
    row_cost: @import("row_cost.zig").RowCost = .{}, // what a row costs past shared_costs here, under k.learn_rows
    round_ms: Table = .{}, // a round's wall time by drafts, every stream's: what sampled streams price by
    overhead: std.ArrayList(Overhead) = .empty, // a shared round's ms past its forward, by streams (insertion order)
    greedy_ms: Table = .{}, // round_ms from greedy rounds alone: what greedy streams price by
    shared_rounds: u64 = 0,
    greedy_overhead: std.ArrayList(Overhead) = .empty, // overhead from rounds without a sampled stream

    shared_price: @import("shared_price.zig").Prices = .{},

    hold_order: []usize = &.{}, // reused hold ordering, at most batch_rows entries
    offer_order: []@import("hold_offers.zig").Offer = &.{}, // cached hold values, at most batch_rows entries
    explore_price: ?@import("hold_price.zig").ProbeCache = null,

    pub fn deinit(r: *Rule) void {
        r.shared_price.deinit(r.gpa);
        r.gpa.free(r.hold_order);
        r.gpa.free(r.offer_order);
        r.row_cost.deinit(r.gpa);
        r.round_ms.deinit(r.gpa);
        r.overhead.deinit(r.gpa);
        r.greedy_ms.deinit(r.gpa);
        r.greedy_overhead.deinit(r.gpa);
    }

    pub fn rates(r: *Rule, who: Who) ![]f64 {
        if (who.state.* == null) {
            const prior = if (who.head_prior.len > 0) who.head_prior else r.prior;
            const n = @min(prior.len, @max(1, r.most_drafts));
            const data = try r.gpa.alloc(f64, n * @as(usize, if (r.granted_depth) 2 else 1));
            @memcpy(data[0..n], prior[0..n]);
            who.state.* = .{ .p = data[0..n], .observed = data[n..] };
        }
        return who.state.*.?.p;
    }

    /// Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves toward whether it was kept.
    pub fn observeDepth(r: *Rule, who: Who, proposed: i64, accepted: i64) !void {
        const p = try r.rates(who);
        const empirical = r.granted_depth and r.k.hold_costs.len > 0 and r.k.depth_rate > 0;
        const tried = who.state.*.?.tried;
        const observed = who.state.*.?.observed;
        const n = @min(@max(proposed, 0), @as(i64, @intCast(p.len)));
        var j: i64 = 0;
        while (j < n) : (j += 1) {
            if (accepted < j) break;
            const hit: f64 = if (accepted > j) 1.0 else 0.0;
            const at: usize = @intCast(j);
            p[at] += r.k.depth_rate * (hit - p[at]);
            if (empirical and at < observed.len) observed[at] = if (at >= tried) hit else observed[at] + r.k.depth_rate * (hit - observed[at]);
        }
        if (empirical) who.state.*.?.tried = @max(tried, @as(usize, @intCast(j)));
    }

    /// Round times by depth: the stream's own under `plain_guard`, else greedy rounds' for a greedy stream.
    fn costs(r: *Rule, who: ?Who) !*Table {
        const w = who orelse return &r.round_ms;
        if (!r.plain_guard) return if (w.sampled) &r.round_ms else &r.greedy_ms;
        _ = try r.rates(w);
        const st = &w.state.*.?;
        if (st.ms == null) st.ms = .{};
        return &st.ms.?;
    }

    /// A stream's first round carries the prefill-to-decode switch: its time stays out of the costs.
    pub fn observeCost(r: *Rule, drafts: i64, ms: f64, initializing: bool, who: ?Who) !void {
        if (drafts < 0 or (drafts == 0 and !r.plain_guard) or initializing) return;
        const table = try r.costs(who);
        try r.blend(table, drafts, ms);
        if (table == &r.greedy_ms) try r.blend(&r.round_ms, drafts, ms); // a greedy round counts for every stream too
    }

    fn blend(r: *Rule, table: *Table, drafts: i64, ms: f64) !void {
        const before = table.get(drafts);
        try table.put(r.gpa, drafts, if (before) |b| b + r.k.cost_rate * (ms - b) else ms);
    }

    /// A lone stream's `drafts`-deep head: its calibrated one-block hold pass when timed, else the step model.
    fn draftHeadMs(r: *const Rule, drafts: i64) f64 {
        if (drafts > 0 and r.k.hold_costs.len > 0) return holdMs(r.k.hold_costs, 1, drafts);
        const steps: f64 = if (r.k.fixed_draft_block) @floatFromInt(if (drafts > 0) r.most_drafts else 0) else @floatFromInt(drafts);
        return r.mtp_step_ms * steps;
    }

    /// Measured round time when there is one, else the load-time forward cost plus head steps.
    pub fn roundCost(r: *Rule, drafts: i64, who: ?Who) !?f64 {
        const table = try r.costs(who);
        if (table.get(drafts)) |v| return v;
        const forward = r.family_costs.get(drafts + 1) orelse return null;
        var modeled = forward + r.draftHeadMs(drafts);
        if (r.plain_guard) {
            // untimed: its model plus the least time a row of a timed round took beyond its model
            var least: ?f64 = null;
            for (table.slots, 0..) |slot, d| {
                const ms = slot orelse continue;
                const fc = r.family_costs.get(@intCast(d + 1)) orelse continue;
                const beyond = (ms - fc - r.draftHeadMs(@intCast(d))) / @as(f64, @floatFromInt(d + 1));
                least = if (least) |l| (if (beyond < l) beyond else l) else beyond;
            }
            const floor = least orelse 0.0;
            modeled += @as(f64, @floatFromInt(drafts + 1)) * (if (floor > 0.0) floor else 0.0);
        }
        if (drafts > 0) if (who) |w| {
            return modeled + @import("hold_price.zig").contextMs(r.k.hold_context_costs, 1, w.hold_context_rows);
        };
        return modeled;
    }

    /// The most expected tokens a ms, probing one depth farther every `depth_probe_every` rounds.
    pub fn depth(r: *Rule, who: Who) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        if (most <= 0) return 0;
        if (r.confidence_floor != null and !who.sampled and !r.k.floor_rates) return most;
        const p = try r.rates(who);
        if (r.family_costs.empty()) {
            const want: i64 = if (p[0] < 0.8) 1 else if (p[0] < 0.9) 2 else if (r.shared_cap != 0) r.most_drafts else 3;
            return @max(1, @min(most, want));
        }
        const st = try r.state(who);
        var best = (try r.bestRows(who, p, most, st.last)).rows;
        st.rounds += 1;
        if (best == 0) {
            st.last = null; // plain holds no depth: the next drafted depth is chosen fresh
            st.probe = null;
            return @intFromBool(r.probePlain(st));
        }
        st.plain = 0;
        st.wait = r.k.depth_probe_every;
        st.last = best;
        st.probe = null;
        if (best < most and st.rounds % r.k.depth_probe_every == 0) {
            best += 1;
            st.probe = best;
        }
        return best;
    }

    /// The stream's estimate state, allocated from its prior on first use.
    fn state(r: *Rule, who: Who) !*State {
        _ = try r.rates(who);
        return &who.state.*.?;
    }

    /// The rows with the most expected tokens a ms (0: plain); `hold` stays unless another beats it by depth_margin.
    fn bestRows(r: *Rule, who: Who, chances: []const f64, most: i64, hold: ?i64) !struct { rows: i64, drafted: i64 } {
        var best: i64 = 1;
        var best_rate: f64 = -1.0;
        if (r.plain_guard) {
            if (try r.roundCost(0, who)) |plain| {
                if (plain != 0.0) { // a draft must beat plain by plain_margin: estimates near the line are noise
                    best = 0;
                    best_rate = r.k.plain_margin / plain;
                }
            }
        }
        var drafted: i64 = 1;
        var drafted_rate: f64 = -1.0;
        var held_rate: f64 = -1.0;
        var expected: f64 = 1.0;
        var run: f64 = 1.0;
        var d: i64 = 1;
        while (d <= most) : (d += 1) {
            const cost = (try r.roundCost(d, who)) orelse break;
            const at: usize = @intCast(d - 1);
            run *= if (at < chances.len) chances[at] else chances[chances.len - 1];
            expected += run;
            const rate = expected / cost;
            if (hold != null and d == hold.?) held_rate = rate;
            if (rate > best_rate) {
                best = d;
                best_rate = rate;
            }
            if (rate > drafted_rate) {
                drafted = d;
                drafted_rate = rate;
            }
        }
        // a noisy round must not flip a held depth: another wins it only by depth_margin
        if (hold) |h| {
            if (best != 0 and h >= 1 and h <= most and h != best and held_rate >= 0.0 and best_rate < held_rate * r.k.depth_margin) best = h;
        }
        return .{ .rows = best, .drafted = drafted };
    }

    /// One draft after `wait` plain rounds in a row; each probe doubles the wait until drafting wins again.
    pub fn probePlain(r: *Rule, st: *State) bool {
        st.plain = (st.plain orelse 0) + 1;
        const w = st.wait orelse r.k.depth_probe_every;
        if (st.plain.? < w) return false;
        st.plain = 0;
        st.wait = @min(2 * w, r.k.plain_wait_most);
        return true;
    }

    /// Drafts past a grant: none when it stopped short of the held drafts (the width did not bind), else grant_slack.
    fn slack(r: *const Rule, who: Who) i64 {
        return if (who.grant_short) r.k.short_slack else r.k.grant_slack;
    }

    /// The last grant plus its slack (the deepest chain before any); after a grant of none, a probe of draft_slack.
    fn grantDepth(r: *Rule, who: Who, reserving: bool) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        const granted = who.granted orelse return most;
        _ = try r.rates(who);
        const st = &who.state.*.?;
        if (granted == 0) {
            const due = if (!r.plain_guard) true else if (reserving and r.k.hold_costs.len > 0)
                @import("hold_offers.zig").ready(st, r.k)
            else
                r.probePlain(st);
            return if (due) @min(most, r.k.draft_slack) else 0;
        }
        if (r.k.hold_costs.len == 0 or !st.hold_capped) {
            st.plain = 0;
            st.wait = r.k.depth_probe_every;
        }
        return @min(most, @as(i64, granted) + r.slack(who));
    }

    /// Python `_head_depth`: the next round's head drafts for a stream (0 when forced, 1 or none with a copy ahead).
    pub fn headDepth(r: *Rule, who: *const Who, budget: ?i64, probe: Probe) !i64 {
        return r.nextDepth(who, budget, probe, false);
    }

    /// A shared reservation only peeks: its funded hold, not the requested depth, spends a due trial.
    pub fn reserveDepth(r: *Rule, who: *const Who, budget: ?i64, probe: Probe) !i64 {
        return r.nextDepth(who, budget, probe, true);
    }

    fn nextDepth(r: *Rule, who: *const Who, budget: ?i64, probe: Probe, reserving: bool) !i64 {
        if (r.k.hold_costs.len > 0 and who.forced) return 0;
        const d: i64 = if (who.finished) 0 else if (budget) |b| b else if (r.granted_depth)
            try r.grantDepth(who.*, reserving)
        else if (r.node_probabilities and !((who.sampled or r.k.floor_rates) and r.confidence_floor != null))
            @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1))
        else
            try r.depth(who.*);
        if (who.forced) return 0;
        if (d != 0 and probe.copy(probe.ptr, who)) return @intFromBool(!r.k.copy_first);
        return d;
    }

    // Held-depth pricing and shared-round budgets live in depth_hold.zig; these keep the rule's call shape.
    pub const capPass = hold_mod.capPass;
    pub const payHolds = hold_mod.payHolds;
    pub const payHoldsBase = hold_mod.payHoldsBase;
    pub const payHoldsFor = hold_mod.payHoldsFor;
    pub const budgets = hold_mod.budgets;
    pub const budgetsFor = hold_mod.budgetsFor;
    pub const sharedPrice = hold_mod.sharedPrice;
    pub const verifyPrice = hold_mod.verifyPrice;
    pub const observeShared = hold_mod.observeShared;
    pub const observeSharedPrice = hold_mod.observeSharedPrice;
    pub const overheadMs = hold_mod.overheadMs;
    pub const fixedMs = hold_mod.fixedMs;
    pub const sharedStepMs = hold_mod.sharedStepMs;
    pub const headMs = hold_mod.headMs;
    pub const pricedHeadMs = hold_mod.pricedHeadMs;
    pub const observeOverhead = hold_mod.observeOverhead;
};

test {
    _ = @import("depth_test.zig");
    _ = @import("shared_price_test.zig");
    _ = @import("shared_verify_price_test.zig");
    _ = @import("shared_warm_price_test.zig");
    _ = @import("shared_calibration_test.zig");
    _ = @import("shared_warmup_test.zig");
    _ = @import("shared_head_price_test.zig");
    _ = @import("shared_price_learning_test.zig");
    _ = @import("shared_price_8_test.zig");
    _ = @import("shared_price_modes_test.zig");
}

test {
    _ = @import("shared_step_test.zig");
}
