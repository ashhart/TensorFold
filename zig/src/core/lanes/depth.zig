//! Draft depth by expected tokens a millisecond (Python family_depth.py DraftDepth and FamilyRounds._head_depth).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
pub const pricedBatched = alloc.pricedBatched;
pub const Constants = @import("depth_constants.zig").Constants;
pub const HoldCost = @import("hold_price.zig").HoldCost;
const holdMs = @import("hold_price.zig").holdMs;

/// A stream's acceptance by depth and probe counters (Python `_depth_state[stream]`).
pub const State = struct {
    p: []f64,
    rounds: u64 = 0,
    plain: ?u64 = null,
    wait: ?u64 = null,
    ms: ?Table = null, // plain_guard: the stream's own round times

    pub fn deinit(st: *State, gpa: Allocator) void {
        gpa.free(st.p);
        if (st.ms) |*t| t.deinit(gpa);
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
    stream: ?*anyopaque = null,
    sampled: bool = false, // keyed draws: host work a row that a greedy round does not have
};

/// Whether a copied continuation is ahead of a stream (the round loop proposes; Python `_copy_proposal`).
pub const Probe = struct {
    ptr: *anyopaque,
    copy: *const fn (ptr: *anyopaque, who: *const Who) bool,
};

/// A shared round's time past its forward: `ms` without its head steps, `head` the steps' modeled time.
pub const Overhead = struct { streams: u32, ms: f64, head: f64 = 0.0 };

pub const Rule = struct {
    gpa: Allocator,
    k: Constants = .{},
    prior: []const f64,
    most_drafts: u32,
    family_costs: *const Table,
    shared_costs: *const Table,
    mtp_step_ms: f64,
    draft_step_ms: f64 = 0, // a shared round's draft past a stream's first (budgets price it)
    plain_guard: bool,
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

    pub fn deinit(r: *Rule) void {
        r.shared_price.deinit(r.gpa);
        r.gpa.free(r.hold_order);
        r.row_cost.deinit(r.gpa);
        r.round_ms.deinit(r.gpa);
        r.overhead.deinit(r.gpa);
        r.greedy_ms.deinit(r.gpa);
        r.greedy_overhead.deinit(r.gpa);
    }

    pub fn rates(r: *Rule, who: Who) ![]f64 {
        if (who.state.* == null) {
            const n = @min(r.prior.len, @max(1, r.most_drafts));
            who.state.* = .{ .p = try r.gpa.dupe(f64, r.prior[0..n]) };
        }
        return who.state.*.?.p;
    }

    /// Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves toward whether it was kept.
    pub fn observeDepth(r: *Rule, who: Who, proposed: i64, accepted: i64) !void {
        const p = try r.rates(who);
        const n = @min(@max(proposed, 0), @as(i64, @intCast(p.len)));
        var j: i64 = 0;
        while (j < n) : (j += 1) {
            if (accepted < j) break;
            const hit: f64 = if (accepted > j) 1.0 else 0.0;
            const at: usize = @intCast(j);
            p[at] += r.k.depth_rate * (hit - p[at]);
        }
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

    /// Measured round time when there is one, else the load-time forward cost plus head steps.
    pub fn roundCost(r: *Rule, drafts: i64, who: ?Who) !?f64 {
        const table = try r.costs(who);
        if (table.get(drafts)) |v| return v;
        const forward = r.family_costs.get(drafts + 1) orelse return null;
        var modeled = forward + r.mtp_step_ms * @as(f64, @floatFromInt(drafts));
        if (r.plain_guard) {
            // untimed: its model plus the least time a row of a timed round took beyond its model
            var least: ?f64 = null;
            for (table.slots, 0..) |slot, d| {
                const ms = slot orelse continue;
                const fc = r.family_costs.get(@intCast(d + 1)) orelse continue;
                const beyond = (ms - fc - r.mtp_step_ms * @as(f64, @floatFromInt(d))) / @as(f64, @floatFromInt(d + 1));
                least = if (least) |l| (if (beyond < l) beyond else l) else beyond;
            }
            const floor = least orelse 0.0;
            modeled += @as(f64, @floatFromInt(drafts + 1)) * (if (floor > 0.0) floor else 0.0);
        }
        return modeled;
    }

    /// The most expected tokens a ms, probing one depth farther every `depth_probe_every` rounds.
    pub fn depth(r: *Rule, who: Who) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        if (most <= 0) return 0;
        if (r.confidence_floor != null and !who.sampled and !r.k.floor_rates) return most;
        const p = try r.rates(who);
        if (r.family_costs.empty()) {
            const want: i64 = if (p[0] < 0.8) 1 else if (p[0] < 0.9) 2 else 3;
            return @max(1, @min(most, want));
        }
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
        var expected: f64 = 1.0;
        var run: f64 = 1.0;
        var d: i64 = 1;
        while (d <= most) : (d += 1) {
            const cost = (try r.roundCost(d, who)) orelse break;
            const at: usize = @intCast(d - 1);
            run *= if (at < p.len) p[at] else p[p.len - 1];
            expected += run;
            if (expected / cost > best_rate) {
                best = d;
                best_rate = expected / cost;
            }
        }
        const st = &who.state.*.?;
        st.rounds += 1;
        if (best == 0) return @intFromBool(r.probePlain(st));
        st.plain = 0;
        st.wait = r.k.depth_probe_every;
        if (best < most and st.rounds % r.k.depth_probe_every == 0) best += 1;
        return best;
    }

    /// One draft after `wait` plain rounds in a row; each probe doubles the wait until drafting wins again.
    fn probePlain(r: *Rule, st: *State) bool {
        st.plain = (st.plain orelse 0) + 1;
        const wait = st.wait orelse r.k.depth_probe_every;
        if (st.plain.? < wait) return false;
        st.plain = 0;
        st.wait = @min(2 * wait, r.k.plain_wait_most);
        return true;
    }

    /// Drafts past a grant: none when it stopped short of the held drafts (the width did not bind), else grant_slack.
    fn slack(r: *const Rule, who: Who) i64 {
        return if (who.grant_short) r.k.short_slack else r.k.grant_slack;
    }

    /// The last grant plus its slack (the deepest chain before any); after a grant of none, a probe of draft_slack.
    fn grantDepth(r: *Rule, who: Who) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        const granted = who.granted orelse return most;
        _ = try r.rates(who);
        const st = &who.state.*.?;
        if (granted == 0) return if (!r.plain_guard or r.probePlain(st)) @min(most, r.k.draft_slack) else 0;
        st.plain = 0;
        st.wait = r.k.depth_probe_every;
        return @min(most, @as(i64, granted) + r.slack(who));
    }

    /// Python `_head_depth`: the next round's head drafts for a stream (0 when forced, 1 or none with a copy ahead).
    pub fn headDepth(r: *Rule, who: *const Who, budget: ?i64, probe: Probe) !i64 {
        const d: i64 = if (who.finished) 0 else if (budget) |b| b else if (r.granted_depth)
            try r.grantDepth(who.*)
        else if (r.node_probabilities and !((who.sampled or r.k.floor_rates) and r.confidence_floor != null))
            @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1))
        else
            try r.depth(who.*);
        if (who.forced) return 0;
        if (d != 0 and probe.copy(probe.ptr, who)) return @intFromBool(!r.k.copy_first);
        return d;
    }

    /// A shared hold pads every block to its deepest: depths cut so the drafting streams' blocks fit `rows` (one pass).
    pub fn capPass(out: []i64, rows: u32) void {
        var n: i64 = 0;
        for (out) |d| n += @intFromBool(d > 0);
        if (n == 0) return;
        const most = @max(1, @divFloor(@as(i64, rows), n));
        for (out) |*d| d.* = @min(d.*, most);
    }

    /// Granted holds kept, most expected drafts a row first, while they raise the next round's expected tokens a ms.
    pub fn payHolds(r: *Rule, whos: []const Who, sampled: bool, out: []i64) !void {
        return r.payHoldsFor(whos, sampled, out, whos.len);
    }

    pub fn payHoldsFor(r: *Rule, whos: []const Who, sampled: bool, out: []i64, total_streams: usize) !void {
        if (whos.len > r.batch_rows) return error.BadHoldStreamCount;
        if (r.hold_order.len == 0) r.hold_order = try r.gpa.alloc(usize, r.batch_rows);
        const order = r.hold_order[0..whos.len];
        var m: usize = 0;
        var probes: usize = 0; // a probe after a grant of none keeps its hold: it measures the stream again
        var deepest: i64 = 0;
        for (whos, out, 0..) |w, d, i| {
            if (d <= 0) continue;
            if (w.granted == @as(?u32, 0)) {
                probes += 1;
                deepest = @max(deepest, d);
            } else {
                order[m] = i;
                m += 1;
            }
        }
        if (m == 0) return;
        const Ctx = struct {
            r: *Rule,
            whos: []const Who,
            fn less(c: @This(), x: usize, y: usize) bool {
                const gx = c.r.gainOf(c.whos[x]) catch 0;
                const gy = c.r.gainOf(c.whos[y]) catch 0;
                return gx * rowsOf(c.whos[y]) > gy * rowsOf(c.whos[x]) or (gx * rowsOf(c.whos[y]) == gy * rowsOf(c.whos[x]) and x < y);
            }
        };
        std.mem.sort(usize, order[0..m], Ctx{ .r = r, .whos = whos }, Ctx.less);
        const paired = r.sharedPrice(total_streams, sampled);
        const streams = if (paired != null) total_streams else whos.len;
        const fixed = if (paired) |p| p.fixed else r.fixedMs(streams, sampled);
        var rows: f64 = @floatFromInt(streams);
        var tokens = rows;
        const plain = r.shared_costs.get(@intCast(streams)) orelse return;
        const probe_ms = if (probes > 0) holdMs(r.k.hold_costs, probes, deepest) else 0;
        var best = tokens / (plain + (if (paired) |p| p.plain else fixed) + probe_ms);
        var keep: usize = 0;
        for (order[0..m], 1..) |i, k| {
            rows += rowsOf(whos[i]);
            tokens += try r.gainOf(whos[i]);
            deepest = @max(deepest, out[i]);
            const verify = r.shared_costs.get(@intFromFloat(@ceil(rows))) orelse break;
            const extra = (if (paired) |p| p.row else r.row_cost.extraMs(streams, sampled)) * (rows - @as(f64, @floatFromInt(streams)));
            const now = tokens / (verify + (if (extra < 0) @max(extra, plain - verify) else extra) + fixed + holdMs(r.k.hold_costs, k + probes, deepest));
            if (now > best) {
                best = now;
                keep = k;
            }
        }
        for (order[keep..m]) |i| out[i] = 0;
    }

    /// The drafts a granted stream's next round is expected to keep: its last grant's, or its first draft's rate.
    fn gainOf(r: *Rule, w: Who) !f64 {
        if (w.gain) |g| if (w.granted != null) return g;
        return (try r.rates(w))[0];
    }

    fn rowsOf(w: Who) f64 {
        return @floatFromInt(@max(1, w.granted orelse 1));
    }

    /// Head drafts for the streams of a shared round, sized by allocation over their chain chances.
    pub fn budgets(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64) !void {
        return r.budgetsFor(whos, probe, sampled, out, whos.len);
    }

    /// Non-head streams still contribute their plain rows to a shared round's width and exact-count price.
    pub fn budgetsFor(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64, total_streams: usize) !void {
        const paired = r.sharedPrice(total_streams, sampled);
        const streams = if (paired != null) total_streams else whos.len;
        // drafts lose here: none asked, no head pass
        if (r.plain_streams > 0 and whos.len >= r.plain_streams) return @memset(out, 0);
        if (r.confidence_floor != null) {
            for (whos, out) |*w, *o| o.* = try r.headDepth(w, null, probe);
            return;
        }
        if (r.node_probabilities) {
            for (whos, out) |*w, *o| {
                const inner = try r.headDepth(w, null, probe);
                const granted: i64 = if (w.granted) |g| g else r.most_drafts;
                o.* = if (r.granted_depth) inner else try r.headDepth(w, @min(inner, granted + r.k.draft_slack), probe);
            }
            if (r.granted_depth) capPass(out, r.batch_rows);
            if (r.granted_depth and r.k.hold_costs.len > 0) try r.payHoldsFor(whos, sampled, out, total_streams);
            return;
        }
        const most = try r.gpa.alloc(i64, whos.len);
        defer r.gpa.free(most);
        const probs = try r.gpa.alloc([]f64, streams);
        @memset(probs, &.{});
        defer r.gpa.free(probs);
        var made: usize = 0;
        defer for (probs[0..made]) |p| r.gpa.free(p);
        for (whos, most, probs[0..whos.len]) |w, *m, *p| {
            m.* = @min(@as(i64, r.most_drafts), @max(1, w.draft_room - 1));
            p.* = try alloc.chainProbabilities(r.gpa, try r.rates(w), @intCast(m.*));
            made += 1;
        }
        const ones = try r.gpa.alloc(u32, streams);
        defer r.gpa.free(ones);
        @memset(ones, 1);
        const head: alloc.HeadCost = .{ .step_ms = r.sharedStepMs(), .batched = r.batched_heads, .row_ms = r.row_cost.extraMs(whos.len, sampled), .least_rows = r.row_cost.leastRows(whos.len, sampled) };
        const all_counts = if (paired) |p|
            try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, p.fixed, r.row_cost.capRows(streams, sampled, @max(streams, r.batch_rows)), .{ .plain_ms = p.plain, .step_ms = r.sharedStepMs(), .extra_step_ms = r.draft_step_ms, .batched = r.batched_heads and pricedBatched(), .row_ms = p.row, .least_rows = r.row_cost.leastRows(streams, sampled) })
        else if (r.draft_step_ms != 0)
            try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, r.overheadMs(whos.len, sampled), r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), .{ .step_ms = r.draft_step_ms, .batched = r.batched_heads and pricedBatched(), .free_first = true, .row_ms = r.row_cost.extraMs(whos.len, sampled), .least_rows = head.least_rows })
        else
            try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, r.fixedMs(whos.len, sampled), r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), head);
        defer r.gpa.free(all_counts);
        const counts = all_counts[0..whos.len];
        if (streams == whos.len) r.probeShared(whos, most, counts);
        if (r.plain_guard) {
            // no draft where none pays; a probe after a growing run of plain rounds
            for (whos, counts) |w, *c| {
                const st = &w.state.*.?;
                st.rounds += 1;
                if (c.* != 0) {
                    st.plain = 0;
                    st.wait = r.k.depth_probe_every;
                } else c.* = @intFromBool(r.probePlain(st));
            }
            for (whos, most, counts, out) |*w, m, c, *o| o.* = try r.headDepth(w, @min(m, @as(i64, c)), probe);
            return;
        }
        for (whos, most, counts, out) |*w, m, c, *o| o.* = try r.headDepth(w, @min(m, @max(1, @as(i64, c))), probe);
    }

    // One extra offered row, rotating streams; the normal window allocator still prices and trims it.
    fn probeShared(r: *Rule, whos: []const Who, most: []const i64, counts: []u32) void {
        r.shared_rounds +%= 1;
        const every = r.k.depth_probe_every;
        if (whos.len == 0 or every == 0 or r.shared_rounds % every != 0) return;
        var rows: u64 = whos.len;
        for (counts) |c| rows += c;
        if (rows >= @max(whos.len, r.batch_rows)) return;
        const first: usize = @intCast((r.shared_rounds / every -% 1) % whos.len);
        for (0..whos.len) |offset| {
            const i = (first + offset) % whos.len;
            if (whos[i].finished or whos[i].forced or counts[i] == 0 or counts[i] >= most[i]) continue;
            counts[i] += 1;
            return;
        }
    }

    /// Independent plain price and the row slope from a plain/drafted pair, at this exact count and mode.
    pub fn sharedPrice(r: *const Rule, streams: usize, sampled: bool) ?@import("shared_price.zig").Price {
        if (streams < 2 or !r.plain_guard or r.confidence_floor != null) return null;
        return r.shared_price.price(streams, sampled, r.row_cost.extraMs(streams, sampled));
    }

    /// Held drafts have already run; every verify option retains the round's same head fixed cost.
    pub fn verifyPrice(r: *const Rule, streams: usize, sampled: bool) ?@import("shared_price.zig").Price {
        var price = r.sharedPrice(streams, sampled) orelse return null;
        const head = if (r.nearest(streams, sampled)) |o| o.head else 0;
        price.plain += head;
        price.fixed += head;
        return price;
    }

    /// Only completed timed shared rounds advance this count/mode's calibration clock.
    pub fn observeShared(r: *Rule, streams: u32, rows: u64, ms: f64, sampled: bool, head_ms: f64, plain: bool, calibration: bool) !void {
        try r.observeSharedPrice(streams, rows, ms, sampled, head_ms, plain, calibration);
        try r.observeOverhead(streams, rows, ms, sampled, head_ms);
    }

    /// The independent residual uses the same batching policy as next-draft allocation.
    pub fn observeSharedPrice(r: *Rule, streams: u32, rows: u64, ms: f64, sampled: bool, head_ms: f64, plain: bool, calibration: bool) !void {
        if (streams > 1 and r.plain_guard and r.confidence_floor == null) {
            const residual = if (r.shared_costs.get(@intCast(rows))) |forward| ms - forward - head_ms else std.math.nan(f64);
            try r.shared_price.observe(r.gpa, streams, rows, residual, sampled, plain, calibration);
        }
    }

    /// A shared round's ms beyond its forward at this many streams (the nearest measured count, first seen on ties).
    pub fn overheadMs(r: *const Rule, streams: usize, sampled: bool) f64 {
        const o = r.nearest(streams, sampled) orelse return 8.0;
        return o.ms + o.head;
    }

    /// overheadMs without the head steps the measured rounds drafted.
    pub fn fixedMs(r: *const Rule, streams: usize, sampled: bool) f64 {
        const o = r.nearest(streams, sampled) orelse return 8.0;
        return o.ms;
    }

    fn nearest(r: *const Rule, streams: usize, sampled: bool) ?Overhead {
        const list = if (sampled) r.overhead.items else r.greedy_overhead.items;
        if (list.len == 0) return null;
        var best = list[0];
        for (list[1..]) |o| {
            if (distance(o.streams, streams) < distance(best.streams, streams)) best = o;
        }
        return best;
    }

    /// Shared pricing can retain a family policy while lone rounds use a measured head step.
    pub fn sharedStepMs(r: *const Rule) f64 {
        return r.k.shared_step_ms orelse r.mtp_step_ms;
    }

    /// The modeled time of a shared round's head steps for the depths it drafted.
    pub fn headMs(r: *const Rule, depths: []const i64) f64 {
        return r.headCost(depths, r.batched_heads);
    }

    pub fn pricedHeadMs(r: *const Rule, depths: []const i64) f64 {
        return r.headCost(depths, r.batched_heads and pricedBatched());
    }

    fn headCost(r: *const Rule, depths: []const i64, batched: bool) f64 {
        var steps: i64 = 0;
        for (depths) |d| steps = if (batched) @max(steps, d) else steps + d;
        return r.sharedStepMs() * @as(f64, @floatFromInt(steps));
    }

    /// `sampled`: the round drew keyed samples, so it counts for sampled streams' overhead alone; `head_ms` its drafts.
    pub fn observeOverhead(r: *Rule, streams: u32, rows: u64, ms: f64, sampled: bool, head_ms: f64) !void {
        const forward = r.shared_costs.get(@intCast(rows)) orelse return;
        if (r.k.learn_rows) try r.row_cost.observe(r.gpa, streams, rows, ms - forward - head_ms, sampled);
        // rows the table underprices are no fixed cost: called fixed, they would make the next draft look cheaper
        const row_ms = r.row_cost.extraMs(streams, sampled);
        // no fit yet: a wide round's excess may be its rows, and adding it to the overhead draws the next round wider
        if (r.k.learn_rows and streams > 1 and row_ms == 0 and rows > streams) return;
        const paid = alloc.rowTerm(r.shared_costs, true, .{ .row_ms = row_ms }, rows, streams);
        const extra = if (ms - forward - paid > 0.0) ms - forward - paid else 0.0;
        const fixed = if (extra - head_ms > 0.0) extra - head_ms else 0.0;
        try r.addOverhead(&r.overhead, streams, fixed, extra - fixed);
        if (!sampled) try r.addOverhead(&r.greedy_overhead, streams, fixed, extra - fixed);
    }

    fn addOverhead(r: *Rule, list: *std.ArrayList(Overhead), streams: u32, fixed: f64, head: f64) !void {
        for (list.items) |*o| {
            if (o.streams == streams) {
                o.ms = o.ms + 0.2 * (fixed - o.ms);
                o.head = o.head + 0.2 * (head - o.head);
                return;
            }
        }
        try list.append(r.gpa, .{ .streams = streams, .ms = fixed, .head = head });
    }
};

fn distance(a: u32, b: usize) usize {
    const x: usize = a;
    return if (x > b) x - b else b - x;
}

test {
    _ = @import("depth_test.zig");
    _ = @import("shared_price_test.zig");
    _ = @import("shared_verify_price_test.zig");
    _ = @import("shared_warm_price_test.zig");
    _ = @import("shared_calibration_test.zig");
    _ = @import("shared_warmup_test.zig");
    _ = @import("shared_head_price_test.zig");
}

test {
    _ = @import("shared_step_test.zig");
}
