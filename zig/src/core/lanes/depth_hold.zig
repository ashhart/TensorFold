//! A rule's held-depth pricing: grant payback, timed hold grids and a shared round's overhead by streams.
const std = @import("std");
const depth = @import("depth.zig");
const alloc = @import("allocate.zig");
const hold_price = @import("hold_price.zig");
const hold_offers = @import("hold_offers.zig");
const shared_price = @import("shared_price.zig");
const Rule = depth.Rule;
const Who = depth.Who;
const Probe = depth.Probe;
const Overhead = depth.Overhead;

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

/// Reserved head prefixes share the full cohort's target rows and one padded hold pass.
pub fn payHoldsBase(r: *Rule, whos: []const Who, sampled: bool, out: []i64, base: hold_price.Base) !void {
    var priced = base;
    priced.row_ms = r.row_cost.extraMs(base.streams, sampled);
    return hold_offers.pay(r, whos, sampled, out, priced);
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
            const gx = gainOf(c.r, c.whos[x]) catch 0;
            const gy = gainOf(c.r, c.whos[y]) catch 0;
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
    const probe_ms = if (probes > 0) hold_price.holdMs(r.k.hold_costs, probes, deepest) else 0;
    var best = tokens / (plain + (if (paired) |p| p.plain else fixed) + probe_ms);
    var keep: usize = 0;
    for (order[0..m], 1..) |i, k| {
        rows += rowsOf(whos[i]);
        tokens += try gainOf(r, whos[i]);
        deepest = @max(deepest, out[i]);
        const verify = r.shared_costs.get(@intFromFloat(@ceil(rows))) orelse break;
        const extra = (if (paired) |p| p.row else r.row_cost.extraMs(streams, sampled)) * (rows - @as(f64, @floatFromInt(streams)));
        const now = tokens / (verify + (if (extra < 0) @max(extra, plain - verify) else extra) + fixed + hold_price.holdMs(r.k.hold_costs, k + probes, deepest));
        if (now > best) {
            best = now;
            keep = k;
        }
    }
    for (order[keep..m]) |i| out[i] = 0;
}

/// The drafts a granted stream's next round is expected to keep: its last grant's, or its first draft's rate.
pub fn gainOf(r: *Rule, w: Who) !f64 {
    if (w.gain) |g| if (w.granted != null) return g;
    return (try r.rates(w))[0];
}

pub fn rowsOf(w: Who) f64 {
    return @floatFromInt(@max(1, w.granted orelse 1));
}

/// Head drafts for the streams of a shared round, sized by allocation over their chain chances.
pub fn budgets(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64) !void {
    return budgetsFor(r, whos, probe, sampled, out, whos.len);
}

/// Non-head streams still contribute their plain rows to a shared round's width and exact-count price.
pub fn budgetsFor(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64, total_streams: usize) !void {
    const paired = r.sharedPrice(total_streams, sampled);
    const streams = if (paired != null) total_streams else whos.len;
    // drafts lose here: none asked, no head pass
    if (r.plain_streams > 0 and whos.len >= r.plain_streams) return @memset(out, 0);
    const top: i64 = if (r.shared_cap != 0) @min(r.most_drafts, r.shared_cap) else r.most_drafts;
    if (r.confidence_floor != null) {
        for (whos, out) |*w, *o| o.* = @min(try r.headDepth(w, null, probe), top);
        return;
    }
    if (r.node_probabilities) {
        const priced_holds = r.k.hold_costs.len > 0;
        for (whos, out) |*w, *o| {
            const inner = @min((if (priced_holds) try r.reserveDepth(w, null, probe) else try r.headDepth(w, null, probe)), top);
            o.* = if (r.granted_depth) inner else try r.headDepth(w, @min(inner, @as(i64, w.granted orelse r.most_drafts) + r.k.draft_slack), probe);
        }
        if (r.granted_depth) capPass(out, r.batch_rows);
        if (r.granted_depth and priced_holds) {
            var base = hold_price.Base.pending(0);
            for (whos) |w| {
                if (w.finished) continue;
                base.streams += 1;
                base.rows += 1;
                base.tokens += 1;
                base.host_ms += w.row_ms;
            }
            try r.payHoldsBase(whos, sampled, out, base);
        }
        return;
    }
    if (r.k.hold_costs.len > 0) return budgetsHold(r, whos, probe, sampled, out, top);
    const most = try r.gpa.alloc(i64, whos.len);
    defer r.gpa.free(most);
    const probs = try r.gpa.alloc([]f64, streams);
    @memset(probs, &.{});
    defer r.gpa.free(probs);
    var made: usize = 0;
    defer for (probs[0..made]) |p| r.gpa.free(p);
    for (whos, most, probs[0..whos.len]) |w, *m, *p| {
        m.* = @min(top, @max(1, w.draft_room - 1));
        p.* = try alloc.chainProbabilities(r.gpa, try r.rates(w), @intCast(m.*));
        made += 1;
    }
    const ones = try r.gpa.alloc(u32, streams);
    defer r.gpa.free(ones);
    @memset(ones, 1);
    const head: alloc.HeadCost = .{ .step_ms = r.sharedStepMs(), .batched = r.batched_heads, .row_ms = r.row_cost.extraMs(whos.len, sampled), .least_rows = r.row_cost.leastRows(whos.len, sampled) };
    const all_counts = if (paired) |p|
        try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, if (r.k.fixed_draft_block) p.plain else p.fixed, r.row_cost.capRows(streams, sampled, @max(streams, r.batch_rows)), .{ .plain_ms = p.plain, .block_ms = if (r.k.fixed_draft_block) p.fixed + p.head - p.plain else 0, .step_ms = if (r.k.fixed_draft_block) 0 else r.sharedStepMs(), .extra_step_ms = if (r.k.fixed_draft_block) 0 else r.draft_step_ms, .batched = r.batched_heads and alloc.pricedBatched(), .row_ms = p.row, .least_rows = r.row_cost.leastRows(streams, sampled) })
    else if (r.draft_step_ms != 0)
        try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, r.overheadMs(whos.len, sampled), r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), .{ .step_ms = r.draft_step_ms, .batched = r.batched_heads and alloc.pricedBatched(), .free_first = true, .row_ms = r.row_cost.extraMs(whos.len, sampled), .least_rows = head.least_rows })
    else
        try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, r.fixedMs(whos.len, sampled), r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), head);
    defer r.gpa.free(all_counts);
    const counts = all_counts[0..whos.len];
    if (streams == whos.len) probeShared(r, whos, most, counts);
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
pub fn probeShared(r: *Rule, whos: []const Who, most: []const i64, counts: []u32) void {
    r.shared_rounds +%= 1;
    const every = r.k.depth_probe_every;
    if (whos.len == 0 or every == 0 or r.shared_rounds % every != 0) return;
    var rows: u64 = whos.len;
    if (r.k.hold_costs.len > 0) {
        rows = 0;
        for (whos) |w| rows += @intFromBool(!w.finished);
    }
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

/// Head drafts under a timed hold grid: finished and forced streams reserve no head rows.
pub fn budgetsHold(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64, top: i64) !void {
    const most = try r.gpa.alloc(i64, whos.len);
    defer r.gpa.free(most);
    const probs = try r.gpa.alloc([]f64, whos.len);
    defer r.gpa.free(probs);
    var made: usize = 0;
    defer for (probs[0..made]) |p| r.gpa.free(p);
    for (whos, most, probs) |w, *m, *p| {
        const excluded = w.finished or w.forced;
        m.* = if (excluded) 0 else @min(top, @max(1, w.draft_room - 1));
        p.* = try alloc.chainProbabilities(r.gpa, if (excluded) r.prior else try r.rates(w), @intCast(m.*));
        made += 1;
    }
    const work = try r.gpa.alloc(u32, whos.len * @as(usize, if (r.k.hold_context_costs.len > 0) 2 else 1));
    defer r.gpa.free(work);
    const ones = work[0..whos.len];
    var row_ms: []f64 = &.{};
    defer r.gpa.free(row_ms);
    var cost_streams: usize = 0;
    for (whos, ones, 0..) |w, *fixed, i| {
        fixed.* = @intFromBool(!w.finished);
        cost_streams += fixed.*;
        if (w.finished) continue;
        if (w.row_ms > 0 and row_ms.len == 0) {
            row_ms = try r.gpa.alloc(f64, whos.len);
            @memset(row_ms, 0);
        }
        if (row_ms.len > 0) row_ms[i] = w.row_ms;
    }
    const own_steps = r.draft_step_ms != 0; // the backend's own draft step prices a stream's drafts past its first
    const context_rows = work[whos.len..];
    for (context_rows, 0..) |*rows, i| rows.* = whos[i].hold_context_rows;
    const extra_row_ms = r.row_cost.extraMs(cost_streams, sampled);
    const least_rows = r.row_cost.leastRows(cost_streams, sampled);
    const head: alloc.HeadCost = if (own_steps)
        .{ .step_ms = r.draft_step_ms, .batched = r.batched_heads and alloc.pricedBatched(), .free_first = true, .hold_costs = r.k.hold_costs, .hold_context_costs = r.k.hold_context_costs, .hold_context_rows = context_rows, .hold_rows = r.k.hold_rows, .hold_block_rows = r.k.hold_block_rows, .host_row_ms = row_ms, .row_ms = extra_row_ms, .least_rows = least_rows }
    else
        .{ .step_ms = r.sharedStepMs(), .batched = r.batched_heads, .hold_costs = r.k.hold_costs, .hold_context_costs = r.k.hold_context_costs, .hold_context_rows = context_rows, .hold_rows = r.k.hold_rows, .hold_block_rows = r.k.hold_block_rows, .host_row_ms = row_ms, .row_ms = extra_row_ms, .least_rows = least_rows };
    // a timed hold grid prices the pass itself: the overhead beyond it is the fixed one either way
    const overhead = r.fixedMs(cost_streams, sampled);
    const counts = try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, overhead, r.row_cost.capRows(cost_streams, sampled, @max(cost_streams, r.batch_rows)), head);
    defer r.gpa.free(counts);
    probeShared(r, whos, most, counts);
    if (r.plain_guard) {
        // no draft where none pays; a probe after a growing run of plain rounds
        for (whos, counts) |w, *c| {
            if (w.finished or w.forced) continue;
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

/// A plain measurement changes only the zero-draft option, never the existing drafted curve.
pub fn sharedPrice(r: *const Rule, streams: usize, sampled: bool) ?shared_price.Price {
    if (streams < 2 or !r.plain_guard or r.confidence_floor != null) return null;
    const head = if (nearest(r, streams, sampled)) |o| o.head else 0;
    const row = r.row_cost.extraMs(streams, sampled);
    var price = (if (r.k.fixed_draft_block)
        r.shared_price.blockPrice(streams, sampled, r.fixedMs(streams, sampled), head, row)
    else
        r.shared_price.price(streams, sampled, row)) orelse return null;
    if (r.k.learn_rows) {
        price.fixed = price.plain;
        price.row = row;
    }
    if (!r.k.fixed_draft_block) price.head = head;
    return price;
}

/// Held drafts have already run; every verify option retains the round's same head fixed cost.
pub fn verifyPrice(r: *const Rule, streams: usize, sampled: bool) ?shared_price.Price {
    var price = r.sharedPrice(streams, sampled) orelse return null;
    const head = price.head;
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
        try r.shared_price.observeHead(r.gpa, streams, rows, residual, head_ms, sampled, plain, calibration, r.k.fixed_draft_block);
    }
}

/// A shared round's ms beyond its forward at this many streams (the nearest measured count, first seen on ties).
pub fn overheadMs(r: *const Rule, streams: usize, sampled: bool) f64 {
    const o = nearest(r, streams, sampled) orelse return unmeasuredMs(r);
    return o.ms + o.head;
}

/// overheadMs without the measured or modeled drafter work attributed to those rounds.
pub fn fixedMs(r: *const Rule, streams: usize, sampled: bool) f64 {
    const o = nearest(r, streams, sampled) orelse return unmeasuredMs(r);
    return o.ms;
}

/// A timed hold grid supplies the head price; do not invent fixed overhead before observation.
pub fn unmeasuredMs(r: *const Rule) f64 {
    return if (r.k.hold_costs.len > 0) 0.0 else 8.0;
}

pub fn nearest(r: *const Rule, streams: usize, sampled: bool) ?Overhead {
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
    return headCost(r, depths, r.batched_heads);
}

pub fn pricedHeadMs(r: *const Rule, depths: []const i64) f64 {
    return headCost(r, depths, r.batched_heads and alloc.pricedBatched());
}

pub fn headCost(r: *const Rule, depths: []const i64, batched: bool) f64 {
    if (r.k.hold_costs.len > 0) return hold_price.passMs(r.k.hold_costs, r.k.hold_rows, r.k.hold_block_rows, depths);
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
    try addOverhead(r, &r.overhead, streams, fixed, extra - fixed);
    if (!sampled) try addOverhead(r, &r.greedy_overhead, streams, fixed, extra - fixed);
}

pub fn addOverhead(r: *Rule, list: *std.ArrayList(Overhead), streams: u32, fixed: f64, head: f64) !void {
    for (list.items) |*o| {
        if (o.streams == streams) {
            o.ms = o.ms + 0.2 * (fixed - o.ms);
            o.head = o.head + 0.2 * (head - o.head);
            return;
        }
    }
    try list.append(r.gpa, .{ .streams = streams, .ms = fixed, .head = head });
}

fn distance(a: u32, b: usize) usize {
    const x: usize = a;
    return if (x > b) x - b else b - x;
}
