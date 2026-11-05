//! Reserved holds and their recovery probes share one complete padded-pass price.
const std = @import("std");
const price = @import("hold_price.zig");

pub const Offer = struct {
    index: usize,
    gain: f64,
    rows: f64,
    row_ms: f64,
    depth: i64,
    hold_context_rows: u32 = 0,
    context_ms: f64 = 0, // solo ordering proxy; funding prices the actual complete group
    lowered: bool,
    shortened: bool,
    pays: bool,
    kept: bool = false, // incumbent membership survives cost-aware reordering

    fn less(_: void, x: Offer, y: Offer) bool {
        const gx = x.gain * y.rows;
        const gy = y.gain * x.rows;
        return gx > gy or (gx == gy and x.index < y.index);
    }

    fn lessAt(rate: f64, x: Offer, y: Offer) bool {
        const gx = (x.gain - rate * x.rows * x.row_ms - rate * x.context_ms) * y.rows;
        const gy = (y.gain - rate * y.rows * y.row_ms - rate * y.context_ms) * x.rows;
        return gx > gy or (gx == gy and x.index < y.index);
    }

    fn lessProbe(_: void, x: Offer, y: Offer) bool {
        const cx = x.context_ms + x.rows * x.row_ms;
        const cy = y.context_ms + y.rows * y.row_ms;
        return cx < cy or (cx == cy and x.index < y.index);
    }
};

/// Peek before reserving: a zero reservation must not spend a due trial.
pub fn ready(state: anytype, constants: anytype) bool {
    return (state.plain orelse 0) + 1 >= (state.wait orelse constants.depth_probe_every);
}

fn tick(state: anytype, constants: anytype) bool {
    const wait = state.wait orelse constants.depth_probe_every;
    state.plain = @min(wait, (state.plain orelse 0) + 1);
    return state.plain.? >= wait;
}

fn spent(state: anytype, constants: anytype) void {
    const wait = state.wait orelse constants.depth_probe_every;
    state.plain = 0;
    state.wait = @min(2 * wait, constants.plain_wait_most);
}

/// The aggregate remains useful to generic callers without fresh prefix prices.
fn gainOf(rule: anytype, who: anytype) !f64 {
    if (who.granted == @as(?u32, 0)) return 0;
    if (who.gain) |gain| if (who.granted) |count| {
        if (who.state.*) |state| if (state.tried > 0)
            return @min(gain, price.observedGain(state.observed, state.tried, count));
        return gain;
    };
    return (try rule.rates(who))[0];
}

fn offerFor(rule: anytype, who: anytype, index: usize, wanted: i64, base: price.Base, fixed: f64, plain: f64) !Offer {
    const observed: []const f64 = if (who.state.*) |state| state.observed else &.{};
    const tried = if (who.state.*) |state| state.tried else 0;
    const context_ms = price.contextMs(rule.k.hold_context_costs, 1, who.hold_context_rows);
    const prefix = if (rule.granted_depth)
        price.prefix(rule.shared_costs, rule.k.hold_costs, context_ms, base, fixed, who.row_ms, who.prices, observed, tried, wanted)
    else
        null;
    const offer: Offer = if (prefix) |p| .{
        .index = index,
        .gain = p.gain,
        .rows = p.rows,
        .row_ms = who.row_ms,
        .depth = p.depth,
        .hold_context_rows = who.hold_context_rows,
        .context_ms = context_ms,
        .lowered = p.lowered,
        .shortened = p.shortened,
        .pays = p.pays,
    } else legacy: {
        const gain = try gainOf(rule, who);
        var held: Offer = .{
            .index = index,
            .gain = gain,
            .rows = @floatFromInt(@max(1, who.granted orelse 1)),
            .row_ms = who.row_ms,
            .depth = wanted,
            .hold_context_rows = who.hold_context_rows,
            .context_ms = context_ms,
            .lowered = (who.granted orelse 0) > 0 and who.gain != null and gain < who.gain.?,
            .shortened = false,
            .pays = false,
        };
        if (base.forwardMs(rule.shared_costs, @as(i64, base.rows) + @as(i64, @intFromFloat(held.rows)))) |verify| {
            held.pays = (base.tokens + held.gain) / (verify + fixed + base.host_ms + held.rows * who.row_ms + price.holdMs(rule.k.hold_costs, 1, held.depth) + context_ms) > plain;
        }
        break :legacy held;
    };
    return offer;
}

/// All rows already owed to non-head work remain in `base`; only reserved head prefixes are optional.
pub fn pay(rule: anytype, whos: anytype, sampled: bool, out: []i64, base: price.Base) !void {
    if (whos.len > rule.batch_rows) return error.BadHoldStreamCount;
    const forward = base.forwardMs(rule.shared_costs, base.rows) orelse {
        var blocks: usize = 0;
        var deepest: i64 = 0;
        var context_rows: u32 = 0;
        for (whos, out) |who, *wanted| {
            if (who.finished or who.forced) wanted.* = 0;
            if (wanted.* <= 0) continue;
            blocks += 1;
            deepest = @max(deepest, wanted.*);
            context_rows = @max(context_rows, who.hold_context_rows);
        }
        const funded = std.math.isFinite(passCost(rule, whos, out, blocks, deepest, context_rows));
        // No price can prove a trial futile. Advance the reservation clock, but only spend funded trials.
        for (whos, out) |who, *wanted| {
            if (who.finished or who.forced) {
                wanted.* = 0;
                continue;
            }
            if (who.state.*) |*state| {
                const historical_zero = who.granted == @as(?u32, 0);
                const clocked = (historical_zero and rule.plain_guard) or (!historical_zero and state.hold_capped);
                const due = if (clocked) tick(state, rule.k) else false;
                if (!funded) wanted.* = 0;
                if (wanted.* > 0 and clocked and (historical_zero or due)) spent(state, rule.k);
            }
        }
        if (!funded) @memset(out, 0);
        return;
    };
    const fixed = rule.fixedMs(base.streams, sampled);
    const plain = base.tokens / (forward + fixed + base.host_ms);
    var normal: usize = 0;
    var probes: usize = 0;
    for (whos, out, 0..) |who, wanted, index| {
        if (who.finished or who.forced) {
            out[index] = 0;
            continue;
        }
        const offer = if (wanted > 0) try offerFor(rule, who, index, wanted, base, fixed, plain) else null;
        const historical_zero = who.granted == @as(?u32, 0);
        var due = false;
        if (who.state.*) |*state| {
            if (offer) |o| if (!o.lowered) {
                state.hold_capped = false;
            };
            if ((historical_zero and rule.plain_guard) or (!historical_zero and state.hold_capped))
                due = tick(state, rule.k);
        }
        var item = offer orelse continue; // fresh zero: no hold, and a due trial remains unspent
        if (rule.hold_order.len == 0) rule.hold_order = try rule.gpa.alloc(Offer, rule.batch_rows);
        if (historical_zero or due) {
            if (who.prices.len > 0) {
                // Probe the first missing profitable depth, not the whole unprofitable grant plus slack.
                const retained: i64 = if (item.pays) item.depth else 0;
                item.depth = @min(wanted, retained + 1);
                item.rows = @floatFromInt(item.depth);
                var raw: f64 = 0;
                for (who.prices[0..@min(@as(usize, @intCast(item.depth)), who.prices.len)]) |chance| raw += chance;
                item.gain = @min(raw, price.observedGain(observedOf(who), triedOf(who), @intCast(item.depth)));
            }
            out[index] = item.depth;
            rule.hold_order[rule.hold_order.len - 1 - probes] = item;
            probes += 1;
        } else {
            out[index] = item.depth;
            rule.hold_order[normal] = item;
            normal += 1;
        }
    }
    var rows: f64 = @floatFromInt(base.rows);
    var tokens = base.tokens;
    var host_ms = base.host_ms;
    var deepest: i64 = 0;
    var context_rows: u32 = 0;
    for (rule.hold_order[0..normal]) |item| out[item.index] = 0;
    if (probes > 0) {
        const offers = rule.hold_order[rule.hold_order.len - probes ..];
        const funded = tryProbes(rule, whos, out, offers, base, fixed, plain);
        probes = funded;
        for (offers, 0..) |probe, index| {
            if (index >= funded) {
                out[probe.index] = 0;
                continue;
            }
            const who = whos[probe.index];
            if (who.state.*) |*state| {
                if (who.granted != @as(?u32, 0) or rule.plain_guard) spent(state, rule.k);
                state.hold_capped = probe.lowered;
            }
            rows += probe.rows;
            tokens += probe.gain;
            host_ms += probe.rows * probe.row_ms;
            deepest = @max(deepest, probe.depth);
            context_rows = @max(context_rows, probe.hold_context_rows);
        }
    }
    if (normal == 0) return;
    const order = rule.hold_order[0..normal];
    std.mem.sort(Offer, order, {}, Offer.less);
    const starting_forward = base.forwardMs(rule.shared_costs, @intFromFloat(@ceil(rows))) orelse return;
    for (order) |item| out[item.index] = 0;
    const probe_ms = passCost(rule, whos, out, probes, deepest, context_rows);
    var best = tokens / (starting_forward + fixed + host_ms + probe_ms);
    const mixed = for (order) |item| {
        if (item.row_ms != order[0].row_ms or item.context_ms > 0) break true;
    } else false;
    while (true) {
        var current_rows = rows;
        var current_tokens = tokens;
        var current_host_ms = host_ms;
        var current_deepest = deepest;
        var current_context_rows = context_rows;
        var keep: ?usize = null;
        for (order, 1..) |item, count| {
            current_rows += item.rows;
            current_tokens += item.gain;
            current_host_ms += item.rows * item.row_ms;
            current_deepest = @max(current_deepest, item.depth);
            current_context_rows = @max(current_context_rows, item.hold_context_rows);
            const verify = base.forwardMs(rule.shared_costs, @intFromFloat(@ceil(current_rows))) orelse break;
            out[item.index] = item.depth;
            const now = current_tokens / (verify + fixed + current_host_ms + passCost(rule, whos, out, count + probes, current_deepest, current_context_rows));
            if (now > best) {
                best = now;
                keep = count;
            }
        }
        if (keep) |count| for (order, 0..) |*item, index| {
            item.kept = index < count;
        };
        // Equal host quotes with no context fee retain the old gain/row prefixes and deterministic ties.
        if (!mixed) break;
        const changed = for (order[1..], order[0 .. order.len - 1]) |next, previous| {
            if (Offer.lessAt(best, next, previous)) break true;
        } else false;
        if (!changed) break;
        // Strictly improving incumbents over finite deterministic orders terminate without an iteration cap.
        std.mem.sort(Offer, order, best, Offer.lessAt);
        for (order) |item| out[item.index] = 0;
        // Restart only normal offers: admitted probes and their spent cadence stay fixed.
    }
    for (order) |item| {
        if (whos[item.index].state.*) |*state|
            state.hold_capped = item.lowered and (!item.kept or item.shortened);
        out[item.index] = if (item.kept) item.depth else 0;
    }
}

/// The common single-pass shape stays O(1); split passes use only selected heads in physical stream order.
fn passCost(rule: anytype, whos: anytype, depths: []const i64, blocks: usize, deepest: i64, context_rows: u32) f64 {
    if (blocks == 0) return 0;
    const rows = rule.k.hold_rows;
    if (rows == 0 or @as(u64, @intCast(blocks)) * @as(u64, @intCast(@max(deepest, rule.k.hold_block_rows))) <= rows)
        return price.holdMs(rule.k.hold_costs, blocks, deepest) +
            price.contextMs(rule.k.hold_context_costs, blocks, context_rows);
    return price.rowsMs(rule.k.hold_costs, rule.k.hold_context_costs, rows, rule.k.hold_block_rows, View{ .whos = whos, .depths = depths });
}

/// A dear debt cannot veto affordable trials; reuse the offer storage and spend only the funded prefix.
fn tryProbes(rule: anytype, whos: anytype, depths: []i64, offers: []Offer, base: price.Base, fixed: f64, plain: f64) usize {
    if (!price.canExplore(rule, whos, base, fixed)) return 0;
    var deepest: i64 = 0;
    var context_rows: u32 = 0;
    for (offers) |item| {
        deepest = @max(deepest, item.depth);
        context_rows = @max(context_rows, item.hold_context_rows);
    }
    if (contextAllowsProbe(rule, whos, depths, base, fixed, plain, offers.len, deepest, context_rows))
        return offers.len;
    std.mem.sort(Offer, offers, {}, Offer.lessProbe);
    for (offers) |item| depths[item.index] = 0;
    deepest = 0;
    context_rows = 0;
    var kept: usize = 0;
    for (offers, 1..) |item, count| {
        depths[item.index] = item.depth;
        deepest = @max(deepest, item.depth);
        context_rows = @max(context_rows, item.hold_context_rows);
        if (contextAllowsProbe(rule, whos, depths, base, fixed, plain, count, deepest, context_rows)) kept = count;
    }
    return kept;
}

const View = struct {
    whos: []const @import("depth.zig").Who,
    depths: []const i64,
    padded: ?u32 = null,
    most: u32 = 0,

    pub fn len(v: @This()) usize {
        return v.depths.len;
    }
    pub fn at(v: @This(), i: usize) i64 {
        const d = v.depths[i];
        if (d <= 0) return 0;
        return if (v.padded) |padded| @min(padded, v.most, @max(0, v.whos[i].draft_room - 1)) else d;
    }
    pub fn contextRows(v: @This(), i: usize) u32 {
        return v.whos[i].hold_context_rows;
    }
};

/// Context belongs to the selected trial, while clean shape optimism still permits recovery through deeper holds.
fn contextAllowsProbe(rule: anytype, whos: anytype, depths: []const i64, base: price.Base, fixed: f64, plain: f64, blocks: usize, deepest: i64, context_rows: u32) bool {
    if (context_rows == 0 or rule.k.hold_context_costs.len == 0) return true;
    const one_pass = rule.k.hold_rows == 0 or
        @as(u64, @intCast(blocks)) * @as(u64, @intCast(@max(deepest, rule.k.hold_block_rows))) <= rule.k.hold_rows;
    const context_ms = if (one_pass) price.contextMs(rule.k.hold_context_costs, blocks, context_rows) else price.rowsMs(&.{}, rule.k.hold_context_costs, rule.k.hold_rows, rule.k.hold_block_rows, View{ .whos = whos, .depths = depths });
    if (!std.math.isFinite(context_ms)) return false;
    var view = View{ .whos = whos, .depths = depths, .most = rule.most_drafts };
    var padded: u32 = 1;
    while (padded <= rule.most_drafts) : (padded += 1) {
        view.padded = padded;
        var extra: u32 = 0;
        var host_ms = base.host_ms;
        for (0..depths.len) |i| {
            const count: u32 = @intCast(view.at(i));
            extra += count;
            host_ms += @as(f64, @floatFromInt(count)) * whos[i].row_ms;
        }
        if (base.rows + extra > rule.batch_rows) break;
        const verify = base.forwardMs(rule.shared_costs, base.rows + extra) orelse return true;
        const clean_ms = price.rowsMs(rule.k.hold_costs, &.{}, rule.k.hold_rows, rule.k.hold_block_rows, view);
        if ((base.tokens + @as(f64, @floatFromInt(extra))) / (verify + fixed + host_ms + clean_ms + context_ms) > plain)
            return true;
    }
    return false;
}

fn observedOf(who: anytype) []const f64 {
    return if (who.state.*) |state| state.observed else &.{};
}

fn triedOf(who: anytype) usize {
    return if (who.state.*) |state| state.tried else 0;
}
