//! Reserved holds and their recovery probes share one complete padded-pass price.
const std = @import("std");
const price = @import("hold_price.zig");

pub const Offer = struct {
    index: usize,
    gain: f64,
    rows: f64,
    depth: i64,
    lowered: bool,
    shortened: bool,
    pays: bool,

    fn less(_: void, x: Offer, y: Offer) bool {
        const gx = x.gain * y.rows;
        const gy = y.gain * x.rows;
        return gx > gy or (gx == gy and x.index < y.index);
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
    const prefix = if (rule.granted_depth)
        price.prefix(rule.shared_costs, rule.k.hold_costs, base, fixed, who.prices, observed, tried, wanted)
    else
        null;
    const offer: Offer = if (prefix) |p| .{
        .index = index,
        .gain = p.gain,
        .rows = p.rows,
        .depth = p.depth,
        .lowered = p.lowered,
        .shortened = p.shortened,
        .pays = p.pays,
    } else legacy: {
        const gain = try gainOf(rule, who);
        var held: Offer = .{
            .index = index,
            .gain = gain,
            .rows = @floatFromInt(@max(1, who.granted orelse 1)),
            .depth = wanted,
            .lowered = (who.granted orelse 0) > 0 and who.gain != null and gain < who.gain.?,
            .shortened = false,
            .pays = false,
        };
        if (rule.shared_costs.get(@as(i64, base.rows) + @as(i64, @intFromFloat(held.rows)))) |verify| {
            held.pays = (base.tokens + held.gain) / (verify + fixed + price.holdMs(rule.k.hold_costs, 1, held.depth)) > plain;
        }
        break :legacy held;
    };
    return offer;
}

/// All rows already owed to non-head work remain in `base`; only reserved head prefixes are optional.
pub fn pay(rule: anytype, whos: anytype, sampled: bool, out: []i64, base: price.Base) !void {
    if (whos.len > rule.batch_rows) return error.BadHoldStreamCount;
    const forward = rule.shared_costs.get(base.rows) orelse {
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
                if (wanted.* > 0 and clocked and (historical_zero or due)) spent(state, rule.k);
            }
        }
        return;
    };
    const fixed = rule.fixedMs(base.streams, sampled);
    const plain = base.tokens / (forward + fixed);
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
    var deepest: i64 = 0;
    if (probes > 0) {
        const possible = price.canExplore(rule, whos, base, fixed);
        for (rule.hold_order[rule.hold_order.len - probes ..]) |probe| {
            if (!possible) {
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
            deepest = @max(deepest, probe.depth);
        }
        if (!possible) probes = 0;
    }
    if (normal == 0) return;
    const order = rule.hold_order[0..normal];
    std.mem.sort(Offer, order, {}, Offer.less);
    const starting_forward = rule.shared_costs.get(@intFromFloat(@ceil(rows))) orelse return;
    for (order) |item| out[item.index] = 0;
    const probe_ms = passCost(rule, out, probes, deepest);
    var best = tokens / (starting_forward + fixed + probe_ms);
    var keep: usize = 0;
    for (order, 1..) |item, count| {
        rows += item.rows;
        tokens += item.gain;
        deepest = @max(deepest, item.depth);
        const verify = rule.shared_costs.get(@intFromFloat(@ceil(rows))) orelse break;
        out[item.index] = item.depth;
        const now = tokens / (verify + fixed + passCost(rule, out, count + probes, deepest));
        if (now > best) {
            best = now;
            keep = count;
        }
    }
    for (order, 0..) |item, index| {
        if (whos[item.index].state.*) |*state|
            state.hold_capped = item.lowered and (index >= keep or item.shortened);
        if (index >= keep) out[item.index] = 0;
    }
}

/// The common single-pass shape stays O(1); physical splitting preserves the backend's original head order.
fn passCost(rule: anytype, depths: []const i64, blocks: usize, deepest: i64) f64 {
    if (blocks == 0) return 0;
    const rows = rule.k.hold_rows;
    if (rows == 0 or @as(u64, @intCast(blocks)) * @as(u64, @intCast(deepest)) <= rows)
        return price.holdMs(rule.k.hold_costs, blocks, deepest);
    return price.passMs(rule.k.hold_costs, rows, depths);
}

fn observedOf(who: anytype) []const f64 {
    return if (who.state.*) |state| state.observed else &.{};
}

fn triedOf(who: anytype) usize {
    return if (who.state.*) |state| state.tried else 0;
}
