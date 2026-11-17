//! A lone copy window's price parts on its stream's own timed rounds: the copy's cost and the plain round it must beat.
const depth = @import("depth.zig");
const Rule = depth.Rule;
const Who = depth.Who;
const Table = @import("table.zig").Table;

/// A copy as timed, else on the line from plain to its nearest timed copy, else each row as plain's (no head step).
pub fn cost(r: *Rule, rows: i64, who: Who) !?f64 {
    const timed = try copies(r, who);
    if (timed.get(rows)) |v| return v;
    const base = (try plain(r, who)) orelse return null;
    const n: f64 = @floatFromInt(rows);
    var near: ?i64 = null;
    for (timed.slots, 0..) |slot, at| {
        const d: i64 = @intCast(at);
        if (slot == null or d == 0) continue;
        if (near == null or @abs(d - rows) < @abs(near.? - rows)) near = d;
    }
    if (near) |d| return base + n * @max(0.0, (timed.get(d).? - base) / @as(f64, @floatFromInt(d)));
    const one = r.family_costs.get(1) orelse return null;
    const forward = r.family_costs.get(rows + 1) orelse return null;
    return forward + (n + 1) * @max(0.0, base - one);
}

/// The plain round a copy must beat: roundCost's under plain_guard, else its model on the stream's own rounds.
pub fn plain(r: *Rule, who: Who) !?f64 {
    if (r.plain_guard) return r.roundCost(0, who);
    return r.roundCostIn(try r.ownCosts(who), 0, true);
}

/// A lone copy round's time by its copied rows; a stream's first round stays out, as in observeCost.
pub fn observeCopy(r: *Rule, rows: i64, ms: f64, initializing: bool, who: Who) !void {
    if (rows <= 0 or initializing) return;
    try r.blend(try copies(r, who), rows, ms);
    who.state.*.?.copy_rounds += 1;
}

/// A round the head's timing keeps out of the stream's own times (no plain_guard): plain's price reads it.
pub fn observeOwn(r: *Rule, drafts: i64, ms: f64, initializing: bool, who: Who) !void {
    if (drafts < 0 or initializing) return;
    try r.blend(try r.ownCosts(who), drafts, ms);
}

fn copies(r: *Rule, who: Who) !*Table {
    _ = try r.rates(who);
    const st = &who.state.*.?;
    if (st.copy_ms == null) st.copy_ms = .{};
    return &st.copy_ms.?;
}
