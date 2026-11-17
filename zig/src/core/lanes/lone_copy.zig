//! A lone stream's copy windows priced as its head's drafts are: the depth rule on the stream's own timed rounds.
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
const copy_price = @import("copy_price.zig");
const lone_costs = @import("lone_costs.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const f = ev.f;

/// A lone stream whose copies compete with its plain rounds (copy_guard), timed, its head priced by the rule's chain.
pub fn enabled(e: *const Engine, s: *const Stream) bool {
    const c = e.cfg;
    return e.alone and s.drafts and c.timed and c.copy_guard and e.trim == null and !c.root_copies and
        !c.context_copies and !c.node_probabilities and c.lone_cost_tile == 0 and !c.family_costs.empty() and e.windowed(s);
}

/// The copy's rows to verify, a prefix (none: plain or the head's drafts take the round); `take`: the round's plan.
pub fn rows(e: *Engine, s: *Stream, copy: win.Copy, take: bool) !usize {
    const n = copy.tokens.len;
    if (n == 0 or !enabled(e, s)) return n;
    try seed(e, s);
    const rate = win.copiedRate(e, s, copy.recalled);
    const kept: usize = @intCast(try e.rule.copyRows(win.who(s), &.{rate}, @intCast(n), take));
    if (take) try trail.event(e, &.{ f("ev", trail.str("copy-price")), f("stream", trail.str(s.id)), f("rows", trail.int(n)), f("keep", trail.int(kept)), f("rate", ev.bits(rate)) });
    return kept;
}

/// A priced stream's copy round by its rows, and its plain and head rounds the head's timing keeps out of its times.
pub fn time(e: *Engine, s: *Stream, kind: win.Kind, window_rows: usize) !void {
    if (!enabled(e, s)) return;
    // engine.zig's familyRound times a head's and plain rounds into the stream's own times only under plain_guard
    const own = (kind == .none or kind == .head) and !(e.cfg.family_mtp and e.cfg.plain_guard);
    if (kind != .copy and !own) return;
    try seed(e, s);
    const ms = e.clock.elapsedMs(.cost);
    const first = s.rounds == 1; // a stream's first round carries the prefill-to-decode switch
    try trail.event(e, &.{ f("ev", trail.str(if (own) "plain-cost" else "copy-cost")), f("stream", trail.str(s.id)), f("drafts", trail.int(window_rows - 1)), f("init", .{ .bool = first }), f("ms", ev.bits(ms)) });
    const drafts: i64 = @intCast(window_rows - 1);
    if (own) return copy_price.observeOwn(&e.rule, drafts, ms, first, win.who(s));
    return copy_price.observeCopy(&e.rule, drafts, ms, first, win.who(s));
}

/// A stream's first priced round starts from the copy prices the engine kept for its sampling mode and context band.
fn seed(e: *Engine, s: *Stream) !void {
    if (s.prices_seeded) return;
    s.prices_seeded = true;
    _ = try e.rule.rates(win.who(s));
    try lone_costs.seedCopies(e.gpa, lone_costs.slot(&e.lone_kept, s).*, s);
}

/// A finished stream's copy prices for the next lone stream, its plain times too where only copy pricing reads them.
pub fn keep(e: *Engine, s: *const Stream) void {
    if (!e.cfg.copy_guard or e.cfg.lone_cost_tile > 0) return;
    lone_costs.keepCopies(e.gpa, lone_costs.slot(&e.lone_kept, s), s, !e.cfg.plain_guard) catch {};
}
