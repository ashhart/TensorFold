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

/// Whether `time` records this round: its copy rounds, and the plain and head rounds the head's timing keeps out.
pub fn records(e: *const Engine, s: *const Stream, kind: win.Kind) bool {
    if (!enabled(e, s)) return false;
    // engine.zig's familyRound times a head's and plain rounds into the stream's own times only under plain_guard
    return kind == .copy or ((kind == .none or kind == .head) and !(e.cfg.family_mtp and e.cfg.plain_guard));
}

/// A priced stream's round by its rows (`records`); `verified`: its time before the head drafted (a copy's verify).
pub fn time(e: *Engine, s: *Stream, kind: win.Kind, window_rows: usize, verified: ?f64) !void {
    if (!records(e, s, kind)) return;
    const own = kind != .copy;
    try seed(e, s);
    const ms = verified orelse e.clock.elapsedMs(.cost);
    const first = s.rounds == 1; // a stream's first round carries the prefill-to-decode switch
    try trail.event(e, &.{ f("ev", trail.str(if (own) "plain-cost" else "copy-cost")), f("stream", trail.str(s.id)), f("drafts", trail.int(window_rows - 1)), f("init", .{ .bool = first }), f("ms", ev.bits(ms)) });
    const drafts: i64 = @intCast(window_rows - 1);
    if (own) return copy_price.observeOwn(&e.rule, drafts, ms, first, win.who(s));
    return copy_price.observeCopy(&e.rule, drafts, ms, first, win.who(s));
}

/// Copy rounds a stream times before its prices replace the kept ones: a reply of a few tokens times one at most.
const kept_rounds = 2;

/// A stream's first priced round starts from the copy prices kept for its sampling mode and its context's band.
fn seed(e: *Engine, s: *Stream) !void {
    if (s.prices_seeded) return;
    s.prices_seeded = true;
    _ = try e.rule.rates(win.who(s));
    const band = lone_costs.band(s.context.items.len);
    s.depth.?.copy_band = band;
    try lone_costs.seedCopies(e.gpa, slot(e, s, band).*, s);
}

/// A finished stream's copy prices for the next one starting in its band; plain times too when only copies read them.
pub fn keep(e: *Engine, s: *const Stream) void {
    if (!e.cfg.copy_guard or e.cfg.lone_cost_tile > 0) return;
    const st = s.depth orelse return;
    const band = st.copy_band orelse return;
    if (st.copy_rounds < kept_rounds) return;
    lone_costs.keepCopies(e.gpa, slot(e, s, band), s, !e.cfg.plain_guard) catch {};
}

/// The kept copy prices of a stream's sampling mode at a band: those of the band it started in, read and left.
fn slot(e: *Engine, s: *const Stream, band: usize) *?lone_costs.Kept {
    return &e.lone_kept[@intFromBool(s.sampling != null)][band];
}
