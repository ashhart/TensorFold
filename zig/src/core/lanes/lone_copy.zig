//! A lone stream's copy windows priced as its head's drafts are: the depth rule on the stream's own timed rounds.
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
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
    const rate = win.copiedRate(e, s, copy.recalled);
    const keep: usize = @intCast(try e.rule.copyRows(win.who(s), &.{rate}, @intCast(n), take));
    if (take) try trail.event(e, &.{ f("ev", trail.str("copy-price")), f("stream", trail.str(s.id)), f("rows", trail.int(n)), f("keep", trail.int(keep)), f("rate", ev.bits(rate)) });
    return keep;
}

/// A priced stream's copy round by its rows, and a copies-only stream's plain round (a head's are timed with its own).
pub fn time(e: *Engine, s: *Stream, kind: win.Kind, window_rows: usize) !void {
    if (!enabled(e, s)) return;
    const plain = kind == .none and !e.cfg.family_mtp;
    if (kind != .copy and !plain) return;
    const ms = e.clock.elapsedMs(.cost);
    const first = s.rounds == 1; // a stream's first round carries the prefill-to-decode switch
    try trail.event(e, &.{ f("ev", trail.str(if (plain) "cost" else "copy-cost")), f("stream", trail.str(s.id)), f("drafts", trail.int(window_rows - 1)), f("init", .{ .bool = first }), f("ms", ev.bits(ms)) });
    if (plain) return e.rule.observePlain(ms, first, win.who(s));
    return e.rule.observeCopyCost(@intCast(window_rows - 1), ms, first, win.who(s));
}
