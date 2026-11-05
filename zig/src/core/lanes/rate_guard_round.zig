//! Round-loop integration for per-stream plain-versus-drafted throughput.
const win = @import("windows.zig");

pub fn build(e: anytype, p: win.Plan) !@import("backend.zig").Window {
    var w = try win.build(e, p, false);
    if (e.cfg.rate_guard and w.rows() == 1) w.hidden = e.rule.rowsWanted(win.who(p.stream));
    return w;
}

pub fn observe(e: anytype, s: anytype, rows: usize, ms: f64, first: bool, tokens: usize) !void {
    try e.rule.observeCost(@intCast(rows - 1), ms, first, win.who(s));
    if (e.cfg.rate_guard and !first) try e.rule.observeRate(win.who(s), tokens, ms);
}

pub fn keep(e: anytype, windows: anytype, paths: anytype, heads: []const usize, plans: anytype, outcomes: anytype) !void {
    try e.backend.keep(windows, paths);
    // A first round carries the prefill-to-decode switch and is not a throughput sample.
    if (e.cfg.rate_guard and e.cfg.timed) for (heads) |i| if (plans[i].stream.rounds > 1)
        try e.rule.observeRate(win.who(plans[i].stream), outcomes[i].got.len, e.clock.elapsedMs(.cost));
}
