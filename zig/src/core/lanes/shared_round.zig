//! Shared verification and draft scheduling, with bounded all-plain calibration rounds.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Plan = win.Plan;
const Outcome = @import("engine.zig").Outcome;
const LogRow = @import("logprob.zig").Row;
const win = @import("windows.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// The stream windows of a shared round in one forward, then each verified, committed and drafted on its own.
pub fn run(e: *Engine, entries: []*Stream) !void {
    try e.late.flush(e, Engine.askDrafts);
    // A deferred lone draft belongs to the preceding round, not this shared round.
    e.clock.start();
    const a = e.arena.allocator();
    const plans = try a.alloc(Plan, entries.len);
    for (entries, plans) |s, *p| p.* = try win.plan(e, s, null);
    const sampled = win.anySampled(plans);
    const calibration = try @import("shared_price.zig").calibrate(e, plans);
    if (calibration) {
        for (plans) |*p| try @import("window_cut.zig").prefix(a, p, 0);
    } else try win.allocate(e, plans);
    try @import("grammar_parallel.zig").constrain(e, plans);
    const windows = try a.alloc(be.Window, plans.len);
    const out = try a.alloc(be.Verified, plans.len);
    var total: u64 = 0;
    for (plans, windows, out) |p, *w, *o| {
        w.* = try win.build(e, p, false);
        o.* = .{ .sampled = try a.alloc(u32, w.rows()), .drafts = try a.alloc(u32, w.rows() - 1), .rows = if (p.stream.logprobs != null) try a.alloc(LogRow, w.rows()) else &.{} };
        total += w.rows();
    }
    try e.backend.verify(windows, out);
    const outcomes = try a.alloc(Outcome, plans.len);
    const paths = try a.alloc([]const u32, plans.len);
    for (plans, windows, out, outcomes, paths) |p, w, o, *oc, *path| {
        const tokens = try win.tokens(e, w, o);
        oc.* = try e.conclude(p.stream, p, o.sampled, o.rows, tokens, try win.rowParents(e, w), w.positions);
        path.* = oc.path;
        const rows: u32 = @intCast(w.rows());
        p.stream.min_rows = if (p.stream.min_rows == 0) rows else @min(p.stream.min_rows, rows);
    }
    var heads: std.ArrayList(usize) = .empty;
    for (plans, 0..) |p, i| {
        if (e.cfg.family_mtp and p.stream.drafts) try heads.append(a, i);
    }
    const whos = try a.alloc(depth.Who, heads.items.len);
    for (whos, heads.items) |*w, i| w.* = win.who(plans[i].stream);
    const budgets = try a.alloc(i64, heads.items.len);
    if (calibration)
        @memset(budgets, 0)
    else if (e.cfg.allocate_shared_drafts)
        try @import("shared_drafts.zig").budgets(e, plans, heads.items, budgets)
    else
        try e.rule.budgetsFor(whos, win.probe(e), sampled, budgets, plans.len);
    const ids = try a.alloc(ev.Value, heads.items.len);
    for (ids, heads.items) |*id, i| id.* = str(plans[i].stream.id);
    try trail.event(e, &.{ f("ev", str("budgets")), f("streams", .{ .list = ids }), f("depths", .{ .i64s = budgets }) });
    if (heads.items.len > 0 and e.cfg.draft_streams) {
        // every stream's head in one batch (the backend drafts a depth for all of them at once)
        const requests = try a.alloc(be.DraftRequest, heads.items.len);
        for (requests, heads.items, budgets) |*r, i, d| {
            const s = plans[i].stream;
            r.* = .{ .stream = s, .follow = outcomes[i].follow, .rows = outcomes[i].path, .start = plans[i].position, .position = s.cache_len + 1, .depth = @intCast(d) };
        }
        try e.askDrafts(requests);
    } else {
        for (heads.items, budgets) |i, d| try e.draftLate(plans[i].stream, plans[i].position, outcomes[i].follow, outcomes[i].path, d);
    }
    if (e.cfg.root_copies) for (heads.items, budgets) |i, *d| {
        if (plans[i].stream.next) |held| if (held.count == 0) {
            d.* = 0;
        };
    };
    try e.backend.keep(windows, paths);
    if (e.cfg.timed) {
        // the drafts budgets priced past each stream's first are no part of the round's overhead
        var steps: i64 = 0;
        for (budgets) |d| steps = if (e.cfg.draft_streams and depth.pricedBatched()) @max(steps, d - 1) else steps + @max(0, d - 1);
        const ms = e.clock.elapsedMs(.overhead) - e.cfg.draft_step_ms * @as(f64, @floatFromInt(steps));
        try trail.event(e, &.{ f("ev", str("overhead")), f("streams", int(plans.len)), f("rows", int(total)), f("ms", ev.bits(ms)) });
        var plain = total == plans.len;
        for (budgets) |d| plain = plain and d == 0;
        try e.rule.observeShared(@intCast(plans.len), total, ms, sampled, e.rule.headMs(budgets), plain, calibration);
    }
    if (e.backend.round_observer != null) e.backend.observed(paths);
}
