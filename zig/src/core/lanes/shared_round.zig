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
const price_log = @import("price_log.zig");
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// A shared round's windows in one forward, each verified, committed and drafted alone; `ask` drafts their heads.
pub fn run(e: *Engine, entries: []*Stream, comptime ask: anytype) !void {
    return if (e.cfg.plain_guard) runPriced(e, entries, true, ask) else runPriced(e, entries, false, ask);
}

fn runPriced(e: *Engine, entries: []*Stream, comptime independent: bool, comptime ask: anytype) !void {
    var sampled_before = false;
    if (independent) for (entries) |s| {
        sampled_before = sampled_before or s.sampling != null;
    };
    const measured = independent and e.rule.sharedPrice(entries.len, sampled_before) != null;
    var hold_round = false;
    if (e.cfg.k.hold_costs.len > 0) for (entries) |s| {
        hold_round = hold_round or s.head_context != false;
    };
    if (hold_round) {
        @import("hold_price.zig").startRound(e);
        try e.late.flush(e, ask);
    } else {
        if (!measured) e.clock.start();
        try e.late.flush(e, ask);
        // An unmeasured plain price retains the original clock boundary.
        if (measured) e.clock.start();
    }
    const a = e.arena.allocator();
    const plans = try a.alloc(Plan, entries.len);
    for (entries, plans) |s, *p| p.* = try win.plan(e, s, null);
    const sampled = win.anySampled(plans);
    const calibration = !hold_round and independent and try @import("shared_price.zig").calibrate(e, plans);
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
    for (windows) |w| w.stream.verify_host_ms = 0;
    defer for (windows) |w| {
        w.stream.verify_host_ms = 0;
    };
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
        if (e.cfg.family_mtp and p.stream.headDrafts()) try heads.append(a, i);
    }
    const requests: []be.DraftRequest = if (heads.items.len > 0 and
        (e.cfg.draft_streams or e.backend.vtable.prepare_draft != null))
        try a.alloc(be.DraftRequest, heads.items.len)
    else
        &.{};
    for (requests, 0..) |*r, at| {
        const i = heads.items[at];
        const s = plans[i].stream;
        r.* = .{ .stream = s, .follow = outcomes[i].follow, .rows = outcomes[i].path, .start = plans[i].position, .position = s.cache_len + 1, .depth = 0 };
    }
    if (requests.len > 0) if (e.backend.vtable.prepare_draft) |prepare| {
        try prepare(e.backend.ptr, requests);
        for (requests) |*r| r.prepared = true;
    };
    const whos = try a.alloc(depth.Who, heads.items.len);
    for (whos, heads.items) |*w, i| w.* = win.who(plans[i].stream, e.cfg.k.hold_costs.len > 0);
    const budgets = try a.alloc(i64, heads.items.len);
    if (calibration)
        @memset(budgets, 0)
    else if ((heads.items.len > 0 or e.cfg.k.hold_costs.len == 0) and e.cfg.allocate_shared_drafts)
        try @import("shared_drafts.zig").budgets(e, plans, heads.items, budgets)
    else if (heads.items.len > 0 or e.cfg.k.hold_costs.len == 0) if (independent)
        try e.rule.budgetsFor(whos, win.probe(e), sampled, budgets, plans.len)
    else
        try e.rule.budgets(whos, win.probe(e), sampled, budgets);
    const ids = try a.alloc(ev.Value, heads.items.len);
    for (ids, heads.items) |*id, i| id.* = str(plans[i].stream.id);
    try trail.event(e, &.{ f("ev", str("budgets")), f("streams", .{ .list = ids }), f("depths", .{ .i64s = budgets }) });
    if (heads.items.len > 0 and e.cfg.draft_streams) {
        // Every head shares one hold, after all kept context has been staged and priced.
        for (requests, budgets) |*r, d| r.depth = @intCast(d);
        try ask(e, requests);
    } else if (requests.len > 0) {
        for (requests, budgets, 0..) |*r, d, i| {
            var who = win.who(r.stream, e.cfg.k.hold_costs.len > 0);
            r.depth = @intCast(try e.rule.headDepth(&who, d, win.probe(e)));
            try ask(e, requests[i..][0..1]);
        }
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
        // Actual timing includes every head step; the modeled path also removes the steps past a hold's grid.
        const extra = if (e.backend.draft_timing == null)
            @import("hold_price.zig").extraStepMs(budgets, e.cfg.draft_step_ms, e.cfg.draft_step_ms != 0 and e.cfg.draft_streams and depth.pricedBatched())
        else
            0;
        const ms = e.clock.elapsedMs(.overhead) - extra;
        try trail.event(e, &.{ f("ev", str("overhead")), f("streams", int(plans.len)), f("rows", int(total)), f("ms", ev.bits(ms)) });
        // the log covers both grant paths; only the plain guard's price observes them
        const head_ms = if (hold_round) @import("hold_price.zig").observedHeadMs(e.backend, &e.rule, budgets) else e.rule.pricedHeadMs(budgets);
        if (try price_log.enabled(e)) try price_log.round(e, plans, windows, .{ .trial = calibration, .sampled = sampled, .ms = ms, .head_ms = head_ms, .heads = heads.items, .depths = budgets });
        if (hold_round) {
            try e.rule.observeOverhead(@intCast(plans.len), total, ms - takeVerifyHostMs(windows), sampled, head_ms);
        } else {
            if (independent) {
                var plain = total == plans.len;
                for (budgets) |d| plain = plain and d == 0;
                try e.rule.observeSharedPrice(@intCast(plans.len), total, ms, sampled, head_ms, plain, calibration);
            }
            try e.rule.observeOverhead(@intCast(plans.len), total, ms - takeVerifyHostMs(windows), sampled, e.rule.headMs(budgets));
        }
    }
    if (e.backend.round_observer != null) e.backend.observed(paths);
}

/// Drain completed spans once; unquoted routes retain their old residual-overhead price.
fn takeVerifyHostMs(windows: []const be.Window) f64 {
    var ms: f64 = 0;
    for (windows) |w| {
        if (w.stream.verify_row_ms != null and w.stream.grammar == null) ms += w.stream.verify_host_ms;
        w.stream.verify_host_ms = 0;
    }
    return ms;
}
