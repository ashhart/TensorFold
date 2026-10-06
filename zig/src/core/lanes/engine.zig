//! The lane round loop (Python FamilyRounds and SharedRounds): every decision and cost update in Python's order.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Config = @import("config.zig").Config;
const depth = @import("depth.zig");
const accept = @import("accept.zig");
const be = @import("backend.zig");
const sm = @import("stream.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const fill = @import("fill.zig");
const trail = @import("trail.zig");
const Stream = sm.Stream;
const Feed = be.Feed;
const Plan = win.Plan;
const f = ev.f;
const str = trail.str;
const int = trail.int;

const Outcome = struct { got: []const u32, path: []const u32, cut: ?usize, follow: []const u32 };
const Result = struct { rows: u32, keep: u32, got: []const u32 };
const Unread = struct { event: usize, handle: u64 };

pub const Engine = struct {
    gpa: Allocator,
    cfg: *const Config,
    backend: be.Backend,
    clock: be.Clock,
    log: ?*ev.Log = null,
    rule: depth.Rule,
    live: std.ArrayList(*Stream) = .empty, // admitted streams in admission order (Python `_live`)
    drafted: u64 = 0,
    accepted: u64 = 0,
    shared_rounds: u64 = 0,
    alone: bool = true, // this step has one live stream: its copies may take their ramp
    steps: u64 = 0,
    arena: std.heap.ArenaAllocator, // a round's temporaries
    unread: std.ArrayList(Unread) = .empty, // queued draws the log reads at the step's end

    pub fn init(gpa: Allocator, cfg: *const Config, backend: be.Backend, clock: be.Clock) Engine {
        return .{
            .gpa = gpa,
            .cfg = cfg,
            .backend = backend,
            .clock = clock,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .rule = .{
                .gpa = gpa,
                .k = cfg.k,
                .prior = cfg.depth_prior,
                .most_drafts = cfg.most_drafts,
                .family_costs = &cfg.family_costs,
                .shared_costs = &cfg.shared_costs,
                .mtp_step_ms = cfg.mtp_step_ms,
                .plain_guard = cfg.plain_guard,
                .node_probabilities = cfg.node_probabilities,
                .batch_rows = cfg.batch_rows,
            },
        };
    }

    pub fn deinit(e: *Engine) void {
        e.live.deinit(e.gpa);
        e.unread.deinit(e.gpa);
        e.rule.deinit();
        e.arena.deinit();
    }

    /// Streams holding or about to hold a row (paused ones included).
    pub fn activeCount(e: *const Engine) usize {
        var n: usize = 0;
        for (e.live.items) |s| n += @intFromBool(!s.finished);
        return n;
    }

    /// Prefill a stream, draw its first token and ask for its first drafts; it takes part from the next round.
    pub fn addStream(e: *Engine, s: *Stream) !void {
        _ = e.arena.reset(.retain_capacity);
        try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
        e.backend.prefill(s) catch |err| {
            if (err == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
            return err;
        };
        if (s.isCancelled()) { // cancelled in its last chunk: no first token
            e.backend.release(s);
            return error.Cancelled;
        }
        s.context.shrinkRetainingCapacity(s.prompt_len);
        s.pending = null;
        s.cache_len = s.prompt_len;
        const position: u64 = s.prompt_len;
        const drawn = try e.backend.first(s, position);
        var feed: Feed = .{ .handle = drawn };
        if (try e.forcedNext(s)) |t| feed = .{ .value = t };
        var asked: ?u32 = null;
        if (e.cfg.family_mtp and s.drafts) {
            // the head reads the prompt's last row and the first token, and drafts the one after it
            const d: u32 = @intCast(try e.rule.depth(win.who(s)));
            asked = d;
            try e.backend.draft(&.{.{ .stream = s, .follow = &.{}, .first = feed, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = d }});
            s.dropHeld(e.gpa);
            s.next = .{ .count = d };
        } else if (e.cfg.pipelined) {
            try e.queueNext(s, feed);
        }
        const value = try e.readFeed(feed);
        if (asked) |d| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
        if (e.log != null) {
            const first = if (feed == .handle) value else try e.backend.read(drawn);
            try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(first)), f("token", int(value)) });
        }
        _ = try s.commit(e.gpa, &.{value});
        s.pending = value;
        try trail.resolve(e);
        if (s.finished) {
            try trail.finish(e, s);
            e.release(s);
            return;
        }
        try e.live.append(e.gpa, s);
    }

    /// Take over a stream another driver decoded so far: its cache settled, `pending` and `cache_len` set, no drafts held.
    pub fn adopt(e: *Engine, s: *Stream) !void {
        try e.live.append(e.gpa, s);
    }

    /// Release a cancelled stream between rounds: no pending draws or drafts.
    pub fn discard(e: *Engine, s: *Stream) void {
        s.finished = true;
        s.reason = .cancelled;
        for (e.live.items, 0..) |x, i| {
            if (x == s) {
                _ = e.live.orderedRemove(i);
                break;
            }
        }
        e.release(s);
    }

    /// One round for every live stream (Python `_family_step`).
    pub fn step(e: *Engine) !void {
        _ = e.arena.reset(.retain_capacity);
        const a = e.arena.allocator();
        try trail.event(e, &.{ f("ev", str("step")), f("index", int(e.steps)) });
        e.steps += 1;
        var live: std.ArrayList(*Stream) = .empty;
        for (e.live.items) |s| {
            if (!s.finished and !s.paused) try live.append(a, s);
        }
        if (live.items.len > 1 and e.cfg.family_streams) {
            // a shared round is synchronous: a stream that ran a step ahead lands its queued token first
            for (live.items) |s| {
                if (s.inflight != null) try e.landInflight(s);
            }
            var kept: usize = 0;
            for (live.items) |s| {
                if (s.finished) continue;
                live.items[kept] = s;
                kept += 1;
            }
            live.shrinkRetainingCapacity(kept);
        }
        e.alone = live.items.len == 1;
        if (live.items.len > 1 and e.cfg.family_streams) {
            const chosen = try e.takeTurns(live.items);
            e.shared_rounds += 1;
            for (chosen) |s| s.served = e.shared_rounds;
            try e.roundStreams(chosen);
        } else {
            for (live.items) |s| {
                const r = if ((e.cfg.family_mtp and s.drafts) or !e.cfg.pipelined) try e.familyRound(s, null) else try e.pipelinedRound(s);
                s.min_rows = if (s.min_rows == 0) r.rows else @min(s.min_rows, r.rows);
            }
        }
        try trail.resolve(e);
        var kept: usize = 0;
        for (e.live.items) |s| {
            if (s.finished) {
                try trail.finish(e, s);
                e.release(s);
                continue;
            }
            e.live.items[kept] = s;
            kept += 1;
        }
        e.live.shrinkRetainingCapacity(kept);
        try trail.state(e);
    }

    /// Least recently served streams within the stream and row limits, never cutting pending or forced rows.
    fn takeTurns(e: *Engine, live: []*Stream) ![]*Stream {
        const a = e.arena.allocator();
        const order = try a.alloc(usize, live.len);
        for (order, 0..) |*o, i| o.* = i;
        const Ctx = struct {
            live: []*Stream,
            fn served(c: @This(), i: usize) i64 {
                return if (c.live[i].served) |v| @intCast(v) else -1;
            }
            fn less(c: @This(), x: usize, y: usize) bool {
                const sx = c.served(x);
                const sy = c.served(y);
                return sx < sy or (sx == sy and x < y);
            }
        };
        std.mem.sort(usize, order, Ctx{ .live = live }, Ctx.less);
        var chosen: std.ArrayList(usize) = .empty;
        var rows: u64 = 0;
        for (order) |i| {
            const s = live[i];
            const need: u64 = if (s.drafts) @min(@min(e.cfg.base_width, e.cfg.batch_rows), 1 + s.force.items.len) else 1;
            if (chosen.items.len == e.cfg.batch_streams or (chosen.items.len > 0 and rows + need > e.cfg.batch_rows)) break;
            try chosen.append(a, i);
            rows += need;
        }
        std.mem.sort(usize, chosen.items, {}, std.sort.asc(usize));
        const out = try a.alloc(*Stream, chosen.items.len);
        const ids = try a.alloc(ev.Value, chosen.items.len);
        for (out, ids, chosen.items) |*o, *id, i| {
            o.* = live[i];
            id.* = str(live[i].id);
        }
        try trail.event(e, &.{ f("ev", str("turns")), f("streams", .{ .list = ids }) });
        return out;
    }

    /// One stream's round: verify pending and drafted tokens together, keep drafts to the first mismatch.
    fn familyRound(e: *Engine, s: *Stream, copied: ?[]const u32) !Result {
        e.clock.start();
        const priced = s.priced;
        s.priced = null;
        const a = e.arena.allocator();
        var plans = [_]Plan{try win.plan(e, s, copied)};
        if (plans[0].kind == .head and e.cfg.node_probabilities) try win.allocate(e, &plans);
        const plan = plans[0];
        const early = e.cfg.family_mtp and s.drafts and plan.kind != .forced and e.cfg.speculate_early and plan.parents == null;
        const windows = [_]be.Window{try win.build(e, plan, early)};
        const rows = windows[0].rows();
        var out = [_]be.Verified{.{ .sampled = try a.alloc(u32, rows), .drafts = try a.alloc(u32, rows - 1) }};
        try e.backend.verify(&windows, &out);
        const tokens = try win.tokens(e, windows[0], out[0]);
        const parents = try win.rowParents(e, windows[0]);
        const o = try e.conclude(s, plan, out[0].sampled, tokens, parents, windows[0].positions);
        if (o.path.len < rows or plan.parents != null) try e.backend.keep(&windows, &.{o.path});
        const held_levels = s.held_levels;
        if (e.cfg.family_mtp and s.drafts) {
            if (early and o.cut == null) {
                var w = win.who(s);
                const d: u32 = @intCast(try e.rule.headDepth(&w, null, win.probe(e)));
                try e.askDrafts(&.{.{ .stream = s, .follow = o.follow, .rows = o.path, .start = plan.position, .position = s.cache_len + 1, .depth = d, .early = true }});
            } else {
                if (early) if (e.backend.vtable.unspeculate) |undo| try undo(e.backend.ptr, s);
                try e.draftLate(s, plan.position, o.follow, o.path, null);
            }
            if (plan.kind == .head or (plan.kind == .none and e.cfg.plain_guard)) {
                const ms = e.clock.elapsedMs(.cost);
                const first = s.rounds == 1; // a stream's first round carries the prefill-to-decode switch
                try trail.event(e, &.{ f("ev", str("cost")), f("stream", str(s.id)), f("drafts", int(rows - 1)), f("init", .{ .bool = first }), f("ms", ev.bits(ms)) });
                try e.rule.observeCost(@intCast(rows - 1), ms, first, win.who(s));
            }
        }
        if (s.graft_room != null and s.rounds > 1) {
            // the stream's round time beyond its window and head levels, for the lane planner
            if (e.cfg.lane_costs.get(@intCast(rows))) |w| {
                const over = e.clock.elapsedMs(.cost) - w - e.cfg.mtp_step_ms * @as(f64, @floatFromInt(held_levels));
                const kept = if (plan.lanes != null and plan.parents != null) &s.tree_over else &s.round_over;
                kept.* = if (kept.*) |x| x + 0.25 * (over - x) else over;
            }
            // the pick that shaped this round, priced from its own measured rounds next time
            if (priced) |p| s.measured[p.choice].add(e.clock.elapsedMs(.cost) - p.ms);
        }
        if (plan.kind == .head) {
            const ms = e.clock.elapsedMs(.overhead);
            try trail.event(e, &.{ f("ev", str("overhead")), f("streams", int(1)), f("rows", int(rows)), f("ms", ev.bits(ms)) });
            try e.rule.observeOverhead(1, rows, ms);
        }
        return .{ .rows = @intCast(rows), .keep = @intCast(o.path.len), .got = o.got };
    }

    /// The stream windows of a shared round in one forward, then each verified, committed and drafted on its own.
    fn roundStreams(e: *Engine, entries: []*Stream) !void {
        e.clock.start();
        const a = e.arena.allocator();
        const plans = try a.alloc(Plan, entries.len);
        for (entries, plans) |s, *p| p.* = try win.plan(e, s, null);
        try win.allocate(e, plans);
        const windows = try a.alloc(be.Window, plans.len);
        const out = try a.alloc(be.Verified, plans.len);
        var total: u64 = 0;
        for (plans, windows, out) |p, *w, *o| {
            w.* = try win.build(e, p, false);
            o.* = .{ .sampled = try a.alloc(u32, w.rows()), .drafts = try a.alloc(u32, w.rows() - 1) };
            total += w.rows();
        }
        try e.backend.verify(windows, out);
        const outcomes = try a.alloc(Outcome, plans.len);
        const paths = try a.alloc([]const u32, plans.len);
        for (plans, windows, out, outcomes, paths) |p, w, o, *oc, *path| {
            const tokens = try win.tokens(e, w, o);
            oc.* = try e.conclude(p.stream, p, o.sampled, tokens, try win.rowParents(e, w), w.positions);
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
        try e.rule.budgets(whos, win.probe(e), budgets);
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
        try e.backend.keep(windows, paths);
        const ms = e.clock.elapsedMs(.overhead);
        try trail.event(e, &.{ f("ev", str("overhead")), f("streams", int(plans.len)), f("rows", int(total)), f("ms", ev.bits(ms)) });
        try e.rule.observeOverhead(@intCast(plans.len), total, ms);
    }

    /// Commit the target-sampled path (after the thinking budget's cut); the kept rows and their following tokens.
    fn conclude(e: *Engine, s: *Stream, p: Plan, sampled: []const u32, tokens: []const u32, parents: []const i32, positions: []const u64) !Outcome {
        const a = e.arena.allocator();
        const rows = tokens.len;
        var path: []u32 = undefined;
        var committed: std.ArrayList(u32) = .empty;
        if (p.kind == .forced) {
            path = try a.alloc(u32, rows);
            for (path, 0..) |*r, i| r.* = @intCast(i);
            try committed.appendSlice(a, p.forced);
            try committed.append(a, s.popForce() orelse sampled[rows - 1]);
        } else {
            path = try accept.acceptPath(a, tokens, parents, sampled);
            for (path[1..]) |r| try committed.append(a, tokens[r]);
            try committed.append(a, sampled[path[path.len - 1]]);
            if (rows > 1) {
                const kept = path.len - 1;
                e.drafted += rows - 1;
                e.accepted += kept;
                s.drafted += rows - 1;
                s.accepted += kept;
                if (p.kind == .copy) {
                    if (s.proposer) |pr| pr.observe(@intCast(rows - 1), @intCast(kept));
                    if (e.alone) {
                        const width = s.copy_width orelse e.cfg.first_copy;
                        s.copy_width = if (kept == rows - 1) @min(e.cfg.max_copy, 2 * width + 1) else e.cfg.first_copy;
                    }
                } else if (p.kind == .head) {
                    if (p.lanes) |l| if (s.odds != null) observeLanes(s, l, path);
                    const ds = try accept.depths(a, parents);
                    const chain = if (p.branch_from > 0) ds[0..p.branch_from] else ds;
                    var chain_kept: usize = 0;
                    for (path[1..]) |r| chain_kept += @intFromBool(p.branch_from == 0 or r < p.branch_from);
                    try e.rule.observeDepth(win.who(s), std.mem.max(u32, chain), @intCast(chain_kept));
                }
                var kept_branch: u64 = 0;
                if (p.branch_from > 0) {
                    s.branch_rows += rows - p.branch_from;
                    for (path[1..]) |r| kept_branch += @intFromBool(r >= p.branch_from);
                    s.branch_accepted += kept_branch;
                }
                if (p.tried) {
                    const tried: f64 = @floatFromInt(if (p.branch_from > 0) rows - p.branch_from else 1);
                    s.graft_hit = 0.75 * s.graft_hit + 0.25 * @as(f64, @floatFromInt(kept_branch)) / tried;
                }
            }
        }
        const budget_cut = s.thinkCut(committed.items);
        const loop_cut = sm.loopCut(s, committed.items);
        const loop_wins = if (loop_cut) |c| budget_cut == null or c < budget_cut.? else false;
        var cut: ?usize = null;
        if (loop_wins) {
            cut = loop_cut;
            path = path[0 .. loop_cut.? + 1];
            committed.shrinkRetainingCapacity(loop_cut.? + 1);
        } else if (budget_cut) |c| {
            cut = c;
            path = path[0 .. c + 1];
            committed.shrinkRetainingCapacity(c);
            try committed.append(a, try s.startClose(e.gpa));
        }
        s.rounds += 1;
        const landed = try s.commit(e.gpa, committed.items);
        const got = committed.items[0..landed];
        if (s.finished and got.len < path.len) path = path[0 .. got.len + 1];
        const keep = path.len;
        s.cache_len += keep;
        s.pending = committed.items[keep - 1];
        const chain = accept.isChain(parents);
        try trail.event(e, &.{
            f("ev", str("round")),                                    f("stream", str(s.id)),
            f("kind", str(@tagName(p.kind))),                         f("window", .{ .u32s = tokens }),
            f("parents", if (chain) .null else .{ .i32s = parents }), f("positions", .{ .u64s = positions }),
            f("sampled", .{ .u32s = sampled }),                       f("forced", .{ .u32s = p.forced }),
            f("path", .{ .u32s = path }),                             f("got", .{ .u32s = got }),
            f("cut", if (cut) |c| int(c) else .null),
        });
        return .{ .got = got, .path = path, .cut = cut, .follow = committed.items[0..keep] };
    }

    /// Read the kept rows with their following tokens and ask the head for the next round's drafts.
    fn draftLate(e: *Engine, s: *Stream, start: u64, follow: []const u32, path: []const u32, budget: ?i64) !void {
        var w = win.who(s);
        const d: u32 = @intCast(try e.rule.headDepth(&w, budget, win.probe(e)));
        try e.askDrafts(&.{.{ .stream = s, .follow = follow, .rows = path, .start = start, .position = s.cache_len + 1, .depth = d }});
    }

    fn askDrafts(e: *Engine, requests: []const be.DraftRequest) !void {
        for (requests) |r| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(r.stream.id)), f("depth", int(r.depth)), f("position", int(r.position)), f("follow", .{ .u32s = r.follow }), f("rows", if (r.rows) |x| .{ .u32s = x } else .null) });
        // a lone stream's lanes: the head tree or chain + grafts with the most expected tokens a round time
        var lone: ?shape.Shape = null;
        errdefer if (lone) |l| l.deinit(e.gpa);
        var asked = requests;
        var one: [1]be.DraftRequest = undefined;
        if (e.cfg.head_trees and e.cfg.head_lanes > 0 and e.alone and requests.len == 1 and requests[0].depth > 0) {
            const s = requests[0].stream;
            if (s.odds == null) s.odds = shape.Odds.init(shape.start_odds, shape.start_weight);
            const cap: usize = @intCast(@max(1, @min(@min(shape.max_depth, e.cfg.head_depth), s.draftRoom() - 1)));
            const costs = plan_lanes.Costs{ .window = &e.cfg.lane_costs, .head_ms = e.cfg.mtp_step_ms, .over_ms = s.round_over orelse 0, .tree_over_ms = s.tree_over, .measured = if (e.cfg.own_prices) &s.measured else null };
            // every eighth round one size wider than the best, so the odds of deeper lanes stay measured
            const p = try plan_lanes.pick(e.gpa, s.odds.?, costs, e.cfg.head_lanes, cap, try e.graftProspect(s), s.rounds % 8 == 4);
            s.picks[p.choice] += 1;
            s.graft_room = p.grafts;
            if (plan_lanes.tablePrice(costs, p)) |ms| s.priced = .{ .choice = p.choice, .ms = ms };
            one[0] = requests[0];
            if (p.tree) |t| {
                // a chain-shaped tree drafts and verifies as the head's chain
                lone = t;
                if (t.isChain()) one[0].depth = @intCast(t.parents.len) else one[0].lanes = &lone.?;
                s.held_levels = levelsOf(t);
            } else {
                one[0].depth = p.depth;
                one[0].ranks = true;
                s.held_levels = p.depth;
            }
            asked = &one;
        } else if (requests.len == 1) requests[0].stream.held_levels = requests[0].depth;
        try e.backend.draft(asked);
        for (asked) |r| {
            r.stream.dropHeld(e.gpa);
            r.stream.next = .{ .count = r.depth };
            if (lone) |l| {
                if (r.lanes != null) r.stream.next = .{ .count = @intCast(l.parents.len), .parents = try e.gpa.dupe(i32, l.parents) };
                r.stream.lanes = l;
                lone = null;
            } else if (e.backend.vtable.tree) |tree| {
                if (try tree(e.backend.ptr, r.stream, e.gpa)) |held| r.stream.next = held;
            }
        }
    }

    /// The rows a stream's suffix matches would graft this round and the share of grafted rows it keeps (null: none).
    fn graftProspect(e: *Engine, s: *Stream) !?plan_lanes.Grafts {
        if (e.cfg.fill_lanes == 0) return null;
        const probe_round = s.rounds % 8 == 0;
        if (s.graft_hit < e.cfg.fill_floor and !probe_round) return null;
        const a = e.arena.allocator();
        const room = e.cfg.fill_lanes -| 1;
        const found = try fill.suffixCandidates(a, s.context.items, 16, e.cfg.fill_match, room, 8);
        if (found.len == 0) return null;
        const b = try fill.graft(a, .{}, found, room);
        // the stream's recent share of grafted rows kept; every eighth round at least the floor, so grafts get retried
        const hit = if (probe_round) @max(s.graft_hit, e.cfg.fill_floor) else s.graft_hit;
        return .{ .rows = @intCast(b.tokens.len), .hit = hit };
    }

    fn levelsOf(t: shape.Shape) u32 {
        var most: u32 = 0;
        for (t.depths) |d| most = @max(most, d);
        return most + 1;
    }

    /// Each held tree lane whose parent row the round reached: the target took it or not (its depth and rank).
    fn observeLanes(s: *Stream, l: *const shape.Shape, path: []const u32) void {
        var on: [256]bool = @splat(false);
        for (path) |r| if (r < on.len) {
            on[r] = true;
        };
        for (l.parents, l.depths, l.ranks, 0..) |par, d, r, i| {
            const parent_row: usize = if (par < 0) 0 else 1 + @as(usize, @intCast(par));
            if (!on[parent_row]) continue;
            const took = on[1 + i];
            s.odds.?.observe(d, r, took);
            if (d < shape.max_depth and r < shape.ranks) {
                s.landed[d][r][0] += 1;
                s.landed[d][r][1] += @intFromBool(took);
            }
        }
    }

    /// One token, the next forward queued first; a copied continuation ahead switches to verify windows.
    fn pipelinedRound(e: *Engine, s: *Stream) !Result {
        const before = s.mode;
        const mode = s.mode orelse .pipe;
        var result: Result = .{ .rows = 1, .keep = 1, .got = &.{} };
        if (mode == .verify or mode == .exit) {
            const proposal = if (mode == .verify and e.cfg.family_width >= 2) try win.copyProposal(e, s) else &.{};
            if (proposal.len > 0) {
                result = try e.familyRound(s, try e.arena.allocator().dupe(u32, proposal));
                if (result.keep == 1 or s.force.items.len > 0) s.mode = .exit;
            } else {
                // nothing to copy: back to steps queued ahead (this one lands next round)
                s.mode = .pipe;
                try e.queueNext(s, .{ .value = s.pending.? });
            }
            try trail.pipe(e, s, before, result.got);
            return result;
        }
        var current: Feed = .{ .handle = s.inflight.? };
        s.inflight = null;
        if (try e.forcedNext(s)) |t| current = .{ .value = t };
        var token: u32 = undefined;
        if (mode == .drain) {
            token = try e.readFeed(current); // the last queued step: no new one
            s.mode = .verify;
        } else {
            try e.queueNext(s, current); // the GPU starts the next step first
            token = try e.readFeed(current);
        }
        s.rounds += 1;
        const landed = try s.commit(e.gpa, &.{token});
        s.pending = token;
        if (e.cfg.family_width >= 2 and (s.mode orelse .pipe) == .pipe and !s.finished and s.drafts and (try win.copyProposal(e, s)).len > 0)
            s.mode = .drain; // a copy window is ahead: land the queued step
        result.got = try e.arena.allocator().dupe(u32, s.emitted()[s.emitted().len - landed ..]);
        try trail.pipe(e, s, before, result.got);
        return result;
    }

    /// Land the queued token without queuing another (a shared round is synchronous).
    fn landInflight(e: *Engine, s: *Stream) !void {
        var current: Feed = .{ .handle = s.inflight.? };
        s.inflight = null;
        if (try e.forcedNext(s)) |t| current = .{ .value = t };
        const token = try e.readFeed(current);
        s.rounds += 1;
        const landed = try s.commit(e.gpa, &.{token});
        s.pending = token;
        s.mode = .verify;
        try trail.event(e, &.{ f("ev", str("land")), f("stream", str(s.id)), f("got", .{ .u32s = s.emitted()[s.emitted().len - landed ..] }) });
    }

    /// Feed a token and queue the draw of the next at the new cache length.
    fn queueNext(e: *Engine, s: *Stream, feed: Feed) !void {
        s.cache_len += 1;
        const h = try e.backend.queue(s, feed, s.cache_len);
        s.inflight = h;
        if (e.log) |log| {
            const i = try log.add(&.{ f("ev", str("queue")), f("stream", str(s.id)), f("position", int(s.cache_len)), f("token", .null) });
            try e.unread.append(e.gpa, .{ .event = i, .handle = h });
        }
    }

    /// The thinking budget's or a forced fix's token at the next position instead of the draw.
    fn forcedNext(e: *Engine, s: *Stream) !?u32 {
        if (s.popForce()) |t| return t;
        if (s.cutsNext()) return try s.startClose(e.gpa);
        return null;
    }

    fn readFeed(e: *Engine, feed: Feed) !u32 {
        return switch (feed) {
            .handle => |h| try e.backend.read(h),
            .value => |v| v,
        };
    }

    fn release(e: *Engine, s: *Stream) void {
        s.dropRoundState(e.gpa);
        e.backend.release(s);
    }
};
