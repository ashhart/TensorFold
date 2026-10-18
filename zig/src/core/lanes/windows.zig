//! A round's windows (Python `_plan_window`, `_copy_proposal`, `_allocate`): forced, copied or drafted rows.
const std = @import("std");
const accept = @import("accept.zig");
const cut = @import("window_cut.zig");
const lone_costs = @import("lone_costs.zig");
const alloc = @import("allocate.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const trim = @import("trim.zig");
const fill = @import("fill.zig");
const shape = @import("shape.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Value = ev.Value;
const f = ev.f;

pub const Kind = enum { forced, copy, head, fill, grammar, none };

/// One stream's window as planned, then as allocated (Python's plan lists).
pub const Plan = struct {
    stream: *Stream,
    position: u64, // the stream's cache length when the round started
    kind: Kind,
    held: u32 = 0, // head drafts the backend holds
    tokens: []const u32 = &.{}, // drafts known on the host (forced, copied, tree)
    forced: []const u32 = &.{},
    parents: ?[]const i32 = null, // a tree's drafts' parents
    branch_from: u32 = 0, // the first row of suffix-match branches (0: none; lanes/fill.zig)
    cost_probe_rows: ?u32 = null, // an unseen width selected for calibration, confirmed by a matching completed round
    context_copy: bool = false, // branch_from names a copied continuation appended to the head
    tried: bool = false, // the planner read the head's chain back to graft branches
    lanes: ?*const shape.Shape = null, // the held drafts' tree shape (each lane's depth and rank)
    masks: []const u32 = &.{}, // structured output: the rows' allowed-token bits (constrain)
    grammar_from: u32 = 0, // the first row the grammar drafted itself (0: none)
    recalled: bool = false, // the copy came from Recall's record
    graft_recalled: []const bool = &.{}, // each row from branch_from: a Recall candidate grafted it

    pub fn count(p: Plan) usize {
        return p.held + p.tokens.len;
    }

    /// Whether window row `r` (a graft) is Recall's.
    pub fn recallRow(p: Plan, r: usize) bool {
        if (p.branch_from == 0 or r < p.branch_from) return false;
        const i = r - p.branch_from;
        return i < p.graft_recalled.len and p.graft_recalled[i];
    }
};

/// A copied continuation and its source: the stream's own context, or Recall's record.
pub const Copy = struct { tokens: []const u32 = &.{}, recalled: bool = false };

/// Forced tokens first, then a copied continuation, then the head's drafts (Python `_plan_window`).
pub fn plan(e: *Engine, s: *Stream, copied: ?Copy) !Plan {
    const a = e.arena.allocator();
    var p: Plan = .{ .stream = s, .position = s.cache_len, .kind = .none };
    const queued = s.next;
    s.next = null;
    defer if (queued) |q| {
        if (q.tokens) |t| e.gpa.free(t);
        if (q.parents) |parents| e.gpa.free(parents);
    };
    if (s.force.items.len > 0) {
        const width: usize = if (s.drafts) @min(e.cfg.base_width, e.cfg.batch_rows) else 1;
        const take = @min(width - 1, s.force.items.len);
        const forced = try a.dupe(u32, s.force.items[0..take]);
        s.force.replaceRangeAssumeCapacity(0, take, &.{});
        p.kind = .forced;
        p.forced = forced;
        p.tokens = forced;
    } else {
        var copy: Copy = .{};
        if (copied) |c| copy = c else if (s.drafts) copy = try copyProposal(e, s);
        const extend = e.alone and e.cfg.context_copies and !copy.recalled and queued != null and queued.?.count > 0;
        if (copy.tokens.len > 0 and !extend) {
            p.kind = .copy;
            p.recalled = copy.recalled;
            p.tokens = try a.dupe(u32, copy.tokens);
        } else if (s.drafts and queued != null) {
            const q = queued.?;
            if (q.tokens) |t| {
                p.tokens = try a.dupe(u32, t);
                if (q.parents) |parents| {
                    if (!accept.isChain(parents)) p.parents = try a.dupe(i32, parents);
                }
            } else {
                p.held = q.count;
                if (s.lanes) |*l| {
                    // a held tree is a lone stream's (shared rounds verify chains); a chain's lanes are observed too
                    if (q.parents) |parents| {
                        if (e.alone) p.parents = try a.dupe(i32, parents) else p.held = 0;
                    }
                    if (p.held > 0) p.lanes = l;
                }
            }
            if (p.count() > 0) p.kind = .head;
            if (extend and copy.tokens.len > 0) try @import("context_copy.zig").extend(e, &p, copy.tokens);
        }
    }
    if (!p.context_copy and e.cfg.fill_lanes > 0 and s.drafts and p.kind != .forced and (p.parents == null or p.lanes != null)) try fillLanes(e, s, &p);
    try trail.event(e, &.{ f("ev", trail.str("plan")), f("stream", trail.str(s.id)), f("kind", trail.str(@tagName(p.kind))), f("n", trail.int(p.count())), f("tree", .{ .bool = p.parents != null }) });
    return p;
}

/// Suffix-match branches grafted onto the planned drafts, up to cfg.fill_lanes rows.
fn fillLanes(e: *Engine, s: *Stream, p: *Plan) !void {
    const a = e.arena.allocator();
    const first = p.count();
    const room = @min(@as(usize, e.cfg.fill_lanes) -| (1 + first), s.graft_room orelse std.math.maxInt(u32));
    if (room == 0) return;
    const found = try candidates(e, s, room, s.rounds % 8 == 0);
    if (found.len == 0) return;
    // the trunk's lanes: the drafts in order for a chain, a tree's rank-0 path from the pending row
    var trunk_lanes: [shape.max_depth * shape.ranks * 4]usize = undefined;
    const levels = if (p.lanes) |l| shape.chainLanes(l.*, &trunk_lanes) else first;
    if (p.lanes == null) {
        for (trunk_lanes[0..first], 0..) |*x, k| x.* = k;
    }
    const trunk = try headTrunk(e, s, p.*, levels) orelse return;
    p.tried = p.held > 0;
    const b = try fill.graft(a, trunk, found, room);
    if (b.tokens.len == 0) return;
    const parents = try a.alloc(i32, first + b.tokens.len);
    if (p.parents) |held| {
        @memcpy(parents[0..first], held);
    } else {
        for (parents[0..first], 0..) |*q, k| q.* = @as(i32, @intCast(k)) - 1;
    }
    // a graft's parent: the pending row, a trunk level (that level's lane), or one of its own rows (after the drafts)
    for (parents[first..], b.parents) |*q, g| q.* = if (g < 0) -1 else if (g < levels) @intCast(trunk_lanes[@intCast(g)]) else @intCast(first + @as(usize, @intCast(g)) - levels);
    p.tokens = if (p.held > 0) b.tokens else try std.mem.concat(a, u32, &.{ p.tokens, b.tokens });
    p.parents = parents;
    p.branch_from = @intCast(1 + first);
    p.graft_recalled = b.recalled;
    if (p.kind == .none) p.kind = .fill;
}

/// Graft candidates: the context's while its rows land, then Recall's while its do (both on a probe round); own first.
pub fn candidates(e: *Engine, s: *Stream, room: usize, probe_round: bool) ![]const fill.Proposal {
    const a = e.arena.allocator();
    const live = s.graft_hit >= e.cfg.fill_floor or probe_round;
    const own: []const fill.Proposal = if (!live) &.{} else try fill.suffixCandidates(
        a,
        s.context.items,
        longest_graft,
        e.cfg.fill_match,
        room,
        8,
    );
    const r = s.recall orelse return own;
    if (s.recall_hit < e.cfg.fill_floor and !probe_round) return own;
    return fill.merge(a, own, try r.candidates(a, s.context.items, room, @max(2, 8 -| own.len), longest_graft));
}

/// The longest suffix a graft candidate's weight counts, for the context's matches and Recall's alike.
const longest_graft = 16;

/// The drafts' trunk of `levels` lanes: host tokens or held head ranks; null means unread.
fn headTrunk(e: *Engine, s: *Stream, p: Plan, levels: usize) !?fill.Trunk {
    if (p.held == 0) return .{ .tokens = p.tokens };
    const a = e.arena.allocator();
    const read = e.backend.vtable.alternatives orelse return null;
    const alts = try a.alloc(be.Alternative, levels);
    if (try read(e.backend.ptr, s, alts) < levels) return null;
    const chain = try a.alloc(u32, levels);
    const others = try a.alloc([3]u32, levels);
    for (alts, chain, others) |x, *t, *o| {
        t.* = x.tokens[0];
        o.* = x.tokens[1..4].*;
    }
    return .{ .tokens = chain, .others = others };
}

/// A copy from the context backed by `enter_match` ids (Python `_copy_proposal`), else Recall's when it should pay.
pub fn copyProposal(e: *Engine, s: *Stream) !Copy {
    if (e.cfg.max_copy == 0 or s.force.items.len > 0) return .{};
    const width: u32 = if (e.alone) (s.copy_width orelse e.cfg.first_copy) else e.cfg.first_copy;
    const most = @min(@as(i64, width), s.draftRoom() - 1);
    if (s.proposer) |p| if (!e.alone or e.cfg.lone_copies) {
        // a priced proposer's drafts are as it gives them, its none too: no width, length or match gate, no Recall
        if (p.vtable.priced) return .{ .tokens = p.propose(s.context.items, @min(@as(i64, e.cfg.max_copy), s.draftRoom() - 1)) catch &.{} };
        const copied = p.proposeAfter(s.context.items, &.{}, most) catch &.{};
        const least: usize = if (e.alone and e.cfg.context_copies) 1 else 2;
        if (copied.len >= least and p.lastMatch() >= e.cfg.enter_match) return .{ .tokens = copied };
    };
    const r = s.recall orelse return .{};
    if (most < 2) return .{};
    const got = r.propose(s.context.items, @intCast(most), s.rounds);
    return if (got.len >= 2) .{ .tokens = got, .recalled = true } else .{};
}

pub fn copyAhead(ptr: *anyopaque, w: *const depth.Who) bool {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *Stream = @ptrCast(@alignCast(w.stream.?));
    const copied = copyProposal(e, s) catch return false;
    return copied.tokens.len > 0;
}

pub fn probe(e: *Engine) depth.Probe {
    return .{ .ptr = e, .copy = copyAhead };
}

pub fn who(s: *Stream) depth.Who {
    return .{ .state = &s.depth, .draft_room = s.draftRoom(), .finished = s.finished, .forced = s.force.items.len > 0, .granted = s.granted, .stream = s, .sampled = s.sampling != null };
}

/// Whether any of a round's streams draws keyed samples (its rows carry their host work).
pub fn anySampled(plans: []const Plan) bool {
    for (plans) |p| if (p.stream.sampling != null) return true;
    return false;
}

/// Trim draft prefixes by landing probability to the shared width with the most expected tokens a ms.
pub fn allocate(e: *Engine, plans: []Plan) !void {
    if (e.trim) |t| return trimmed(e, plans, t);
    const a = e.arena.allocator();
    const fixed = try a.alloc(u32, plans.len);
    const probs = try a.alloc([]const f64, plans.len);
    for (plans, fixed, probs) |p, *fx, *pr| {
        const n = p.count();
        if (p.kind == .forced or n == 0) {
            fx.* = 1 + @as(u32, @intCast(if (p.kind == .forced) n else 0));
            pr.* = &.{};
        } else if (p.kind == .copy) {
            fx.* = 1;
            pr.* = try alloc.chainProbabilities(a, &.{e.cfg.copy_rate}, n);
        } else {
            fx.* = 1;
            const chances = try nodeChances(e, p.stream, n);
            const controls = @import("mtp_policy.zig").Controls{ .cap = e.cfg.most_drafts, .floor = e.cfg.draft_confidence };
            pr.* = if (p.kind == .head and e.cfg.draft_confidence != null) chances[0..controls.prefix(chances)] else chances;
        }
    }
    const untimed: @import("table.zig").Table = .{};
    const explicit = e.cfg.draft_confidence != null;
    const live = plans.len == 1 and lone_costs.enabled(e, plans[0]);
    const own = if (live) try lone_costs.prices(e, plans[0]) else untimed;
    const costs = if (explicit) &untimed else if (live) &own else &e.cfg.shared_costs;
    const overhead = if (explicit or live) 0 else e.rule.overheadMs(plans.len, anySampled(plans));
    const counts = try alloc.allocate(a, fixed, probs, costs, overhead, @max(plans.len, e.cfg.batch_rows));
    const trial = if (live) lone_costs.probe(e, plans[0]) else null;
    if (trial) |count| {
        counts[0] = count;
        plans[0].cost_probe_rows = count + 1;
        plans[0].stream.cost_probe_trial = lone_costs.trial(plans[0].stream, probs[0], count + 1);
    }
    if (live) {
        var expected: f64 = 1;
        for (probs[0][0..counts[0]]) |chance| expected += chance;
        const ms = own.get(1 + @as(i64, counts[0]));
        try trail.event(e, &.{ f("ev", trail.str("lone-price")), f("stream", trail.str(plans[0].stream.id)), f("rows", trail.int(counts[0] + 1)), f("expected", ev.bits(expected)), f("ms", if (ms) |value| ev.bits(value) else .null), f("probe", .{ .bool = trial != null }) });
    }
    for (plans, counts) |*p, c| {
        if (p.kind == .head) p.stream.granted = c;
        if (p.kind == .forced or p.count() == 0 or c == p.count()) continue;
        try cut.prefix(a, p, c);
    }
    const rows = try a.alloc(Value, plans.len);
    for (rows, plans) |*r, p| {
        const pair = try a.alloc(Value, 2);
        pair[0] = trail.str(@tagName(p.kind));
        pair[1] = trail.int(p.count());
        r.* = .{ .list = pair };
    }
    try trail.event(e, &.{ f("ev", trail.str("alloc")), f("plans", .{ .list = rows }) });
}

/// A family's row policy decides each head window's drafts (lanes/trim.zig); other windows run whole.
fn trimmed(e: *Engine, plans: []Plan, t: trim.Trim) !void {
    const a = e.arena.allocator();
    const asks = try a.alloc(trim.Window, plans.len);
    for (asks, plans) |*w, p| w.* = .{ .stream = p.stream, .drafted = p.kind == .head, .rows = @intCast(1 + p.count()), .parents = p.parents };
    const out = try a.alloc(trim.Choice, plans.len);
    for (out, plans) |*o, p| o.* = .{ .count = @intCast(p.count()) };
    try t.choose(asks, e.alone and plans.len == 1, a, out);
    for (plans, out) |*p, c| {
        if (p.kind != .head) continue;
        if (c.nodes) |nodes| {
            if (p.parents == null or p.held > 0) return error.TrimNodesNeedTree;
            const sub = try trim.select(a, p.tokens, p.parents.?, nodes);
            try cut.selected(a, p, nodes);
            p.tokens = sub.tokens;
            p.parents = if (accept.isChain(try accept.rowParents(a, 1 + sub.tokens.len, sub.parents))) null else sub.parents;
        } else try cut.prefix(a, p, c.count);
        if (p.count() == 0) p.kind = .none;
    }
}

/// The head's chance for each held draft, else chain chances from the stream's per-depth acceptance.
pub fn nodeChances(e: *Engine, s: *Stream, n: usize) ![]const f64 {
    const a = e.arena.allocator();
    if (e.cfg.node_probabilities) {
        if (e.backend.vtable.probabilities) |chances| {
            const out = try a.alloc(f64, n);
            if (try chances(e.backend.ptr, s, out)) return out;
        }
    }
    if (e.cfg.draft_confidence != null) return error.NoDraftProbabilities;
    return alloc.chainProbabilities(a, try e.rule.rates(who(s)), n);
}

/// Structured output: the rows an accepted path can use, each row's bits, and the grammar's drafts after a lone chain.
pub fn constrain(e: *Engine, p: *Plan) !void {
    const g = p.stream.grammar orelse return;
    const a = e.arena.allocator();
    if (p.held > 0) {
        // the held drafts (a chain or a tree, in window order) come back as host tokens, grafts after them
        const read = e.backend.vtable.held orelse return error.HeldDraftsUnreadable;
        const held = try a.alloc(u32, p.held);
        try read(e.backend.ptr, p.stream, held);
        p.tokens = try std.mem.concat(a, u32, &.{ held, p.tokens });
        p.held = 0;
    }
    const rows = try a.alloc(u32, 1 + p.tokens.len);
    rows[0] = p.stream.pending.?;
    @memcpy(rows[1..], p.tokens);
    const grows = p.stream.drafts and p.kind != .forced and e.alone;
    const room = if (grows) @max(rows.len, e.cfg.family_width) else rows.len;
    const w = try g.window(a, rows, try accept.rowParents(a, rows.len, p.parents), p.kind == .forced, room);
    p.masks = w.masks;
    if (w.rows.len == rows.len and w.extra.len == 0) return;
    if (w.rows.len < rows.len) p.lanes = null; // the tree shape no longer matches the rows
    const kept = try a.alloc(u32, w.rows.len - 1 + w.extra.len);
    for (kept[0 .. w.rows.len - 1], w.rows[1..]) |*t, r| t.* = rows[r];
    @memcpy(kept[w.rows.len - 1 ..], w.extra);
    p.tokens = kept;
    if (w.extra.len > 0) p.grammar_from = @intCast(w.rows.len);
    if (p.parents != null) {
        // the grammar drafts only past a chain's last kept row (Constraint.chain), so its drafts extend that chain
        const parents = try a.alloc(i32, kept.len);
        for (parents, 0..) |*q, i| q.* = if (i + 1 < w.parents.len) w.parents[i + 1] - 1 else @as(i32, @intCast(i)) - 1;
        p.parents = if (accept.isChain(parents)) null else parents;
    }
    if (p.branch_from > 0) {
        var below: u32 = 0;
        for (w.rows) |r| below += @intFromBool(r < p.branch_from);
        if (p.graft_recalled.len > 0) { // the kept rows are in window order: the grafts' sources follow them
            const flags = try a.alloc(bool, w.rows.len - below);
            for (flags, w.rows[below..]) |*fl, r| fl.* = p.recallRow(r);
            p.graft_recalled = flags;
        }
        p.branch_from = below;
    }
    if (p.kind != .forced and w.rows.len == 1) p.kind = if (w.extra.len > 0) .grammar else .none;
}

/// The window's rows, parents and keyed positions, ready for the backend.
pub fn build(e: *Engine, p: Plan, early: bool) !be.Window {
    const a = e.arena.allocator();
    const rows = 1 + p.count();
    const parents = try accept.rowParents(a, rows, p.parents);
    const ds = try accept.depths(a, parents);
    const positions = try a.alloc(u64, rows);
    for (positions, ds) |*x, d| x.* = p.position + 1 + d;
    return .{ .stream = p.stream, .pending = p.stream.pending.?, .held = p.held, .tokens = p.tokens, .parents = if (p.parents != null) parents else null, .positions = positions, .early = early, .masks = p.masks };
}

pub fn rowParents(e: *Engine, w: be.Window) ![]const i32 {
    if (w.parents) |p| return p;
    return accept.rowParents(e.arena.allocator(), w.rows(), null);
}

/// The rows' tokens: the pending one, then the drafts (held drafts as the backend read them back).
pub fn tokens(e: *Engine, w: be.Window, o: be.Verified) ![]u32 {
    const out = try e.arena.allocator().alloc(u32, w.rows());
    out[0] = w.pending;
    @memcpy(out[1 .. 1 + w.held], o.drafts[0..w.held]);
    @memcpy(out[1 + w.held ..], w.tokens);
    return out;
}
