//! A round's windows (Python `_plan_window`, `_copy_proposal`, `_allocate`): forced, copied or drafted rows.
const std = @import("std");
const accept = @import("accept.zig");
const alloc = @import("allocate.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const fill = @import("fill.zig");
const shape = @import("shape.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Value = ev.Value;
const f = ev.f;

pub const Kind = enum { forced, copy, head, fill, none };

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
    tried: bool = false, // the planner read the head's chain back to graft branches
    lanes: ?*const shape.Shape = null, // the held drafts' tree shape (each lane's depth and rank)
    masks: []const u32 = &.{}, // structured output: the rows' allowed-token bits (constrain)

    pub fn count(p: Plan) usize {
        return p.held + p.tokens.len;
    }
};

/// Forced tokens first, then a copied continuation, then the head's drafts (Python `_plan_window`).
pub fn plan(e: *Engine, s: *Stream, copied: ?[]const u32) !Plan {
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
        var copy: []const u32 = &.{};
        if (copied) |c| copy = c else if (s.drafts) copy = try copyProposal(e, s);
        if (copy.len > 0) {
            p.kind = .copy;
            p.tokens = try a.dupe(u32, copy);
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
        }
    }
    if (e.cfg.fill_lanes > 0 and s.drafts and p.kind != .forced and (p.parents == null or p.lanes != null)) try fillLanes(e, s, &p);
    try trail.event(e, &.{ f("ev", trail.str("plan")), f("stream", trail.str(s.id)), f("kind", trail.str(@tagName(p.kind))), f("n", trail.int(p.count())), f("tree", .{ .bool = p.parents != null }) });
    return p;
}

/// Suffix-match branches grafted onto the planned drafts, up to cfg.fill_lanes rows.
fn fillLanes(e: *Engine, s: *Stream, p: *Plan) !void {
    const a = e.arena.allocator();
    const first = p.count();
    const room = @min(@as(usize, e.cfg.fill_lanes) -| (1 + first), s.graft_room orelse std.math.maxInt(u32));
    if (room == 0) return;
    const found = try fill.suffixCandidates(a, s.context.items, 16, e.cfg.fill_match, room, 8);
    if (found.len == 0 or (s.graft_hit < e.cfg.fill_floor and s.rounds % 8 != 0)) return;
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
    if (p.kind == .none) p.kind = .fill;
}

/// The planned drafts' trunk of `levels` lanes: host drafts as they are, held ones with the head's ranks (null: unread).
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

/// Copied spans backed by `enter_match` matching tokens (Python `_copy_proposal`); the proposer owns them.
pub fn copyProposal(e: *Engine, s: *Stream) ![]const u32 {
    const p = s.proposer orelse return &.{};
    if (e.cfg.max_copy == 0 or s.force.items.len > 0) return &.{};
    const width: u32 = if (e.alone) (s.copy_width orelse e.cfg.first_copy) else e.cfg.first_copy;
    const copied = p.propose(s.context.items, @min(@as(i64, width), s.draftRoom() - 1)) catch return &.{};
    if (copied.len < 2 or p.lastMatch() < e.cfg.enter_match) return &.{};
    return copied;
}

pub fn copyAhead(ptr: *anyopaque, w: *const depth.Who) bool {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *Stream = @ptrCast(@alignCast(w.stream.?));
    const copied = copyProposal(e, s) catch return false;
    return copied.len > 0;
}

pub fn probe(e: *Engine) depth.Probe {
    return .{ .ptr = e, .copy = copyAhead };
}

pub fn who(s: *Stream) depth.Who {
    return .{ .state = &s.depth, .draft_room = s.draftRoom(), .finished = s.finished, .forced = s.force.items.len > 0, .granted = s.granted, .stream = s };
}

/// Trim draft prefixes by landing probability to the shared width with the most expected tokens a ms.
pub fn allocate(e: *Engine, plans: []Plan) !void {
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
            pr.* = try nodeChances(e, p.stream, n);
        }
    }
    const counts = try alloc.allocate(a, fixed, probs, &e.cfg.shared_costs, e.rule.overheadMs(plans.len), @max(plans.len, e.cfg.batch_rows));
    for (plans, probs, counts) |*p, chances, c| {
        if (p.kind == .head) p.stream.granted = c;
        if (chances.len == 0 or c == chances.len) continue;
        if (c == 0) {
            p.kind = .none;
            p.held = 0;
            p.tokens = &.{};
            p.parents = null;
        } else if (p.parents) |parents| {
            const tree = try accept.sanitizeTree(a, p.tokens, parents, c);
            p.tokens = tree.tokens;
            p.parents = tree.parents;
        } else if (p.held > 0) {
            p.held = c;
        } else p.tokens = p.tokens[0..c];
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

/// The head's chance for each held draft, else chain chances from the stream's per-depth acceptance.
pub fn nodeChances(e: *Engine, s: *Stream, n: usize) ![]const f64 {
    const a = e.arena.allocator();
    if (e.cfg.node_probabilities) {
        if (e.backend.vtable.probabilities) |chances| {
            const out = try a.alloc(f64, n);
            if (try chances(e.backend.ptr, s, out)) return out;
        }
    }
    return alloc.chainProbabilities(a, try e.rule.rates(who(s)), n);
}

/// Structured output (Python Constraint.window): the planned rows an accepted path can use under the stream's grammar,
/// and each row's allowed-token bits. The grammar walks host tokens: held drafts come back as the head's ranked chain,
/// and a held tree (or a head without ranks) verifies no drafts.
pub fn constrain(e: *Engine, p: *Plan) !void {
    const g = p.stream.grammar orelse return;
    const a = e.arena.allocator();
    if (p.held > 0) {
        const chain = for (0..p.held) |k| {
            if (p.parents) |q| if (q[k] != @as(i32, @intCast(k)) - 1) break null;
        } else (try headTrunk(e, p.stream, p.*, p.held)) orelse null;
        if (chain) |c| {
            p.tokens = try std.mem.concat(a, u32, &.{ c.tokens, p.tokens });
        } else {
            p.tokens = &.{};
            p.parents = null;
            p.branch_from = 0;
        }
        p.held = 0;
        p.lanes = null;
    }
    const rows = try a.alloc(u32, 1 + p.tokens.len);
    rows[0] = p.stream.pending.?;
    @memcpy(rows[1..], p.tokens);
    const w = try g.window(a, rows, try accept.rowParents(a, rows.len, p.parents), p.kind == .forced);
    p.masks = w.masks;
    if (w.rows.len == rows.len) return;
    const kept = try a.alloc(u32, w.rows.len - 1);
    for (kept, w.rows[1..]) |*t, r| t.* = rows[r];
    p.tokens = kept;
    if (p.parents != null) {
        const parents = try a.alloc(i32, kept.len);
        for (parents, w.parents[1..]) |*q, r| q.* = r - 1;
        p.parents = if (accept.isChain(parents)) null else parents;
    }
    if (p.branch_from > 0) {
        var below: u32 = 0;
        for (w.rows) |r| below += @intFromBool(r < p.branch_from);
        p.branch_from = below;
    }
    if (kept.len == 0 and p.kind != .forced) p.kind = .none;
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
