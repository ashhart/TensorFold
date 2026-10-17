//! The DSpark depth as the lane engine's row policy (lanes.trim); commits feed calibration, rates, counterfactual.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const lt = lanes.trim;
const depth = @import("depth.zig");
const calib = @import("calib.zig");
const tree = @import("tree.zig");
const Costs = @import("costs.zig").Costs;
const ln = @import("lanes.zig");
const acclog = @import("acclog.zig");

pub const Policy = struct {
    gpa: Allocator,
    depth: depth.Depth,
    sibcal: calib.SibCal = .{},
    owner: *ln.Lanes = undefined,
    sib_rows: usize,
    dup: tree.Dup,
    trees: u64 = 0, // rounds that verified a sibling
    sib_wins: u64 = 0,

    pub fn init(gpa: Allocator, costs: Costs, opts: ln.Options) !Policy {
        return .{ .gpa = gpa, .depth = try depth.Depth.init(gpa, costs, opts.shape.block, opts.depth), .sib_rows = opts.sib_rows, .dup = opts.dup };
    }

    pub fn deinit(p: *Policy) void {
        p.depth.deinit();
    }

    pub fn rows(p: *Policy) lt.Trim {
        return .{ .ptr = p, .vtable = &.{ .choose = choose, .commit = commit } };
    }

    fn self(ptr: *anyopaque) *Policy {
        return @ptrCast(@alignCast(ptr));
    }

    fn choose(ptr: *anyopaque, windows: []const lt.Window, alone: bool, arena: Allocator, out: []lt.Choice) anyerror!void {
        try chooseRows(ptr, windows, alone, arena, out);
        if (std.c.getenv("TF_DSV41_ACCEPT_LOG") != null) try logRound(self(ptr), windows, arena);
    }

    /// TF_DSV41_ACCEPT_LOG: the round's windows, each drafted slot's raw / calibrated confidences and chosen depth.
    fn logRound(p: *Policy, windows: []const lt.Window, arena: Allocator) !void {
        const x = p.owner;
        var total: u64 = 0;
        var others: u64 = 0;
        var la: std.ArrayList(acclog.Ask) = .empty;
        for (windows) |w| {
            total += w.rows;
            const st = x.state(w.stream);
            if (!w.drafted or st == null or st.?.n == 0) {
                others += w.rows;
                continue;
            }
            const s = st.?;
            var k: usize = 0;
            if (s.shown_set) for (s.shown.items) |n| {
                k += @intFromBool(n < s.n);
            };
            const main = @min(s.n, w.rows - 1);
            const q = try arena.alloc(f64, main);
            for (s.conf[0..main], q, 0..) |c, *v, j| v.* = p.depth.cal.q(j, c);
            try la.append(arena, .{ .slot = s.slot, .start = s.start, .most = main, .k = k, .conf = s.conf[0..main], .q = q });
        }
        acclog.choose(windows.len, total, others, la.items);
    }

    fn chooseRows(ptr: *anyopaque, windows: []const lt.Window, alone: bool, arena: Allocator, out: []lt.Choice) anyerror!void {
        const p = self(ptr);
        const x = p.owner;
        var asks: std.ArrayList(depth.Depth.Ask) = .empty;
        var at: std.ArrayList(usize) = .empty; // the window of each ask
        var others: u64 = 0;
        var other_windows: usize = 0;
        for (windows, 0..) |w, i| {
            const st = x.state(w.stream);
            if (!w.drafted or st == null or st.?.n == 0) {
                others += w.rows;
                other_windows += 1;
                continue;
            }
            const main = @min(st.?.n, w.rows - 1);
            const conf = try arena.dupe(f64, st.?.conf[0..main]);
            try asks.append(arena, .{ .slot = st.?.slot, .conf = conf, .most = main, .first = st.?.held[0], .position = st.?.start });
            try at.append(arena, i);
        }
        if (asks.items.len == 0) return;
        const single = alone and asks.items.len == 1 and other_windows == 0;
        if (single) {
            const i = at.items[0];
            const st = x.state(windows[i].stream).?;
            if (st.n_sib > 0 and windows[i].parents != null and windows[i].rows - 1 == st.nodes()) {
                out[i] = try p.loneTree(arena, st, asks.items[0]);
                return;
            }
        }
        const ks = try arena.alloc(usize, asks.items.len);
        try p.depth.chooseJoint(asks.items, others, other_windows, single, ks);
        for (at.items, ks) |i, k| {
            const st = x.state(windows[i].stream).?;
            out[i] = .{ .count = @intCast(k) };
            try show(p.gpa, st, k, &.{});
        }
    }

    /// A lone window with siblings: tree.plan picks the main chain's k and each sibling's rows.
    fn loneTree(p: *Policy, arena: Allocator, st: *ln.St, ask: depth.Depth.Ask) !lt.Choice {
        var sibs: [tree.max_siblings]f64 = undefined;
        for (st.sib[0..st.n_sib], sibs[0..st.n_sib], 0..) |s, *q, j| q.* = p.sibcal.q(j, s.p);
        const c = try p.depth.chooseTree(ask.slot, ask.conf, ask.most, sibs[0..st.n_sib], true, ask.first, p.dup, p.sib_rows);
        var lens: [tree.max_siblings]u32 = @splat(0);
        var total: u32 = @intCast(c.k);
        for (lens[0..st.n_sib], c.lens[0..st.n_sib], st.sib_len[0..st.n_sib]) |*l, want, have| {
            l.* = @intCast(@min(want, have));
            total += l.*;
        }
        if (total == c.k) {
            try show(p.gpa, st, c.k, &.{});
            return .{ .count = @intCast(c.k) };
        }
        // the main chain's first k, then each sibling's first rows (node indices in the held tree's order)
        const nodes = try arena.alloc(u32, total);
        for (nodes[0..c.k], 0..) |*n, i| n.* = @intCast(i);
        var at: usize = c.k;
        var node: u32 = st.n;
        for (lens[0..st.n_sib], st.sib_len[0..st.n_sib]) |l, have| {
            for (0..l) |i| nodes[at + i] = node + @as(u32, @intCast(i));
            at += l;
            node += have;
        }
        try show(p.gpa, st, c.k, lens[0..st.n_sib]);
        return .{ .count = total, .nodes = nodes };
    }

    /// Remember what the window verifies: `k` main drafts, then `lens[j]` rows of sibling j.
    fn show(gpa: Allocator, st: *ln.St, k: usize, lens: []const u32) !void {
        st.shown.clearRetainingCapacity();
        for (0..k) |i| try st.shown.append(gpa, @intCast(i));
        var node: u32 = st.n;
        for (lens, st.sib_len[0..lens.len]) |l, have| {
            for (0..l) |i| try st.shown.append(gpa, node + @as(u32, @intCast(i)));
            node += have;
        }
        st.shown_set = true;
    }

    /// The sibling a node belongs to (null: the main chain).
    fn siblingOf(st: *const ln.St, node: u32) ?usize {
        if (node < st.n) return null;
        var at: u32 = st.n;
        for (st.sib_len[0..st.n_sib], 0..) |l, j| {
            if (node < at + l) return j;
            at += l;
        }
        return null;
    }

    fn commit(ptr: *anyopaque, c: lt.Commit) anyerror!void {
        const p = self(ptr);
        const st = p.owner.state(c.stream) orelse return;
        defer st.shown_set = false;
        acclog.commit(st.slot, c.rows, c.path.len, c.drafted, c.stream.finished, c.stream.emitted().len, c.stream.max_new);
        if (!st.shown_set or !c.drafted) {
            try p.depth.record(st.slot, c.rows, c.path.len, null);
            p.depth.bonus(st.slot, c.bonus);
            return;
        }
        var main_rows: usize = 1;
        var any_sib = false;
        for (st.shown.items) |n| {
            if (n < st.n) main_rows += 1 else any_sib = true;
        }
        if (!any_sib) {
            try p.depth.record(st.slot, c.rows, c.path.len, null);
            p.depth.bonus(st.slot, c.bonus);
            return;
        }
        // the branch the target chose: its first row's node, a sibling root or the main chain's first draft
        const won: ?usize = if (c.path.len > 1) siblingOf(st, st.shown.items[c.path[1] - 1]) else null;
        var seen: [tree.max_siblings]bool = @splat(false);
        for (st.shown.items) |n| if (siblingOf(st, n)) |j| {
            seen[j] = true;
        };
        for (seen[0..st.n_sib], st.sib[0..st.n_sib], 0..) |s, sib, j| if (s) p.sibcal.record(j, sib.p, won == j);
        p.trees += 1;
        p.sib_wins += @intFromBool(won != null);
        const main_keep = if (won == null) c.path.len else 1;
        try p.depth.record(st.slot, main_rows, main_keep, c.path.len);
        p.depth.bonus(st.slot, c.bonus);
    }
};
