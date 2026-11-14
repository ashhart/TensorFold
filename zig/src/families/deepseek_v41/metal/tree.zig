//! Tree windows on Metal: each branch a chain on a shadow slot equal to the main slot below the window, one launch.
const std = @import("std");
const host = @import("dsv41");
const kv = @import("kv.zig");
const snapshots = @import("snapshots.zig");
const pipelines = @import("pipelines.zig");
const target_ = @import("target.zig");

const iface = host.iface;
const Config = host.config.Config;
const Target = target_.Target;
const R = pipelines.max_rows;
pub const unrun: u32 = std.math.maxInt(u32);
const none: u8 = std.math.maxInt(u8);

/// True when `g`'s rows are not one chain from row 0.
pub fn branched(g: iface.Segment) bool {
    const ps = g.parents orelse return false;
    for (ps, 0..) |q, r| if (q != @as(i32, @intCast(r)) - 1) return true;
    return false;
}

/// A spare slot state: equal to slot `owner`'s below position `synced`; `high`: the most positions it wrote.
pub const Shadow = struct { kv: kv.Slot, owner: ?u32 = null, synced: u64 = 0, high: u64 = 0 };

/// One branch: tree rows root first (a path, then first children), its ids, and its shadow (null: the main slot).
const Chain = struct { rows: [R]u8 = undefined, ids: [R]u32 = undefined, n: u8 = 0, shadow: ?u8 = null };

pub const Tree = struct {
    shadows: std.ArrayList(Shadow) = .empty,
    /// the slot's last window was a tree (`map` holds); `pending` until its keep
    live: bool = false,
    pending: bool = false,
    slot: u32 = 0,
    start: u64 = 0,
    rows: u8 = 0,
    chains: [R]Chain = undefined,
    n_chains: u8 = 0,
    /// tree row -> the chain it was run for (its other copies are duplicates)
    home: [R]u8 = undefined,
    /// tree row -> its row in the slot's taps (unrun: not in them)
    rowmap: [R]u32 = undefined,
    stats: Stats = .{},

    pub const Stats = struct { trees: u64 = 0, launches: u64 = 0, shadow_rows: u64 = 0, swaps: u64 = 0, binds: u64 = 0, synced_rows: u64 = 0 };

    pub fn deinit(tr: *Tree, a: std.mem.Allocator) void {
        for (tr.shadows.items) |*sh| sh.kv.deinit();
        tr.shadows.deinit(a);
    }

    /// The slot's tree window waits for its keep.
    pub fn open(tr: *const Tree, slot: u32) bool {
        return tr.pending and tr.slot == slot;
    }

    /// The slot's last window's tree rows -> taps rows, while it is a tree.
    pub fn map(tr: *const Tree, slot: u32) ?[]const u32 {
        return if (tr.live and tr.slot == slot) tr.rowmap[0..tr.rows] else null;
    }

    /// The slot ran another window: its tree is over.
    pub fn close(tr: *Tree, slot: u32) void {
        if (tr.slot == slot) {
            tr.live = false;
            tr.pending = false;
        }
    }

    /// The slot's request ended or a new one began: its tree is over and no shadow equals it any more.
    pub fn forget(tr: *Tree, slot: u32) void {
        tr.close(slot);
        for (tr.shadows.items) |*sh| if (sh.owner == slot) {
            sh.owner = null;
        };
    }

    /// Chains of the tree: one from row 0, one from each row that is not its parent's first child.
    fn split(tr: *Tree, g: iface.Segment) !void {
        const ps = g.parents.?;
        const n = ps.len;
        if (n == 0 or n > R or g.draws.len != n or ps[0] != -1) return error.TreeRows;
        var first: [R]u8 = @splat(none);
        for (ps[1..], 1..) |q, r| {
            if (q < 0 or q >= r) return error.TreeParents;
            const p: usize = @intCast(q);
            if (g.draws[r] != g.draws[p] + 1) return error.TreeDraws; // a child sits one position after its parent
            if (first[p] == none) first[p] = @intCast(r);
        }
        tr.n_chains = 0;
        for (0..n) |r| {
            if (r > 0 and first[@intCast(ps[r])] == r) continue;
            var ch: Chain = .{};
            var path: [R]u8 = undefined;
            var d: usize = 0;
            var at = ps[r];
            while (at >= 0) : (at = ps[@intCast(at)]) {
                path[d] = @intCast(at);
                d += 1;
            }
            for (0..d) |i| ch.rows[i] = path[d - 1 - i];
            ch.n = @intCast(d);
            var x: u8 = @intCast(r);
            while (x != none) : (x = first[x]) {
                ch.rows[ch.n] = x;
                ch.n += 1;
                tr.home[x] = tr.n_chains;
            }
            for (ch.rows[0..ch.n], ch.ids[0..ch.n]) |row, *id| id.* = g.tokens[row];
            if (tr.n_chains > 0) ch.shadow = tr.n_chains - 1;
            tr.chains[tr.n_chains] = ch;
            tr.n_chains += 1;
        }
        tr.rows = @intCast(n);
    }

    /// Tree window `g` into `out`: chain 0 on the slot, each other chain on a shadow synced to it, R rows a launch.
    pub fn window(tr: *Tree, t: *Target, g: iface.Segment, out: []u32) !void {
        const c = t.env.c;
        const V: usize = c.vocab;
        if (g.slot >= t.slots.len) return error.NoSlot;
        const s = &t.slots[g.slot];
        if (!s.live) return error.SlotNotLive;
        if (g.start != s.hist.items.len) return error.PositionMismatch;
        if (out.len < g.tokens.len or g.start + g.tokens.len > t.opt.limit) return error.BadRows;
        for (g.tokens) |id| if (id >= c.vocab) return error.BadToken;
        tr.close(g.slot);
        try tr.split(g);
        while (tr.shadows.items.len + 1 < tr.n_chains) {
            var st = try kv.Slot.init(t.a, t.env.device, c, t.opt.limit);
            errdefer st.deinit();
            try tr.shadows.append(t.a, .{ .kv = st });
        }
        if (s.open) t.commitTaps(s, s.rows); // a window never kept keeps all its rows
        s.whole = false;
        for (tr.chains[1..tr.n_chains]) |ch| tr.sync(c, &tr.shadows.items[ch.shadow.?], &s.kv, g.slot, g.start);
        const main = tr.chains[0];
        try s.hist.appendSlice(t.a, main.ids[0..main.n]);
        s.high = @max(s.high, s.hist.items.len);
        const look = c.engram_max_ngram - 1;
        const lookback = s.hist.items[@as(usize, @intCast(g.start)) -| look..@intCast(g.start)];
        var done: [R]bool = @splat(false);
        var from: usize = 0;
        while (from < tr.n_chains) {
            var runs: [R]Target.Run = undefined;
            var rows: usize = 0;
            var to = from;
            while (to < tr.n_chains and (to == from or rows + tr.chains[to].n <= R)) : (to += 1) {
                const ch = &tr.chains[to];
                const st = if (ch.shadow) |j| &tr.shadows.items[j].kv else &s.kv;
                runs[to - from] = .{ .kv = st, .ids = ch.ids[0..ch.n], .start = @intCast(g.start), .lookback = lookback };
                rows += ch.n;
            }
            t.launchRuns(runs[0 .. to - from]) catch |e| return t.fail(e);
            tr.stats.launches += 1;
            t.stats.launches += 1;
            const lg = t.b.logitsOf(c, @intCast(rows));
            var r0: usize = 0;
            for (tr.chains[from..to], from..) |*ch, k| {
                for (ch.rows[0..ch.n], 0..) |row, i| {
                    const pick = try t.choose(lg[(r0 + i) * V ..][0..V], g.draws[row], g.sampling);
                    // a row run again in front of a branch must choose as it did on its own chain
                    if (done[row] and out[row] != pick) return error.BranchUnstable;
                    if (tr.home[row] == k) out[row] = pick;
                    done[row] = done[row] or tr.home[row] == k;
                }
                if (ch.shadow) |j| {
                    const sh = &tr.shadows.items[j];
                    sh.high = @max(sh.high, g.start + ch.n);
                    tr.stats.shadow_rows += ch.n;
                }
                r0 += ch.n;
            }
            from = to;
        }
        s.base = g.start;
        s.rows = main.n;
        s.open = true;
        s.after_prompt = false;
        s.tail.end = @intCast(s.hist.items.len);
        s.tail.done = s.tail.end;
        s.whole = true;
        t.stats.windows += 1;
        t.stats.rows += g.tokens.len;
        tr.stats.trees += 1;
        tr.slot = g.slot;
        tr.start = g.start;
        tr.mapChain(0);
        tr.live = true;
        tr.pending = true;
    }

    /// The taps map of chain `k`: its rows in order, every other tree row unrun.
    fn mapChain(tr: *Tree, k: usize) void {
        @memset(tr.rowmap[0..tr.rows], unrun);
        const ch = &tr.chains[k];
        for (ch.rows[0..ch.n], 0..) |row, i| tr.rowmap[row] = @intCast(i);
    }

    /// Keep `path` (tree rows from row 0): its chain's first rows; a shadow's chain trades states with the slot.
    pub fn keep(tr: *Tree, t: *Target, slot: u32, path: []const u32) !void {
        if (path.len == 0 or path.len > tr.rows) return error.NotAPath;
        const last = path[path.len - 1];
        if (last >= tr.rows) return error.NotAPath;
        const k = tr.home[last];
        const ch = &tr.chains[k];
        if (path.len > ch.n) return error.NotAPath;
        for (path, ch.rows[0..path.len]) |r, row| if (r != row) return error.NotAPath;
        tr.pending = false;
        tr.mapChain(k);
        const j = ch.shadow orelse return;
        const s = &t.slots[slot];
        const sh = &tr.shadows.items[j];
        // the shadow holds the branch's rows and equals the slot below the window: they trade states
        std.mem.swap(kv.Slot, &s.kv, &sh.kv);
        const h = @max(s.high, sh.high);
        s.high = h;
        sh.high = h;
        sh.owner = slot;
        sh.synced = tr.start;
        s.hist.shrinkRetainingCapacity(@intCast(tr.start));
        try s.hist.appendSlice(t.a, ch.ids[0..ch.n]);
        s.rows = ch.n;
        s.tail.end = @intCast(s.hist.items.len);
        s.tail.done = s.tail.end;
        tr.stats.swaps += 1;
    }

    /// `sh` made equal to `main` below `start`: cleared and copied whole when bound elsewhere, else the new rows.
    fn sync(tr: *Tree, c: *const Config, sh: *Shadow, main: *const kv.Slot, slot: u32, start: u64) void {
        const fresh = sh.owner != slot or sh.synced > start;
        if (fresh) {
            snapshots.clear(c, &sh.kv, sh.high);
            sh.owner = slot;
            sh.synced = 0;
            sh.high = 0;
            tr.stats.binds += 1;
        }
        copyRows(c, &sh.kv, main, sh.synced, start, fresh);
        tr.stats.synced_rows += start - sh.synced;
        sh.synced = start;
        sh.high = @max(sh.high, start);
    }
};

/// Positions [from, to) of `src` into `dst`: window rows (all when `whole`), pool and key rows they reach, carries.
pub fn copyRows(c: *const Config, dst: *const kv.Slot, src: *const kv.Slot, from: u64, to: u64, whole: bool) void {
    const nw = kv.windowSlots(c);
    const row = @as(usize, c.head_dim) * 2;
    for (dst.regions, src.regions, 0..) |d, s, l| {
        const dw = d.window.contents();
        const sw = s.window.contents();
        if (whole or to - from >= nw) @memcpy(dw[0..d.window.length()], sw[0..d.window.length()]) else {
            var p = from;
            while (p < to) : (p += 1) for (kv.windowRows(c, @intCast(p))) |w| @memcpy(dw[w * row ..][0..row], sw[w * row ..][0..row]);
        }
        if (d.pool) |dp| {
            const ratio = c.compressRatio(@intCast(l));
            const all = kv.poolRows(c, @intCast(l), dst.limit);
            const lo: usize = @min(all, from / ratio);
            const hi: usize = @min(all, to / ratio + 1);
            @memcpy(dp.contents()[lo * row .. hi * row], s.pool.?.contents()[lo * row .. hi * row]);
            const kb = kv.index_key_bytes;
            @memcpy(d.keys.?.contents()[lo * kb .. hi * kb], s.keys.?.contents()[lo * kb .. hi * kb]);
        }
        if (d.carry) |dc| @memcpy(dc.contents()[0..dc.length()], s.carry.?.contents()[0..dc.length()]);
    }
}

test "chains of a tree: one from row 0, one a later sibling, each with the path before it" {
    var tr: Tree = .{};
    const parents = [_]i32{ -1, 0, 1, 0, 3, 0, 2 };
    const draws = [_]u64{ 5, 6, 7, 6, 7, 6, 8 };
    const toks = [_]u32{ 10, 11, 12, 13, 14, 15, 16 };
    try tr.split(.{ .slot = 0, .start = 4, .tokens = &toks, .parents = &parents, .draws = &draws, .sampling = null });
    try std.testing.expectEqual(@as(u8, 3), tr.n_chains);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 6 }, tr.chains[0].rows[0..tr.chains[0].n]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 3, 4 }, tr.chains[1].rows[0..tr.chains[1].n]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 5 }, tr.chains[2].rows[0..tr.chains[2].n]);
    try std.testing.expectEqualSlices(u32, &.{ 10, 13, 14 }, tr.chains[1].ids[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 1, 1, 2, 0 }, tr.home[0..7]);
    try std.testing.expectEqual(@as(?u8, 1), tr.chains[2].shadow);
    const bad = [_]u64{ 5, 6, 8, 6, 7, 6, 8 };
    try std.testing.expectError(error.TreeDraws, tr.split(.{ .slot = 0, .start = 4, .tokens = &toks, .parents = &parents, .draws = &bad, .sampling = null }));
}
