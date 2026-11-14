//! Tree windows on Metal: each branch a chain on a shadow slot equal to its stream's slot below the window, one launch.
const std = @import("std");
const mtl = @import("metal");
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

/// A spare slot state equal to slot `owner`'s below `synced`; `high`: positions it wrote; `joined`: held resident.
pub const Shadow = struct { kv: kv.Slot, owner: ?u32 = null, synced: u64 = 0, high: u64 = 0, joined: bool = false };

/// One branch: tree rows root first (a path, then first children), its ids, and its shadow (null: the main slot).
const Chain = struct { rows: [R]u8 = undefined, ids: [R]u32 = undefined, n: u8 = 0, shadow: ?u8 = null };

/// One slot's tree window: its chains, each tree row's chain and taps row; `pending` until its keep.
pub const Open = struct {
    live: bool = false,
    pending: bool = false,
    start: u64 = 0,
    rows: u8 = 0,
    chains: [R]Chain = undefined,
    n_chains: u8 = 0,
    /// tree row -> the chain it was run for (its other copies are duplicates)
    home: [R]u8 = undefined,
    /// tree row -> its row in the slot's taps (unrun: not in them)
    rowmap: [R]u32 = undefined,
    /// tree rows whose choice is in (the launch phase)
    done: [R]bool = undefined,
    draws: []const u64 = &.{},
    sampling: ?host.lanes.Sampling = null,
    out: []u32 = &.{},

    /// Chains of the tree: one from row 0, one from each row that is not its parent's first child.
    fn split(o: *Open, g: iface.Segment) !void {
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
        o.n_chains = 0;
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
                o.home[x] = o.n_chains;
            }
            for (ch.rows[0..ch.n], ch.ids[0..ch.n]) |row, *id| id.* = g.tokens[row];
            o.chains[o.n_chains] = ch;
            o.n_chains += 1;
        }
        o.rows = @intCast(n);
    }

    /// The taps map of chain `k`: its rows in order, every other tree row unrun.
    fn mapChain(o: *Open, k: usize) void {
        @memset(o.rowmap[0..o.rows], unrun);
        const ch = &o.chains[k];
        for (ch.rows[0..ch.n], 0..) |row, i| o.rowmap[row] = @intCast(i);
    }
};

pub const Tree = struct {
    shadows: std.ArrayList(Shadow) = .empty,
    /// each slot's tree window (made on the first one)
    open: []Open = &.{},
    stats: Stats = .{},

    pub const Stats = struct { trees: u64 = 0, launches: u64 = 0, shadow_rows: u64 = 0, adopted: u64 = 0, binds: u64 = 0, synced_rows: u64 = 0, by_slot: [16]u64 = @splat(0) };

    /// The shadows leave the weights' residency owner, then go.
    pub fn deinit(tr: *Tree, t: *Target) void {
        for (tr.shadows.items) |*sh| {
            var buf: [5 * 64 + 1]mtl.Buffer = undefined;
            if (sh.joined) for (buffers(&sh.kv, &buf)) |b| t.opt.resident.?.remove(b);
            sh.kv.deinit();
        }
        tr.shadows.deinit(t.a);
        t.a.free(tr.open);
    }

    fn of(tr: *const Tree, slot: u32) ?*Open {
        return if (slot < tr.open.len) &tr.open[slot] else null;
    }

    /// The slot's tree window waits for its keep.
    pub fn pending(tr: *const Tree, slot: u32) bool {
        return if (tr.of(slot)) |o| o.pending else false;
    }

    /// The slot's last window's tree rows -> taps rows, while it is a tree.
    pub fn map(tr: *const Tree, slot: u32) ?[]const u32 {
        const o = tr.of(slot) orelse return null;
        return if (o.live) o.rowmap[0..o.rows] else null;
    }

    /// The slot ran another window: its tree is over.
    pub fn close(tr: *Tree, slot: u32) void {
        if (tr.of(slot)) |o| {
            o.live = false;
            o.pending = false;
        }
    }

    /// The slot's request ended or a new one began: its tree is over and no shadow equals it any more.
    pub fn forget(tr: *Tree, slot: u32) void {
        tr.close(slot);
        for (tr.shadows.items) |*sh| if (sh.owner == slot) {
            sh.owner = null;
        };
    }

    /// The tree windows among `segs` into their `outs`: every chain of every tree in launches of R rows at most.
    pub fn window(tr: *Tree, t: *Target, segs: []const iface.Segment, outs: [][]u32) !void {
        var trees: [R]u32 = undefined;
        var n: usize = 0;
        for (segs) |g| if (branched(g)) {
            if (n == R) return error.TooManySlots;
            trees[n] = g.slot;
            n += 1;
        };
        if (n == 0) return;
        if (tr.open.len < t.slots.len) {
            const all = try t.a.alloc(Open, t.slots.len);
            @memset(all, .{});
            t.a.free(tr.open);
            tr.open = all;
        }
        for (segs, outs, 0..) |g, out, i| if (branched(g)) try tr.prepare(t, segs, i, out);
        // every tree's chains in turn (chain 0 first), as many as R rows take in each launch
        const c = t.env.c;
        const V: usize = c.vocab;
        const look = c.engram_max_ngram - 1;
        var ti: usize = 0;
        var ci: usize = 0;
        while (ti < n) {
            var runs: [R]Target.Run = undefined;
            var whose: [R][2]usize = undefined;
            var m: usize = 0;
            var rows: usize = 0;
            while (ti < n) {
                const o = &tr.open[trees[ti]];
                const ch = &o.chains[ci];
                if (m > 0 and rows + ch.n > R) break;
                const s = &t.slots[trees[ti]];
                const st = if (ch.shadow) |j| &tr.shadows.items[j].kv else &s.kv;
                const at: usize = @intCast(o.start);
                runs[m] = .{ .kv = st, .ids = ch.ids[0..ch.n], .start = @intCast(o.start), .lookback = s.hist.items[at -| look..at] };
                whose[m] = .{ ti, ci };
                m += 1;
                rows += ch.n;
                ci += 1;
                if (ci == o.n_chains) {
                    ti += 1;
                    ci = 0;
                }
            }
            t.launchRuns(runs[0..m]) catch |e| return t.fail(e);
            tr.stats.launches += 1;
            t.stats.launches += 1;
            const lg = t.b.logitsOf(c, @intCast(rows));
            var r0: usize = 0;
            for (whose[0..m]) |w| {
                const o = &tr.open[trees[w[0]]];
                const ch = &o.chains[w[1]];
                for (ch.rows[0..ch.n], 0..) |row, i| {
                    const got = try t.choose(lg[(r0 + i) * V ..][0..V], o.draws[row], o.sampling);
                    // a row run again in front of a branch must choose as it did on its own chain
                    if (o.done[row] and o.out[row] != got) return error.BranchUnstable;
                    if (o.home[row] == w[1]) {
                        o.out[row] = got;
                        o.done[row] = true;
                    }
                }
                if (ch.shadow) |j| {
                    const sh = &tr.shadows.items[j];
                    sh.high = @max(sh.high, o.start + ch.n);
                    tr.stats.shadow_rows += ch.n;
                }
                r0 += ch.n;
            }
        }
        for (trees[0..n]) |slot| tr.finish(t, slot);
    }

    /// One tree's checks, chains and shadows (synced to its slot), and its main chain's ids on the slot.
    fn prepare(tr: *Tree, t: *Target, segs: []const iface.Segment, i: usize, out: []u32) !void {
        const g = segs[i];
        const c = t.env.c;
        if (g.slot >= t.slots.len) return error.NoSlot;
        for (segs, 0..) |x, j| if (j != i and x.slot == g.slot) return error.BadRows;
        const s = &t.slots[g.slot];
        if (!s.live) return error.SlotNotLive;
        if (g.start != s.hist.items.len) return error.PositionMismatch;
        if (out.len < g.tokens.len or g.start + g.tokens.len > t.opt.limit) return error.BadRows;
        for (g.tokens) |id| if (id >= c.vocab) return error.BadToken;
        const o = &tr.open[g.slot];
        o.* = .{};
        errdefer o.* = .{};
        try o.split(g);
        o.pending = true; // the shadows it picks are taken for this round
        for (o.chains[1..o.n_chains]) |*ch| ch.shadow = try tr.pick(t, g.slot);
        o.start = g.start;
        o.draws = g.draws;
        o.sampling = g.sampling;
        o.out = out;
        @memset(o.done[0..o.rows], false);
        if (s.open) t.commitTaps(s, s.rows); // a window never kept keeps all its rows
        s.whole = false;
        for (o.chains[1..o.n_chains]) |ch| tr.sync(c, &tr.shadows.items[ch.shadow.?], &s.kv, g.slot, g.start);
        try s.hist.appendSlice(t.a, o.chains[0].ids[0..o.chains[0].n]);
        s.high = @max(s.high, s.hist.items.len);
    }

    /// A shadow for one of `slot`'s chains no pending tree holds: its own first, a free one, another slot's, a new one.
    fn pick(tr: *Tree, t: *Target, slot: u32) !u8 {
        for (0..3) |pass| for (tr.shadows.items, 0..) |sh, j| {
            if (tr.taken(j)) continue;
            const ok = switch (pass) {
                0 => sh.owner == slot,
                1 => sh.owner == null,
                else => true,
            };
            if (ok) return @intCast(j);
        };
        if (tr.shadows.items.len == none) return error.TooManySlots;
        var st = try kv.Slot.init(t.a, t.env.device, t.env.c, t.opt.limit);
        errdefer st.deinit();
        try tr.shadows.append(t.a, .{ .kv = st });
        const sh = &tr.shadows.items[tr.shadows.items.len - 1];
        // beside the target's own buffers in the weights' residency owner, which refuses past its load limit
        var buf: [5 * 64 + 1]mtl.Buffer = undefined;
        if (t.joined) sh.joined = t.opt.resident.?.tryAddNow(buffers(&sh.kv, &buf));
        return @intCast(tr.shadows.items.len - 1);
    }

    /// Shadow `j` runs a chain of a tree still waiting for its keep (this round's included).
    fn taken(tr: *const Tree, j: usize) bool {
        for (tr.open) |*o| if (o.pending) for (o.chains[1..o.n_chains]) |ch| if (ch.shadow) |x| if (x == j) return true;
        return false;
    }

    /// After the launches: the slot's window is its main chain's, its tree waits for the keep.
    fn finish(tr: *Tree, t: *Target, slot: u32) void {
        const o = &tr.open[slot];
        const s = &t.slots[slot];
        s.base = o.start;
        s.rows = o.chains[0].n;
        s.open = true;
        s.after_prompt = false;
        s.tail.end = @intCast(s.hist.items.len);
        s.tail.done = s.tail.end;
        s.whole = true;
        t.stats.windows += 1;
        t.stats.rows += o.rows;
        tr.stats.trees += 1;
        if (slot < tr.stats.by_slot.len) tr.stats.by_slot[slot] += 1;
        o.mapChain(0);
        o.live = true;
    }

    /// Keep `path` (tree rows from row 0): its chain's first rows; a shadow's kept rows are copied into the slot.
    pub fn keep(tr: *Tree, t: *Target, slot: u32, path: []const u32) !void {
        const o = tr.of(slot) orelse return error.NotAPath;
        if (path.len == 0 or path.len > o.rows) return error.NotAPath;
        const last = path[path.len - 1];
        if (last >= o.rows) return error.NotAPath;
        const k = o.home[last];
        const ch = &o.chains[k];
        if (path.len > ch.n) return error.NotAPath;
        for (path, ch.rows[0..path.len]) |r, row| if (r != row) return error.NotAPath;
        o.pending = false;
        o.mapChain(k);
        const j = ch.shadow orelse return;
        const s = &t.slots[slot];
        const sh = &tr.shadows.items[j];
        // the shadow equals the slot below the window: its kept rows, carries, compressor inputs and taps go across
        const m: u32 = @intCast(path.len);
        copyRows(t.env.c, &s.kv, &sh.kv, o.start, o.start + m, false);
        for (s.kv.regions, sh.kv.regions) |d, x| if (d.inputs) |b| @memcpy(b.contents()[0..b.length()], x.inputs.?.contents()[0..b.length()]);
        if (s.kv.taps) |b| {
            const bytes = @as(usize, m) * tapBytes(t.env.c);
            @memcpy(b.contents()[0..bytes], sh.kv.taps.?.contents()[0..bytes]);
        }
        s.hist.shrinkRetainingCapacity(@intCast(o.start));
        try s.hist.appendSlice(t.a, ch.ids[0..ch.n]);
        s.high = @max(s.high, s.hist.items.len);
        s.rows = ch.n;
        s.tail.end = @intCast(s.hist.items.len);
        s.tail.done = s.tail.end;
        tr.stats.adopted += 1;
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

/// Bytes of one row of a slot's taps.
fn tapBytes(c: *const Config) usize {
    return c.dspark_targets.len * @as(usize, c.hidden) * 2;
}

/// A slot state's buffers (its regions' and taps), in `out`.
fn buffers(st: *const kv.Slot, out: *[5 * 64 + 1]mtl.Buffer) []const mtl.Buffer {
    var n: usize = 0;
    for (st.regions) |r| {
        out[n] = r.window;
        n += 1;
        inline for (.{ "pool", "keys", "carry", "inputs" }) |f| if (@field(r, f)) |b| {
            out[n] = b;
            n += 1;
        };
    }
    if (st.taps) |b| {
        out[n] = b;
        n += 1;
    }
    return out[0..n];
}

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
    var o: Open = .{};
    const parents = [_]i32{ -1, 0, 1, 0, 3, 0, 2 };
    const draws = [_]u64{ 5, 6, 7, 6, 7, 6, 8 };
    const toks = [_]u32{ 10, 11, 12, 13, 14, 15, 16 };
    try o.split(.{ .slot = 0, .start = 4, .tokens = &toks, .parents = &parents, .draws = &draws, .sampling = null });
    try std.testing.expectEqual(@as(u8, 3), o.n_chains);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 6 }, o.chains[0].rows[0..o.chains[0].n]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 3, 4 }, o.chains[1].rows[0..o.chains[1].n]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 5 }, o.chains[2].rows[0..o.chains[2].n]);
    try std.testing.expectEqualSlices(u32, &.{ 10, 13, 14 }, o.chains[1].ids[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 1, 1, 2, 0 }, o.home[0..7]);
    const bad = [_]u64{ 5, 6, 8, 6, 7, 6, 8 };
    try std.testing.expectError(error.TreeDraws, o.split(.{ .slot = 0, .start = 4, .tokens = &toks, .parents = &parents, .draws = &bad, .sampling = null }));
}
