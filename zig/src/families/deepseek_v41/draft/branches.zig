//! Tree verify windows on a chain-only target: siblings run lazily as fresh chain windows on the slot itself, exact.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");

pub const max_rows = 64;
pub const unrun: u32 = std.math.maxInt(u32);
const none: u8 = std.math.maxInt(u8);

pub const Error = error{ TreeRows, TreeParents, TreeDraws, BranchUnstable, NotResolved, NoTree };

/// A target's chain operations on one slot.
pub const Chains = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Chain window: `tokens` at `start` (row r at start + r, choice keyed at draws[r]); unresolved keeps all rows.
        run: *const fn (ptr: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, sampling: ?lanes.Sampling, choices: []u32) anyerror!void,
        /// Drop the slot's pending chain window without committing a row (the slot stays at its start).
        drop: *const fn (ptr: *anyopaque, slot: u32) anyerror!void,
    };
};

/// One slot's resolved tree window: the last chain's rows (tree row ids) and each tree row's row in it.
pub const Resolver = struct {
    live: bool = false,
    /// the last chain: rows[i] is the tree row at chain row i
    rows: [max_rows]u8 = undefined,
    n: u8 = 0,
    /// tree row -> its row in the last chain (none: not in it); the taps' map for lanes' ingest
    of: [max_rows]u32 = undefined,
    tree_rows: u8 = 0,
    stats: Stats = .{},

    pub const Stats = struct { trees: u64 = 0, chains: u64 = 0, rerun_rows: u64 = 0, deep: u64 = 0 };

    /// Resolve tree window `s` (parents set) through `ch` into `choices`; the accepted path's chain stays pending.
    pub fn window(r: *Resolver, ch: Chains, s: iface.Segment, choices: []u32) !void {
        const n = s.tokens.len;
        const parents = s.parents orelse return error.NoTree;
        if (n == 0 or n > max_rows or parents.len != n or s.draws.len != n or choices.len < n) return error.TreeRows;
        // each row's children in row order (a linked list: first child, next sibling), parents before children
        var first: [max_rows]u8 = undefined;
        var next: [max_rows]u8 = undefined;
        var last: [max_rows]u8 = undefined;
        @memset(first[0..n], none);
        @memset(next[0..n], none);
        if (parents[0] != -1) return error.TreeParents;
        for (parents[1..], 1..) |q, row| {
            if (q < 0 or q >= row) return error.TreeParents;
            const p: usize = @intCast(q);
            if (s.draws[row] != s.draws[p] + 1) return error.TreeDraws; // a child sits one position after its parent
            if (first[p] == none) first[p] = @intCast(row) else next[last[p]] = @intCast(row);
            last[p] = @intCast(row);
        }
        @memset(choices[0..n], unrun);
        r.live = false;
        r.stats.trees += 1;
        var tok: [max_rows]u32 = undefined;
        var draws: [max_rows]u64 = undefined;
        var picks: [max_rows]u32 = undefined;
        var held: usize = 0; // chain rows already on the path (re-run in front of a branch)
        var head: u8 = 0;
        var runs: usize = 0;
        while (true) {
            // the chain: the path so far, then `head` and its first children
            var m = held;
            var at = head;
            while (at != none) : (at = first[at]) {
                r.rows[m] = at;
                m += 1;
            }
            for (r.rows[0..m], tok[0..m], draws[0..m]) |row, *t, *d| {
                t.* = s.tokens[row];
                d.* = s.draws[row];
            }
            if (runs > 0) try ch.vtable.drop(ch.ptr, s.slot);
            try ch.vtable.run(ch.ptr, s.slot, s.start, tok[0..m], draws[0..m], s.sampling, picks[0..m]);
            runs += 1;
            r.stats.chains += 1;
            r.stats.rerun_rows += held;
            for (r.rows[0..held], picks[0..held]) |row, p| if (choices[row] != p) return error.BranchUnstable;
            for (r.rows[0..m], picks[0..m]) |row, p| choices[row] = p;
            // follow the choices from the head: the first child by row whose token is the row's choice
            var i = held;
            const branch: ?u8 = while (i < m) : (i += 1) {
                var c = first[r.rows[i]];
                while (c != none and s.tokens[c] != picks[i]) c = next[c];
                if (c == none) break null; // the path ends at chain row i
                if (i + 1 < m and r.rows[i + 1] == c) continue;
                break c; // a later sibling: its chain runs behind the path so far
            } else null;
            const b = branch orelse {
                r.n = @intCast(m);
                break;
            };
            held = i + 1;
            head = b;
            r.stats.deep += @intFromBool(held > 1); // a branch below row 0
        }
        @memset(r.of[0..n], unrun);
        for (r.rows[0..r.n], 0..) |row, i| r.of[row] = @intCast(i);
        r.tree_rows = @intCast(n);
        r.live = true;
    }

    /// The kept path (tree rows, row 0 first) as the last chain's accepted count; the map stays until the next window.
    pub fn keep(r: *Resolver, path: []const u32) !u32 {
        if (!r.live) return error.NoTree;
        if (path.len == 0 or path.len > r.n) return error.NotResolved;
        for (path, 0..) |row, i| if (row >= r.tree_rows or r.of[row] != i) return error.NotResolved;
        return @intCast(path.len - 1);
    }

    /// Tree row -> taps row of the last chain (unrun: not in it), while the tree is the slot's last window.
    pub fn map(r: *const Resolver) ?[]const u32 {
        return if (r.live) r.of[0..r.tree_rows] else null;
    }

    /// The slot moved on (a chain window, a prefill, a release): its taps are no longer the tree's.
    pub fn clear(r: *Resolver) void {
        r.live = false;
    }
};
