//! Tree windows on the Metal target: branches on shadow slots choose as serial decode, keep as the path's chain would.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fs = @import("fixture_serve.zig");
const ff = @import("fixture_fwd.zig");

const iface = host.iface;
const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const seed = 71;

/// One tree round's rows: row 0 the pending token, then each row's token and parent row.
const Tree = struct {
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    n: usize = 1,

    fn add(t: *Tree, parent: usize, token: u32) usize {
        t.tokens[t.n] = token;
        t.parents[t.n] = @intCast(parent);
        t.n += 1;
        return t.n - 1;
    }

    /// A chain of `toks` under row `parent`.
    fn chain(t: *Tree, parent: usize, toks: []const u32) void {
        var at = parent;
        for (toks) |x| at = t.add(at, x);
    }
};

/// A token that is not `x`.
fn other(x: u32, k: u32) u32 {
    return (x + 7 + k) % fs.config().vocab;
}

/// Round `r`'s tree at want[i] (pending): main wins, a sibling wins, the second sibling wins, none, a deeper branch.
fn build(r: usize, w: []const u32) Tree {
    var t: Tree = .{};
    t.tokens[0] = w[0];
    t.parents[0] = -1;
    switch (r % 5) {
        0 => {
            t.chain(0, &.{ w[1], w[2], other(w[3], 0) });
            t.chain(0, &.{ other(w[1], 1), w[2] });
        },
        1 => {
            t.chain(0, &.{ other(w[1], 0), w[2] });
            t.chain(0, &.{ w[1], w[2], other(w[3], 1) });
        },
        2 => {
            t.chain(0, &.{ other(w[1], 0), w[2], w[3] });
            t.chain(0, &.{other(w[1], 1)});
            t.chain(0, &.{ w[1], w[2], w[3] });
        },
        3 => {
            t.chain(0, &.{ other(w[1], 0), other(w[2], 0) });
            t.chain(0, &.{other(w[1], 1)});
        },
        else => {
            t.chain(0, &.{ w[1], other(w[2], 0), other(w[3], 0) });
            t.chain(1, &.{ w[2], w[3] });
        },
    }
    return t;
}

fn tapsRow(tp: iface.Taps, row: u32) []const u16 {
    return @as([*]const u16, @ptrFromInt(tp.address))[@as(usize, row) * tp.stride ..][0..tp.stride];
}

/// One slot's reply in progress: its prompt, the serial reply, the tokens committed and this round's tree.
const Lane = struct { prompt: []const u32, want: []u32 = &.{}, i: usize = 0, tree: Tree = .{}, draws: [16]u64 = undefined, out: [16]u32 = undefined };

/// Serial replies, then tree rounds on `a` (every slot's tree in one call) and the kept chains on `b`, all compared.
fn run(a: *pkg.serve.Metal, b: *pkg.serve.Metal, prompts: []const []const u32, n: usize, smp: ?host.lanes.Sampling, replies: [][]u32) !struct { rounds: usize, both: usize } {
    var lanes: [2]Lane = undefined;
    const ls = lanes[0..prompts.len];
    for (ls, prompts) |*l, p| l.* = .{ .prompt = p, .want = try b.t.serial(0, p, n + 8, smp) };
    defer for (ls) |l| gpa.free(l.want);
    const ta = a.t.target();
    const tb = b.t.target();
    for (ls, 0..) |l, k| {
        try testing.expectEqual(l.want[0], try ta.prefill(@intCast(k), l.prompt, smp, l.prompt.len));
        try testing.expectEqual(l.want[0], try tb.prefill(@intCast(k), l.prompt, smp, l.prompt.len));
    }
    var r: usize = 0;
    var both: usize = 0;
    while (true) : (r += 1) {
        var segs: [2]iface.Segment = undefined;
        var outs: [2][]u32 = undefined;
        var who: [2]usize = undefined;
        var m: usize = 0;
        for (ls, 0..) |*l, k| if (l.i < n) {
            l.tree = build(r + 2 * k, l.want[l.i..]);
            const start: u64 = l.prompt.len + l.i;
            var depth: [16]u64 = undefined;
            for (0..l.tree.n) |x| {
                depth[x] = if (x == 0) 0 else depth[@intCast(l.tree.parents[x])] + 1;
                l.draws[x] = start + depth[x] + 1;
            }
            segs[m] = .{ .slot = @intCast(k), .start = start, .tokens = l.tree.tokens[0..l.tree.n], .parents = l.tree.parents[0..l.tree.n], .draws = l.draws[0..l.tree.n], .sampling = smp };
            outs[m] = l.out[0..l.tree.n];
            who[m] = k;
            m += 1;
        };
        if (m == 0) break;
        both += @intFromBool(m == 2);
        try ta.window(segs[0..m], outs[0..m]);
        for (who[0..m]) |k| {
            const l = &ls[k];
            const tree = &l.tree;
            // the accepted path: from row 0, the child whose token is its parent's choice
            var path: [16]u32 = undefined;
            path[0] = 0;
            var len: usize = 1;
            walk: while (true) {
                for (1..tree.n) |x| if (tree.parents[x] == path[len - 1] and tree.tokens[x] == l.out[path[len - 1]]) {
                    path[len] = @intCast(x);
                    len += 1;
                    continue :walk;
                };
                break;
            }
            for (path[0..len], 0..) |row, j| try testing.expectEqual(l.want[l.i + 1 + j], l.out[row]);
            try ta.keep(@intCast(k), path[0..len]);
            // the accepted path as a chain on `b`: the same choices and the same taps, bit for bit
            var bout: [16]u32 = undefined;
            var bouts = [_][]u32{bout[0..len]};
            const start: u64 = l.prompt.len + l.i;
            try tb.window(&.{.{ .slot = @intCast(k), .start = start, .tokens = l.want[l.i..][0..len], .parents = null, .draws = l.draws[0..len], .sampling = smp }}, &bouts);
            try testing.expectEqualSlices(u32, l.want[l.i + 1 ..][0..len], bout[0..len]);
            const tpa = ta.taps(@intCast(k));
            const tpb = tb.taps(@intCast(k));
            const map = tpa.map orelse return error.NoTreeMap;
            for (path[0..len], 0..) |row, j| try testing.expectEqualSlices(u16, tapsRow(tpb, @intCast(j)), tapsRow(tpa, map[row]));
            l.i += len;
        }
    }
    for (ls, replies, 0..) |l, *o, k| {
        ta.release(@intCast(k));
        tb.release(@intCast(k));
        o.* = try gpa.dupe(u32, l.want[0..l.i]);
    }
    return .{ .rounds = r, .both = both };
}

test "dsv41 metal tree windows: shadow-slot branches == serial choices, the kept path == its chain (taps, state)" {
    const a = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false, .cache_bytes = 1 << 30, .resident = true });
    var a_open = true;
    defer if (a_open) fs.close(a);
    const b = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    defer fs.close(b);
    const p = try ff.ids(gpa, &fs.config(), 130, 40);
    defer gpa.free(p);
    const hot: host.lanes.Sampling = .{ .seed = 77, .temperature = 0.8, .top_k = 0, .top_p = 1.0 };
    var gr: [1][]u32 = undefined;
    const g = try run(a, b, &.{p}, 70, null, &gr);
    defer gpa.free(gr[0]);
    var sr: [1][]u32 = undefined;
    const s = try run(a, b, &.{p}, 70, hot, &sr);
    defer gpa.free(sr[0]);
    const st = a.t.tree.stats;
    // a later turn from the state the trees left (prompt cache) decodes as a fresh slot
    const more = try ff.ids(gpa, &fs.config(), 20, 41);
    defer gpa.free(more);
    const turn = try std.mem.concat(gpa, u32, &.{ p, sr[0], more });
    defer gpa.free(turn);
    const hits0 = a.t.stats.hits;
    const cached = try a.t.serial(0, turn, 24, null);
    defer gpa.free(cached);
    const fresh = try b.t.serial(0, turn, 24, null);
    defer gpa.free(fresh);
    const ok = std.mem.eql(u32, cached, fresh) and a.t.stats.hits > hits0 and a.t.stats.cached_rows >= p.len + sr[0].len;
    std.debug.print("{s} dsv41 metal tree windows: {d} + {d} rounds, {d} + {d} tokens; {d} trees, {d} adopted, {d} binds, {d} shadow rows, {d} rows synced, {d} launches; cached turn after trees {s} fresh\n", .{ if (ok) "PASS" else "FAIL", g.rounds, s.rounds, gr[0].len, sr[0].len, st.trees, st.adopted, st.binds, st.shadow_rows, st.synced_rows, st.launches, if (ok) "==" else "!=" });
    try testing.expect(st.adopted >= 2 * (g.rounds / 5) and st.binds == 2 * 2);
    try testing.expect(ok);
    // the shadows joined the residency owner beside the target's buffers, and left it before they were freed
    var joined: usize = 0;
    for (a.t.tree.shadows.items) |sh| joined += @intFromBool(sh.joined);
    const before = a.parts.resident.?.held.bytes;
    const target_joined = a.t.joined;
    const shadows = a.t.tree.shadows.items.len;
    fs.close(a);
    a_open = false;
    const left = fs.resident_left == true;
    std.debug.print("{s} dsv41 metal tree windows: target joined {}, {d} of {d} shadows joined the residency owner ({d:.1} MiB held), all left at close: {}\n", .{ if (target_joined and joined == 2 and left) "PASS" else "FAIL", target_joined, joined, shadows, @as(f64, @floatFromInt(before)) / (1 << 20), left });
    try testing.expect(target_joined and joined == 2 and left);
}

test "dsv41 metal tree windows: two slots' trees in one window call keep apart, each == serial and its chains" {
    const a = try fs.open(gpa, io, seed, .{ .slots = 2, .drafts = false });
    defer fs.close(a);
    const b = try fs.open(gpa, io, seed, .{ .slots = 2, .drafts = false });
    defer fs.close(b);
    const p0 = try ff.ids(gpa, &fs.config(), 120, 42);
    defer gpa.free(p0);
    const p1 = try ff.ids(gpa, &fs.config(), 170, 43);
    defer gpa.free(p1);
    const hot: host.lanes.Sampling = .{ .seed = 78, .temperature = 0.8, .top_k = 0, .top_p = 1.0 };
    var both: usize = 0;
    var rounds: usize = 0;
    for ([_]?host.lanes.Sampling{ null, hot }) |smp| {
        var rs: [2][]u32 = undefined;
        const got = try run(a, b, &.{ p0, p1 }, 60, smp, &rs);
        for (rs) |x| gpa.free(x);
        both += got.both;
        rounds += got.rounds;
    }
    const st = a.t.tree.stats;
    const ok = both > 0 and st.adopted > 0 and a.t.tree.shadows.items.len >= 3;
    std.debug.print("{s} dsv41 metal tree windows, two slots: {d} rounds, {d} with both slots' trees in one call; {d} trees, {d} adopted, {d} shadows, {d} binds, {d} launches\n", .{ if (ok) "PASS" else "FAIL", rounds, both, st.trees, st.adopted, a.t.tree.shadows.items.len, st.binds, st.launches });
    try testing.expect(ok);
}
