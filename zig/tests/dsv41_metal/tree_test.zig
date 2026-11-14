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

/// The serial reply, then the same reply through tree rounds on `a` and the accepted chains on `b`, all compared.
fn run(a: *pkg.serve.Metal, b: *pkg.serve.Metal, prompt: []const u32, n: usize, smp: ?host.lanes.Sampling) !struct { rounds: usize, reply: []u32 } {
    const want = try b.t.serial(0, prompt, n + 8, smp);
    defer gpa.free(want);
    const ta = a.t.target();
    const tb = b.t.target();
    try testing.expectEqual(want[0], try ta.prefill(0, prompt, smp, prompt.len));
    try testing.expectEqual(want[0], try tb.prefill(0, prompt, smp, prompt.len));
    var i: usize = 0;
    var r: usize = 0;
    while (i < n) : (r += 1) {
        const tree = build(r, want[i..]);
        const start: u64 = prompt.len + i;
        var draws: [16]u64 = undefined;
        var depth: [16]u64 = undefined;
        for (0..tree.n) |k| {
            depth[k] = if (k == 0) 0 else depth[@intCast(tree.parents[k])] + 1;
            draws[k] = start + depth[k] + 1;
        }
        var out: [16]u32 = undefined;
        var outs = [_][]u32{out[0..tree.n]};
        try ta.window(&.{.{ .slot = 0, .start = start, .tokens = tree.tokens[0..tree.n], .parents = tree.parents[0..tree.n], .draws = draws[0..tree.n], .sampling = smp }}, &outs);
        // the accepted path: from row 0, the child whose token is its parent's choice
        var path: [16]u32 = undefined;
        path[0] = 0;
        var m: usize = 1;
        walk: while (true) {
            for (1..tree.n) |k| if (tree.parents[k] == path[m - 1] and tree.tokens[k] == out[path[m - 1]]) {
                path[m] = @intCast(k);
                m += 1;
                continue :walk;
            };
            break;
        }
        for (path[0..m], 0..) |row, j| try testing.expectEqual(want[i + 1 + j], out[row]);
        try ta.keep(0, path[0..m]);
        // the accepted path as a chain on `b`: the same choices and the same taps, bit for bit
        var bout: [16]u32 = undefined;
        var bouts = [_][]u32{bout[0..m]};
        try tb.window(&.{.{ .slot = 0, .start = start, .tokens = want[i..][0..m], .parents = null, .draws = draws[0..m], .sampling = smp }}, &bouts);
        try testing.expectEqualSlices(u32, want[i + 1 ..][0..m], bout[0..m]);
        const tpa = ta.taps(0);
        const tpb = tb.taps(0);
        const map = tpa.map orelse return error.NoTreeMap;
        for (path[0..m], 0..) |row, j| try testing.expectEqualSlices(u16, tapsRow(tpb, @intCast(j)), tapsRow(tpa, map[row]));
        i += m;
    }
    ta.release(0);
    tb.release(0);
    return .{ .rounds = r, .reply = try gpa.dupe(u32, want[0..i]) };
}

test "dsv41 metal tree windows: shadow-slot branches == serial choices, the kept path == its chain (taps, state)" {
    const a = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false, .cache_bytes = 1 << 30 });
    defer fs.close(a);
    const b = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    defer fs.close(b);
    const p = try ff.ids(gpa, &fs.config(), 130, 40);
    defer gpa.free(p);
    const hot: host.lanes.Sampling = .{ .seed = 77, .temperature = 0.8, .top_k = 0, .top_p = 1.0 };
    const g = try run(a, b, p, 70, null);
    defer gpa.free(g.reply);
    const s = try run(a, b, p, 70, hot);
    defer gpa.free(s.reply);
    const st = a.t.tree.stats;
    // a later turn from the state the trees left (prompt cache) decodes as a fresh slot
    const more = try ff.ids(gpa, &fs.config(), 20, 41);
    defer gpa.free(more);
    const turn = try std.mem.concat(gpa, u32, &.{ p, s.reply, more });
    defer gpa.free(turn);
    const hits0 = a.t.stats.hits;
    const cached = try a.t.serial(0, turn, 24, null);
    defer gpa.free(cached);
    const fresh = try b.t.serial(0, turn, 24, null);
    defer gpa.free(fresh);
    const ok = std.mem.eql(u32, cached, fresh) and a.t.stats.hits > hits0 and a.t.stats.cached_rows >= p.len + s.reply.len;
    std.debug.print("{s} dsv41 metal tree windows: {d} + {d} rounds, {d} + {d} tokens; {d} trees, {d} swaps, {d} binds, {d} shadow rows, {d} rows synced, {d} launches; cached turn after trees {s} fresh\n", .{ if (ok) "PASS" else "FAIL", g.rounds, s.rounds, g.reply.len, s.reply.len, st.trees, st.swaps, st.binds, st.shadow_rows, st.synced_rows, st.launches, if (ok) "==" else "!=" });
    try testing.expect(st.swaps >= 2 * (g.rounds / 5) and st.binds == 2 * 2);
    try testing.expect(ok);
}
