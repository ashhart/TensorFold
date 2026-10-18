//! branches.zig on the CPU twin: chain-only tree windows match the native tree window, and lanes over both agree.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const tree = @import("tree.zig");
const branches = @import("branches.zig");
const oracle = @import("branches_oracle.zig");
const costs_mod = @import("costs.zig");
const accept = lanes.accept;

const gpa = std.testing.allocator;

/// The twin as a chain-only target (GpuTarget's shape): tree windows go through a Resolver, keeps through its map.
const ChainTwin = struct {
    t: *twin.Twin,
    res: [4]branches.Resolver = @splat(.{}),
    native_trees: u64 = 0, // tree windows the twin ran itself (must stay 0)

    fn target(c: *ChainTwin) iface.Target {
        return .{ .ptr = c, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release } };
    }

    fn chains(c: *ChainTwin) branches.Chains {
        return .{ .ptr = c, .vtable = &.{ .run = run, .drop = drop } };
    }

    fn self(p: *anyopaque) *ChainTwin {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, s: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const c = self(p);
        c.res[slot].clear();
        return c.t.target().prefill(slot, ids, s, draw);
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const c = self(p);
        for (segments, choices) |g, out| {
            if (g.parents != null) {
                try c.res[g.slot].window(c.chains(), g, out);
                continue;
            }
            c.res[g.slot].clear();
            var one = [_][]u32{out};
            try c.t.target().window(&.{g}, &one);
        }
    }

    fn run(p: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, s: ?lanes.Sampling, out: []u32) anyerror!void {
        const c = self(p);
        var one = [_][]u32{out};
        try c.t.target().window(&.{.{ .slot = slot, .start = start, .tokens = tokens, .parents = null, .draws = draws, .sampling = s }}, &one);
    }

    fn drop(p: *anyopaque, slot: u32) anyerror!void {
        self(p).t.drop(slot);
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const c = self(p);
        if (!c.res[slot].live) return c.t.target().keep(slot, path);
        const k = try c.res[slot].keep(path);
        var rows: [branches.max_rows]u32 = undefined;
        for (rows[0 .. k + 1], 0..) |*r, i| r.* = @intCast(i);
        try c.t.target().keep(slot, rows[0 .. k + 1]);
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        const c = self(p);
        var t = c.t.target().taps(slot);
        t.map = c.res[slot].map();
        return t;
    }

    fn release(p: *anyopaque, slot: u32) void {
        const c = self(p);
        c.res[slot].clear();
        c.t.target().release(slot);
    }
};

const Stats = struct { out: []u32, drafted: u64, accepted: u64, trees: u64, sib_wins: u64, chains: u64 = 0 };

/// One stream through lanes with the twin's DSpark and first-position siblings, over the native or chain-only target.
fn runLanes(prompt: []const u32, max_new: u32, smp: ?lanes.Sampling, set: tree.Settings, chain_only: bool) !Stats {
    const dims: twin.Dims = .{};
    const t = try twin.Twin.init(gpa, dims, 2);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, 2);
    defer d.deinit();
    var ct: ChainTwin = .{ .t = t };
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, if (chain_only) ct.target() else t.target(), d.pass(), c, .{ .shape = d.shape, .siblings = set.siblings, .sib_rows = set.sib_rows, .dup = set.dup, .slots = 1 });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = max_new, .sampling = smp });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    return .{ .out = try gpa.dupe(u32, s.emitted()), .drafted = e.drafted, .accepted = e.accepted, .trees = x.policy.trees, .sib_wins = x.policy.sib_wins, .chains = ct.res[0].stats.chains };
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };
const p3 = [_]u32{ 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30 };
const hot: lanes.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };

test "trees resolved as chains: the native tree window's rounds, the serial decode" {
    var trees: u64 = 0;
    var wins: u64 = 0;
    const sets = [_]tree.Settings{ .{ .siblings = 2, .dup = .{} }, .{ .siblings = 2 }, .{ .siblings = 3, .sib_rows = 6 } };
    for ([_][]const u32{ &p1, &p2, &p3 }) |prompt| for (sets) |set| for ([_]?lanes.Sampling{ null, hot }) |smp| {
        const native = try runLanes(prompt, 70, smp, set, false);
        defer gpa.free(native.out);
        const mine = try runLanes(prompt, 70, smp, set, true);
        defer gpa.free(mine.out);
        const serial = blk: {
            const t = try twin.Twin.init(gpa, .{}, 1);
            defer t.deinit();
            break :blk try t.serial(0, prompt, 70, smp, null);
        };
        defer gpa.free(serial);
        try std.testing.expectEqualSlices(u32, serial, mine.out);
        try std.testing.expectEqualSlices(u32, native.out, mine.out);
        try std.testing.expectEqual(native.drafted, mine.drafted);
        try std.testing.expectEqual(native.accepted, mine.accepted);
        try std.testing.expectEqual(native.trees, mine.trees);
        try std.testing.expectEqual(native.sib_wins, mine.sib_wins);
        trees += mine.trees;
        wins += mine.sib_wins;
    };
    try std.testing.expect(trees > 0 and wins > 0);
}

test "the tree oracle's siblings lose and win through a chain-only target, as through tree windows" {
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const dims: twin.Dims = .{};
    const want = blk: {
        const t = try twin.Twin.init(gpa, dims, 1);
        defer t.deinit();
        break :blk try t.serial(0, &prompt, 90, null, null);
    };
    defer gpa.free(want);
    const shape: @import("dspark.zig").Shape = .{ .block = dims.block, .window = dims.window, .hidden = dims.dim, .candidates = 8 };
    var got: [2]oracle.Result = undefined;
    for ([_]bool{ false, true }, &got) |chain_only, *g| {
        const t = try twin.Twin.init(gpa, dims, 2);
        defer t.deinit();
        var ct: ChainTwin = .{ .t = t };
        var o = try oracle.TreeOracle.init(gpa, want, prompt.len, dims.vocab);
        defer o.deinit(gpa);
        g.* = try oracle.generate(gpa, if (chain_only) ct.target() else t.target(), o.pass(), shape, &prompt, 90, .{ .siblings = 2 });
        try std.testing.expectEqualSlices(u32, want, g.tokens);
        try std.testing.expectEqual(@as(u64, 0), o.off_anchor);
        try std.testing.expect(g.trees > 0 and g.sib_wins > 0 and g.sib_wins < g.trees);
    }
    defer for (got) |g| gpa.free(g.tokens);
    try std.testing.expectEqual(got[0].drafted, got[1].drafted);
    try std.testing.expectEqual(got[0].accepted, got[1].accepted);
    try std.testing.expectEqual(got[0].trees, got[1].trees);
    try std.testing.expectEqual(got[0].sib_wins, got[1].sib_wins);
    try std.testing.expectEqual(got[0].rounds, got[1].rounds);
}

test "any tree shape: the resolver's accepted path and its choices equal the tree window's" {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const t = try twin.Twin.init(gpa, .{}, 2);
    defer t.deinit();
    var ct: ChainTwin = .{ .t = t };
    const prompt = [_]u32{ 5, 9, 2, 6, 5, 3, 5, 8 };
    _ = try t.target().prefill(0, &prompt, null, prompt.len);
    var start: u64 = prompt.len;
    for (0..200) |_| {
        const n = r.intRangeAtMost(usize, 2, 14);
        var parents: [16]i32 = undefined;
        var depth: [16]u64 = undefined;
        var tokens: [16]u32 = undefined;
        var draws: [16]u64 = undefined;
        var native: [16]u32 = undefined;
        parents[0] = -1;
        depth[0] = 0;
        tokens[0] = r.uintLessThan(u32, t.dims.vocab);
        for (1..n) |row| {
            parents[row] = @intCast(r.uintLessThan(usize, row));
            depth[row] = depth[@intCast(parents[row])] + 1;
        }
        for (0..n) |row| draws[row] = start + depth[row] + 1;
        // tokens row by row: often the parent's choice (from the tree window over the rows so far), so paths branch
        for (1..n) |row| {
            if (r.float(f32) < 0.65) {
                try nativeWindow(t, start, tokens[0..row], parents[0..row], draws[0..row], native[0..row]);
                tokens[row] = native[@intCast(parents[row])];
            } else tokens[row] = r.uintLessThan(u32, t.dims.vocab);
        }
        try nativeWindow(t, start, tokens[0..n], parents[0..n], draws[0..n], native[0..n]);
        const want = try accept.acceptPath(gpa, tokens[0..n], parents[0..n], native[0..n]);
        defer gpa.free(want);
        var mine: [16]u32 = undefined;
        var outs = [_][]u32{mine[0..n]};
        try ct.target().window(&.{.{ .slot = 0, .start = start, .tokens = tokens[0..n], .parents = parents[0..n], .draws = draws[0..n], .sampling = null }}, &outs);
        const path = try accept.acceptPath(gpa, tokens[0..n], parents[0..n], mine[0..n]);
        defer gpa.free(path);
        try std.testing.expectEqualSlices(u32, want, path);
        for (path) |row| try std.testing.expectEqual(native[row], mine[row]);
        // the path kept: the slot moves on by its rows, the taps map them to the chain's first rows
        try ct.target().keep(0, path);
        const m = ct.target().taps(0).map.?;
        for (path, 0..) |row, i| try std.testing.expectEqual(@as(u32, @intCast(i)), m[row]);
        start += path.len;
    }
    try std.testing.expect(ct.res[0].stats.deep > 0); // branches below row 0 re-ran their path's rows
}

/// The twin's own tree window over the rows, dropped after (the slot unchanged).
fn nativeWindow(t: *twin.Twin, start: u64, tokens: []const u32, parents: []const i32, draws: []const u64, out: []u32) !void {
    var outs = [_][]u32{out};
    try t.target().window(&.{.{ .slot = 0, .start = start, .tokens = tokens, .parents = parents, .draws = draws, .sampling = null }}, &outs);
    t.drop(0);
}
