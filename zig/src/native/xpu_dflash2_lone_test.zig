//! A lone DFlash2 stream's tree width from the lane rule's lone pricing, and what no width can change: the tokens.
const std = @import("std");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const ad = @import("xpu_dflash2.zig");
const Drafter = xq.drafter.Drafter;
const gpa = std.testing.allocator;
const lc = lanes.lone_costs;

fn engine(cfg: *lanes.Config, target: *lanes.fake.Fake, clock: *lanes.fake.FixedClock) !lanes.Engine {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = 25 + 0.6 * @as(f64, @floatFromInt(rows)) };
    cfg.* = try lanes.Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 15, .plain_guard = false, .hidden_rows = true, .batch_rows = 16, .window_costs = &costs, .lone_cost_tile = ad.lone_tile }, 16, 15);
    target.* = .{ .gpa = gpa };
    clock.* = .{};
    return lanes.Engine.init(gpa, cfg, target.backend(), clock.clock());
}

/// A world's round ms by drafts: a verify costs `per` more a node, over `base`.
const World = struct { base: f64, per: f64 };

/// A lone stream of `context` tokens for `rounds` rounds in `world`: the nodes of its first round and of its last.
fn lone(e: *lanes.Engine, context: usize, world: World, rounds: usize, nodes: u32) !struct { first: u32, last: u32 } {
    const n = ad.priced(nodes) orelse return .{ .first = nodes, .last = nodes };
    const prompt = try gpa.alloc(u32, context);
    defer gpa.free(prompt);
    @memset(prompt, 3);
    var s = try lanes.Stream.init(gpa, .{ .id = "lone-width", .prompt = prompt, .max_new = 256 });
    defer s.deinit(gpa);
    var parents: [15]i32 = undefined;
    var own: [15]f64 = undefined;
    for (&parents, &own, 0..) |*p, *o, i| {
        p.* = @as(i32, @intCast(i)) - 1; // a chain of drafts each 85 % likely under its parent
        o.* = @log(0.85);
    }
    var landing: lanes.draft_tree.Landing = .{};
    var chance: [15]f64 = undefined;
    ad.chances(&landing, false, parents[0..n], own[0..n], chance[0..n]);
    var first: u32 = 0;
    var last: u32 = 0;
    for (0..rounds) |i| {
        const pick = try ad.width(e, &s, chance[0..n]);
        if (i == 0) first = pick.nodes;
        last = pick.nodes;
        s.rounds += 1;
        if (s.rounds > 1) try ad.measured(e, &s, pick.trial, pick.nodes, world.base + world.per * @as(f64, @floatFromInt(pick.nodes)));
    }
    lc.keep(e.gpa, lc.slot(&e.lone_kept, &s), &s) catch {};
    return .{ .first = first, .last = last };
}

/// The drafts maximizing the tokens a ms in `world`, the chain of 85 % drafts `lone` grows, at most `cap`.
fn best(world: World, cap: u32) u32 {
    var top: u32 = 1;
    var top_rate: f64 = 0;
    var tokens: f64 = 1;
    var chance: f64 = 1;
    for (1..cap + 1) |n| {
        chance *= 0.85;
        tokens += chance;
        const rate = tokens / (world.base + world.per * @as(f64, @floatFromInt(n)));
        if (rate > top_rate) {
            top = @intCast(n);
            top_rate = rate;
        }
    }
    return top;
}

test "draft sources: a lone tree's width follows its context band's measured rounds" {
    var cfg: lanes.Config = undefined;
    var target: lanes.fake.Fake = undefined;
    var clock: lanes.fake.FixedClock = undefined;
    var e = try engine(&cfg, &target, &clock);
    defer e.deinit();
    defer cfg.deinit(gpa);
    defer target.deinit();
    const shallow: World = .{ .base = 30, .per = 1 };
    const deep: World = .{ .base = 70, .per = 21.5 };
    const long = 131072;
    const wide = try lone(&e, 100, shallow, 120, 15);
    const narrow = try lone(&e, long, deep, 120, 15);
    try std.testing.expectEqual(best(shallow, 15), wide.last);
    try std.testing.expectEqual(best(deep, 15), narrow.last);
    // each band's next stream starts on what its band measured: the deep one narrow, the short one still wide
    try std.testing.expectEqual(narrow.last, (try lone(&e, long + 9, deep, 1, 15)).first);
    try std.testing.expectEqual(wide.last, (try lone(&e, 120, shallow, 1, 15)).first);
}

test "draft sources: XPU_TREE caps a lone tree, and under 2 nodes nothing is priced" {
    var cfg: lanes.Config = undefined;
    var target: lanes.fake.Fake = undefined;
    var clock: lanes.fake.FixedClock = undefined;
    var e = try engine(&cfg, &target, &clock);
    defer e.deinit();
    defer cfg.deinit(gpa);
    defer target.deinit();
    try std.testing.expectEqual(@as(?u32, 15), ad.priced(40));
    try std.testing.expectEqual(@as(?u32, 3), ad.priced(3));
    try std.testing.expectEqual(@as(?u32, null), ad.priced(1));
    try std.testing.expectEqual(@as(?u32, null), ad.priced(0));
    for ([_]u32{ 2, 3, 7, 40 }) |cap| {
        const r = try lone(&e, 100, .{ .base = 30, .per = 1 }, 60, cap);
        try std.testing.expect(r.last <= ad.priced(cap).?);
        try std.testing.expect(r.last >= 1); // never a round with no draft
    }
    try std.testing.expectEqual(@as(u32, 1), (try lone(&e, 100, .{ .base = 30, .per = 1 }, 5, 1)).last); // fixed, as before
}

fn mix(h: u64, x: u64) u64 {
    return (h ^ x) *% 0x9E3779B97F4A7C15 +% 0x632BE59BD9B4E019;
}

// The tree walk over a synthetic target: a state hash a row, a greedy or keyed pick, the lattice the context's.
const V = 40;
const Sim = struct {
    dr: *Drafter,
    seed: u64,
    sampled: bool,
    rows: [16]u64 = undefined, // each verified row's target state
    pos0: u64 = 0, // the position of the first pending token

    fn pick(s: *const Sim, state: u64, position: u64) u32 {
        const h = if (s.sampled) mix(mix(state, position), s.seed) else mix(state, 7);
        return @intCast((h >> 33) % V);
    }
    fn row(ctx: *anyopaque, r: u32, position: u64) anyerror!u32 {
        const s: *Sim = @ptrCast(@alignCast(ctx));
        return s.pick(s.rows[r], position);
    }
    fn gumbel(ctx: *anyopaque, position: u64, id: u64) f64 {
        const s: *Sim = @ptrCast(@alignCast(ctx));
        const u = (@as(f64, @floatFromInt(mix(mix(s.seed, position), id) >> 11)) + 0.5) / (1 << 53);
        return -@log(-@log(u));
    }
};

const Out = struct { tokens: [16]u32 = undefined, n: usize = 0 };

/// One round from the state `s0` after `pending`, `committed` tokens in: the context's lattice, `nodes` wide.
fn round(sim: *Sim, s0: u64, pending: u32, committed: u64, oracle: []const u32, nodes: u32) !Out {
    const dr = sim.dr;
    const depths = 7;
    for (0..depths) |d| for (0..16) |j| {
        const k = mix(mix(committed, d), j);
        const hit = j == 0 and (k >> 20) % 10 < 7 and committed + d < oracle.len;
        dr.cand[d][j] = if (hit) oracle[committed + d] else @intCast((k >> 33) % V);
        dr.unary[d][j] = 1.5 - 0.3 * @as(f64, @floatFromInt(j)) + @as(f64, @floatFromInt((k >> 40) % 100)) / 400;
    };
    const p0 = sim.pos0 + committed;
    const draw: ?xq.drafter.Draw = if (sim.sampled) .{ .ctx = sim, .row = Sim.row, .gumbel = Sim.gumbel, .temperature = 1.0 } else null;
    const full = xq.drafter.grown(dr, pending, depths, 15, draw, p0);
    const g = xq.drafter.grown(dr, pending, depths, nodes, draw, p0);
    // a tree of `nodes` is the first `nodes` drafts of the 15-node tree, in pop order
    try std.testing.expectEqual(@min(nodes, full.n), g.n);
    try std.testing.expectEqualSlices(u32, full.ids[0 .. g.n + 1], g.ids[0 .. g.n + 1]);
    try std.testing.expectEqualSlices(i32, full.par[1 .. g.n + 1], g.par[1 .. g.n + 1]);
    try std.testing.expectEqualSlices(f64, full.own[1 .. g.n + 1], g.own[1 .. g.n + 1]);
    var rows: [16]i32 = undefined;
    xq.drafter.treeRows(&g, g.n, &rows);
    sim.rows[0] = s0;
    var greedy: [16]u32 = undefined;
    greedy[0] = sim.pick(s0, p0 + 1);
    for (1..g.n + 1) |i| {
        sim.rows[i] = mix(sim.rows[@intCast(rows[i])], g.ids[i]);
        greedy[i] = sim.pick(sim.rows[i], p0 + 1);
    }
    var path_out: [16]u32 = undefined;
    const round_ = try xq.drafter.follow(&g, nodes, greedy[0 .. g.n + 1], draw, &path_out);
    var out: Out = .{};
    @memcpy(out.tokens[0..round_.tokens.len - 1], round_.tokens[1..]);
    out.n = round_.tokens.len - 1;
    out.tokens[out.n] = round_.terminal;
    out.n += 1;
    return out;
}

fn newSim(sampled: bool) !*Sim {
    const dr = try gpa.create(Drafter);
    const pred = try gpa.alloc(u8, 64 * 256 * 2);
    const succ = try gpa.alloc(u8, 64 * 256 * 2);
    var h: u64 = 99;
    for ([_][]u8{ pred, succ }) |t| for (0..t.len / 2) |i| {
        h = mix(h, i);
        std.mem.writeInt(u16, t[i * 2 ..][0..2], 0x3c00 | @as(u16, @truncate(h >> 50)), .little); // small positive bf16
    };
    dr.pred = pred;
    dr.succ = succ;
    dr.depths = 7;
    for (&dr.sp, 0..) |*x, i| x.* = @floatCast(@as(f64, @floatFromInt(i % 7)) / 64);
    const sim = try gpa.create(Sim);
    sim.* = .{ .dr = dr, .seed = 4242, .sampled = sampled };
    return sim;
}

fn freeSim(sim: *Sim) void {
    gpa.free(sim.dr.pred);
    gpa.free(sim.dr.succ);
    gpa.destroy(sim.dr);
    gpa.destroy(sim);
}

/// The target's own continuation of `pending`, `len` tokens, one at a time.
fn autoregress(sim: *Sim, pending: u32, len: usize, out: []u32) void {
    var st = mix(1, pending);
    for (out[0..len], 0..) |*t, k| {
        t.* = sim.pick(st, if (sim.sampled) sim.pos0 + k + 1 else 0);
        st = mix(st, t.*);
    }
}

/// `schedule[i % len]` nodes in round i, until `want` tokens: what the stream emitted.
fn generate(sim: *Sim, oracle: []const u32, schedule: []const u32, want: usize, out: []u32) !void {
    var committed: u64 = 0;
    var st = mix(1, 5);
    var pending: u32 = 5;
    var i: usize = 0;
    while (committed < want) : (i += 1) {
        const r = try round(sim, st, pending, committed, oracle, schedule[i % schedule.len]);
        for (r.tokens[0..r.n]) |t| {
            if (committed < want) out[committed] = t;
            committed += 1;
            st = mix(st, t);
        }
        pending = r.tokens[r.n - 1];
    }
}

test "draft sources: a lone tree's width never changes a token, greedy or keyed" {
    for ([_]bool{ false, true }) |sampled| {
        const sim = try newSim(sampled);
        defer freeSim(sim);
        var oracle: [160]u32 = undefined;
        autoregress(sim, 5, oracle.len, &oracle);
        var got: [120]u32 = undefined;
        for (1..16) |nodes| {
            try generate(sim, &oracle, &.{@intCast(nodes)}, got.len, &got);
            try std.testing.expectEqualSlices(u32, oracle[0..got.len], &got);
        }
        try generate(sim, &oracle, &.{ 15, 3, 7, 1, 15, 2, 11, 3 }, got.len, &got);
        try std.testing.expectEqualSlices(u32, oracle[0..got.len], &got);
    }
}
