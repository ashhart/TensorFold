//! Exactness across the cluster on the toy model: drafted == plain, shared == solo, and the same bits at 1, 2 and 4 nodes.
const std = @import("std");
const lanes = @import("lanes");
const exchange = @import("exchange.zig");
const round = @import("round.zig");
const toy = @import("toy.zig");
const backend = @import("backend.zig");
const toy_cluster = @import("toy_cluster.zig");

const Cluster = toy_cluster.Cluster;
const Variant = toy_cluster.Variant;

const gpa = std.testing.allocator;
const Sampling = lanes.Sampling;

/// A draft head that knows the model: one node's own decode, wrong every fifth position so rollbacks happen.
const Oracle = struct {
    w: *const toy.Weights,

    fn drafter(o: *Oracle) backend.Drafter {
        return .{ .ptr = o, .guess = guess };
    }

    fn guess(ptr: *anyopaque, history: []const u32, s: ?Sampling, position: u64, out: []u32) anyerror!void {
        const o: *Oracle = @ptrCast(@alignCast(ptr));
        const toks = try decode(o.w, history, s, out.len);
        defer gpa.free(toks);
        for (out, toks, 0..) |*d, t, j| d.* = if ((position + j) % 5 == 0) (t + 1) % o.w.dims.vocab else t;
    }
};

/// One node feeding `history` and drawing `n` tokens one at a time: the model's own decode.
pub fn decode(w: *const toy.Weights, history: []const u32, s: ?Sampling, n: usize) ![]u32 {
    const c = try Cluster.init(gpa, w, 1, .{});
    defer c.deinit();
    const l = &c.leader;
    const rows = try gpa.alloc(round.Row, history.len);
    defer gpa.free(rows);
    for (rows, history, 0..) |*r, t, i| r.* = .{ .slot = 0, .token = t, .index = @intCast(i), .sample = i + 1 == history.len };
    var logits = try l.run(rows);
    const out = try gpa.alloc(u32, n);
    for (out, 0..) |*t, j| {
        t.* = try l.draw(logits, s, history.len + j);
        gpa.free(logits);
        logits = try l.run(&.{.{ .slot = 0, .token = t.*, .index = @intCast(history.len + j), .sample = true }});
    }
    gpa.free(logits);
    return out;
}

fn config() !lanes.Config {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(width)) };
    return lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

const Case = struct { prompt: []const u32, max_new: u32 = 24, sampling: ?Sampling = null, drafts: bool = true };

/// The last serve's leader counters: rounds, and rows those rounds carried.
var last: struct { rounds: u64 = 0, rows: u64 = 0 } = .{};

/// Every case's emitted tokens, admitted together into one engine over an n-node cluster.
fn serve(w: *const toy.Weights, n: u32, cases: []const Case, v: Variant) ![][]u32 {
    var cfg = try config();
    defer cfg.deinit(gpa);
    const c = try Cluster.init(gpa, w, n, v);
    defer c.deinit();
    var oracle: Oracle = .{ .w = w };
    c.leader.drafter = oracle.drafter();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, c.leader.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |k, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = k.prompt, .max_new = k.max_new, .sampling = k.sampling, .drafts = k.drafts });
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    last = .{ .rounds = c.leader.rounds, .rows = c.leader.rows_run };
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 14, 15, 92, 65, 35, 89, 79 };
const p2 = [_]u32{ 27, 18, 28, 18, 28, 45, 90, 45, 23, 53 };
const hot: Sampling = .{ .seed = 77, .temperature = 0.8, .top_k = 0, .top_p = 0.95 };

test "four nodes: drafted rounds commit the model's own decode, greedy and sampled" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 1);
    for ([_]?Sampling{ null, hot }) |s| {
        const drafted = try serve(&w, 4, &.{.{ .prompt = &p1, .sampling = s }}, .{});
        defer free(drafted);
        try std.testing.expect(last.rounds < drafted[0].len and last.rows > last.rounds + drafted[0].len);
        const plain = try serve(&w, 4, &.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .{});
        defer free(plain);
        const own = try decode(&w, &p1, s, drafted[0].len);
        defer gpa.free(own);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        try std.testing.expectEqualSlices(u32, own, drafted[0]);
    }
}

test "one, two and four nodes emit the same tokens; two streams sharing rounds commit what each commits alone" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 2);
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 20 } };
    const four = try serve(&w, 4, &cases, .{});
    defer free(four);
    for ([_]u32{ 1, 2 }) |n| {
        const fewer = try serve(&w, n, &cases, .{});
        defer free(fewer);
        for (four, fewer) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    }
    const one = try serve(&w, 4, cases[0..1], .{});
    defer free(one);
    const two = try serve(&w, 4, cases[1..2], .{});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], four[0]);
    try std.testing.expectEqualSlices(u32, two[0], four[1]);
}

/// Slot 0's next row's logits: alone, or in a round with slot 1's window and slot 2's prompt chunk; at `n` nodes.
fn rowBits(w: *const toy.Weights, n: u32, mixed: bool, v: Variant) ![]u32 {
    const c = try Cluster.init(gpa, w, n, v);
    defer c.deinit();
    const l = &c.leader;
    for ([_][]const u32{ &p1, &p2 }, 0..) |p, slot| {
        const rows = try gpa.alloc(round.Row, p.len);
        defer gpa.free(rows);
        for (rows, p, 0..) |*r, t, i| r.* = .{ .slot = @intCast(slot), .token = t, .index = @intCast(i), .sample = false };
        gpa.free(try l.run(rows));
    }
    const target: round.Row = .{ .slot = 0, .token = 41, .index = p1.len, .sample = true };
    const rows: []const round.Row = if (!mixed) &.{target} else &.{
        .{ .slot = 1, .token = 7, .index = p2.len, .sample = true },
        .{ .slot = 1, .token = 8, .index = p2.len + 1, .sample = true },
        target,
        .{ .slot = 2, .token = 5, .index = 0, .sample = false },
        .{ .slot = 2, .token = 6, .index = 1, .sample = false },
        .{ .slot = 2, .token = 9, .index = 2, .sample = true },
    };
    const logits = try l.run(rows);
    defer gpa.free(logits);
    const at: usize = if (mixed) 2 else 0;
    const vocab = w.dims.vocab;
    const out = try gpa.alloc(u32, vocab);
    for (out, logits[at * vocab ..][0..vocab]) |*o, x| o.* = @bitCast(x);
    return out;
}

test "a row's logits are the same bits alone, beside other streams' rows and a prompt chunk, and at 1 or 4 nodes" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 3);
    const solo = try rowBits(&w, 4, false, .{});
    defer gpa.free(solo);
    for ([_]struct { u32, bool }{ .{ 4, true }, .{ 1, false }, .{ 1, true }, .{ 2, true } }) |case| {
        const got = try rowBits(&w, case[0], case[1], .{});
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, solo, got);
    }
    const naive_one = try rowBits(&w, 1, false, .{ .naive = true });
    defer gpa.free(naive_one);
    const naive_four = try rowBits(&w, 4, false, .{ .naive = true });
    defer gpa.free(naive_four);
    try std.testing.expect(!std.mem.eql(u32, naive_one, naive_four));
}

test "the same rounds over the fabric's one-sided channel: four ranks emit what the memory exchange's four emit" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 4);
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 16 } };
    const mem = try serve(&w, 4, &cases, .{});
    defer free(mem);
    const fab = try serve(&w, 4, &cases, .{ .fabric = true });
    defer free(fab);
    for (mem, fab) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    const bits = try rowBits(&w, 4, true, .{ .fabric = true });
    defer gpa.free(bits);
    const solo = try rowBits(&w, 1, false, .{});
    defer gpa.free(solo);
    try std.testing.expectEqualSlices(u32, solo, bits);
}

test "by streams, split sums, Thunderbolt's two-sided links: the same bits at 1-4 nodes, the same tokens at 4" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 5);
    const solo = try rowBits(&w, 1, false, .{});
    defer gpa.free(solo);
    const variants = [_]Variant{ .{ .by_streams = true }, .{ .split_sums = true }, .{ .by_streams = true, .split_sums = true }, .{ .by_streams = true, .fabric = true }, .{ .two_sided = true }, .{ .two_sided = true, .by_streams = true, .split_sums = true } };
    for (variants) |v| for ([_]u32{ 1, 2, 3, 4 }) |n| {
        const got = try rowBits(&w, n, true, v);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, solo, got);
    };
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 16 } };
    const plain = try serve(&w, 4, &cases, .{});
    defer free(plain);
    const dp = try serve(&w, 4, &cases, .{ .by_streams = true, .split_sums = true });
    defer free(dp);
    for (plain, dp) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    const tb = try serve(&w, 4, &cases, .{ .two_sided = true, .by_streams = true, .split_sums = true });
    defer free(tb);
    for (plain, tb) |x, y| try std.testing.expectEqualSlices(u32, x, y);
}

test "every expert split by its intermediate columns: the same bits at 1, 2, 3 and 4 nodes, and exact rounds" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 8);
    const one = try rowBits(&w, 1, false, .{ .split_experts = true });
    defer gpa.free(one);
    for ([_]Variant{ .{ .split_experts = true }, .{ .split_experts = true, .split_sums = true, .two_sided = true } }) |v| for ([_]u32{ 2, 3, 4 }) |n| {
        const got = try rowBits(&w, n, true, v);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, one, got);
    };
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 16 } };
    const four = try serve(&w, 4, &cases, .{ .split_experts = true, .two_sided = true });
    defer free(four);
    const solo = try serve(&w, 4, cases[0..1], .{ .split_experts = true });
    defer free(solo);
    const plain = try serve(&w, 2, &.{.{ .prompt = &p1, .drafts = false }}, .{ .split_experts = true });
    defer free(plain);
    try std.testing.expectEqualSlices(u32, solo[0], four[0]);
    try std.testing.expectEqualSlices(u32, plain[0], four[0]);
}
