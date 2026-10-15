//! The link's join and startup steps over loopback TCP, every rank a thread: what each rank hears.

const std = @import("std");
const link = @import("link.zig");
const rccl = @import("rccl.zig");
const agree = @import("../agree.zig");
const Policy = @import("../policy.zig").Policy;
const collective = @import("../collective.zig");
const Startup = @import("../startup.zig").Startup;
const loopback = @import("loopback.zig");
const hello = loopback.hello;

/// A rank of a test group: what it says at join, and what it got.
const Rank = struct {
    hello: link.Hello,
    port: u16,
    got: ?link.Link = null,
    err: ?link.Error = null,
    fault: agree.Fault = .{},
    words: [Policy.word_count]u32 = @splat(0),
    /// After joining: the startup steps it takes, and their verdicts; `leave` closes its link instead.
    steps: []const agree.Step = &.{},
    verdicts: [4]agree.Verdict = undefined,
    taken: usize = 0,
    leave: bool = false,

    fn run(r: *Rank) void {
        const id: rccl.UniqueId = .{ .internal = @splat(7) };
        const policy: [Policy.word_count]u32 = @splat(@intCast(100 + r.hello.rank));
        const out = link.Link.open(std.testing.io, "127.0.0.1", r.port, id, r.hello, policy, &r.fault) catch |err| {
            r.err = err;
            return;
        };
        r.got = out[0];
        r.words = out[2];
        if (r.leave) {
            r.got.?.close();
            r.got = null;
            return;
        }
        for (r.steps) |s| {
            r.verdicts[r.taken] = r.got.?.agree(s, .least) catch |err| {
                r.err = err;
                return;
            };
            r.taken += 1;
        }
    }
};

/// Runs every rank of `ranks` on its own thread, rank 0 first, and waits for all.
fn group(ranks: []Rank) !void {
    loopback.ignorePipe();
    const port = try loopback.freePort();
    var threads: [4]std.Thread = undefined;
    for (ranks, 0..) |*r, i| {
        r.port = port;
        threads[i] = try std.Thread.spawn(.{}, Rank.run, .{r});
    }
    for (threads[0..ranks.len]) |t| t.join();
    for (ranks) |*r| if (r.got) |*l| l.close();
}

test "ranks that match join, and every follower runs rank 0's policy words" {
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0 },
        .{ .hello = hello(1, 3), .port = 0 },
        .{ .hello = hello(2, 3), .port = 0 },
    };
    try group(&ranks);
    for (ranks) |r| try std.testing.expect(r.err == null);
    for (ranks) |r| try std.testing.expectEqual(@as(u32, 100), r.words[0]);
}

test "a rank started with another world size is refused on every rank before RCCL starts" {
    var ranks = [_]Rank{ .{ .hello = hello(0, 2), .port = 0 }, .{ .hello = hello(1, 4), .port = 0 } };
    try group(&ranks);
    for (ranks) |r| {
        try std.testing.expectEqual(@as(?link.Error, error.RanksDiffer), r.err);
        try std.testing.expectEqual(agree.Fault{ .field = .world, .rank = 1 }, r.fault);
    }
}

test "a rank with other weights is refused, and so is the rank that matched before it" {
    var other = hello(2, 3);
    other.ident.weights[31] = 9;
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0 },
        .{ .hello = hello(1, 3), .port = 0 },
        .{ .hello = other, .port = 0 },
    };
    try group(&ranks);
    for (ranks) |r| try std.testing.expectEqual(@as(?link.Error, error.RanksDiffer), r.err);
    try std.testing.expectEqual(agree.Fault{ .field = .weights, .rank = 2 }, ranks[0].fault);
    try std.testing.expectEqual(ranks[0].fault, ranks[2].fault);
}

test "hellos differ field by field in a fixed order" {
    const lead = hello(0, 2);
    var taken = [_]bool{ true, false };
    try std.testing.expect(link.differs(lead, hello(1, 2), &taken) == null);
    var h = hello(1, 2);
    h.gpu = 12;
    try std.testing.expectEqual(agree.Field.gpu, link.differs(lead, h, &taken).?);
    h.world = 4;
    try std.testing.expectEqual(agree.Field.world, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.ident.config[0] = 0;
    try std.testing.expectEqual(agree.Field.config, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.rccl = 22100;
    try std.testing.expectEqual(agree.Field.rccl, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.ident.program[5] = 0;
    try std.testing.expectEqual(agree.Field.program, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.layout = 1;
    try std.testing.expectEqual(agree.Field.layout, link.differs(lead, h, &taken).?);
    taken[1] = true;
    try std.testing.expectEqual(agree.Field.rank, link.differs(lead, hello(1, 2), &taken).?);
}

test "startup steps: the least of each value on every rank, then one rank's refusal on every rank" {
    const ok0 = [_]agree.Step{ .ok(.{ 8, 30 }), .ok(.{ 1, 1 }) };
    const ok1 = [_]agree.Step{ .ok(.{ 6, 40 }), .ok(.{ 1, 1 }) };
    const no2 = [_]agree.Step{ .ok(.{ 7, 20 }), .refuse("--parallel 4 does not fit") };
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0, .steps = &ok0 },
        .{ .hello = hello(1, 3), .port = 0, .steps = &ok1 },
        .{ .hello = hello(2, 3), .port = 0, .steps = &no2 },
    };
    try group(&ranks);
    for (ranks) |r| {
        try std.testing.expect(r.err == null);
        try std.testing.expectEqual([2]u64{ 6, 20 }, r.verdicts[0].values);
        try std.testing.expectEqual(@as(u32, 1), r.verdicts[1].refused);
        try std.testing.expectEqual(@as(u32, 2), r.verdicts[1].rank);
        try std.testing.expectEqualStrings("--parallel 4 does not fit", r.verdicts[1].reason());
    }
}

test "a rank that leaves before a step is that step's refusal on every rank that stays" {
    const one = [_]agree.Step{.ok(.{ 1, 1 })};
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0, .steps = &one },
        .{ .hello = hello(1, 3), .port = 0, .steps = &one },
        .{ .hello = hello(2, 3), .port = 0, .leave = true },
    };
    try group(&ranks);
    for (ranks[0..2]) |r| {
        try std.testing.expect(r.err == null);
        try std.testing.expectEqual(@as(u32, 1), r.verdicts[0].refused);
        try std.testing.expectEqual(@as(u32, 2), r.verdicts[0].rank);
        try std.testing.expectEqualStrings("its link to rank 0 closed", r.verdicts[0].reason());
    }
}

/// A family's startup on each rank: rank 0's window, admission (`refuse`: the rank that cannot), the streams that fit.
const Families = struct {
    refuse: ?usize = null,
    window: [3]u64 = @splat(0),
    fit: [3]u64 = @splat(0),
    problem: [3][]const u8 = @splat(""),
    text: [3][256]u8 = undefined,

    fn start(f: *Families, rank: usize, l: collective.Link) void {
        var fba: std.heap.FixedBufferAllocator = .init(&f.text[rank]);
        var problem: []const u8 = "";
        f.steps(fba.allocator(), rank, l, &problem) catch {};
        f.problem[rank] = problem;
    }

    fn steps(f: *Families, a: std.mem.Allocator, rank: usize, l: collective.Link, problem: *[]const u8) !void {
        var s: Startup = .{ .link = l };
        f.window[rank] = (try s.together(a, .{ 4096 + rank, 0 }, .first, problem))[0];
        const why = "rank {d} cannot hold a {d}-token window";
        if (f.refuse == rank) return s.refuseAll(a, problem, why, .{ rank, f.window[rank] });
        _ = try s.together(a, .{ 0, 0 }, .first, problem);
        f.fit[rank] = (try s.together(a, .{ 8 - rank, 0 }, .least, problem))[0];
    }
};

test "a family's startup: rank 0's window and the least fit on every rank, or one rank's refusal named on all" {
    var ok: Families = .{};
    try loopback.run(3, &ok, Families.start);
    for (0..3) |r| {
        try std.testing.expectEqual(@as(u64, 4096), ok.window[r]);
        try std.testing.expectEqual(@as(u64, 6), ok.fit[r]);
        try std.testing.expectEqualStrings("", ok.problem[r]);
    }
    var no: Families = .{ .refuse = 2 };
    try loopback.run(3, &no, Families.start);
    const why = "rank 2 cannot hold a 4096-token window";
    try std.testing.expectEqualStrings(why, no.problem[2]);
    for (0..2) |r| {
        try std.testing.expectEqual(@as(u64, 0), no.fit[r]);
        try std.testing.expectEqualStrings("rank 2 of the group refused: " ++ why, no.problem[r]);
    }
}
