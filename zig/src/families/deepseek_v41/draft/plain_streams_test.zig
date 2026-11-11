//! TF_DSV41_DRAFT_PLAIN_STREAMS: the env knob, and a round with enough active windows decodes plain.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const calib_env = @import("calib_env.zig");

const gpa = std.testing.allocator;

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: [*:0]const u8) ?[]const u8 {
        for (pairs) |kv| if (std.mem.eql(u8, kv[0], std.mem.span(name))) return kv[1];
        return null;
    }
};

test "TF_DSV41_DRAFT_PLAIN_STREAMS: parsed, 0 when unset, bad values error" {
    Env.pairs = &.{};
    try std.testing.expectEqual(@as(usize, 0), (try calib_env.depthSettings(&Env.get)).plain_streams);
    Env.pairs = &.{.{ "TF_DSV41_DRAFT_PLAIN_STREAMS", "3" }};
    try std.testing.expectEqual(@as(usize, 3), (try calib_env.depthSettings(&Env.get)).plain_streams);
    Env.pairs = &.{.{ "TF_DSV41_DRAFT_PLAIN_STREAMS", "oops" }};
    try std.testing.expectError(error.InvalidCharacter, calib_env.depthSettings(&Env.get));
    Env.pairs = &.{};
}

fn runPlain(cases: []const []const u32, setup: Setup, max_news: []const u32) !Run {
    const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08 }; // a confident drafter (lanes_test's): drafts are chosen
    const t = try twin.Twin.init(gpa, dims, cases.len + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, cases.len + 1);
    defer d.deinit();
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, t.target(), d.pass(), c, .{ .shape = d.shape, .slots = 4, .depth = setup, .spec = .{ .on = true } });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams, max_news) |p, *s, *n| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = n.* });
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| try e.addStream(s);
    while (e.activeCount() > 0) try e.step();
    var ok = true;
    for (streams) |*s| ok = ok and s.emitted().len > 0;
    return .{ .drafted = e.drafted, .ok = ok, .passes = d.passes, .out = blk: {
        const out = try gpa.alloc([]u32, cases.len);
        for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
        break :blk out;
    } };
}

fn freeRun(r: *Run) void {
    for (r.out) |o| gpa.free(o);
    gpa.free(r.out);
}

const Setup = depth.Settings;

const Run = struct { drafted: u64, ok: bool, passes: u64, out: [][]u32 };

const a1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };

test "a round with plain_streams or more active windows decodes plain; below it drafts; 0 keeps drafts" {
    // identical prompts and greedy sampling keep every round a full 4-window round until the streams end together
    const four = [_][]const u32{ &a1, &a1, &a1, &a1 };
    const fours = [_]u32{ 40, 40, 40, 40 };
    // 4 active, threshold 3: every drafted window is forced to count 0
    const plain = try runPlain(&four, .{ .plain_streams = 3 }, &fours);
    defer freeRun(@constCast(&plain));
    try std.testing.expect(plain.ok);
    try std.testing.expectEqual(@as(u64, 0), plain.drafted);
    // 3 active, threshold 3: at it, plain too
    const at = try runPlain(four[0..3], .{ .plain_streams = 3 }, fours[0..3]);
    defer freeRun(@constCast(&at));
    try std.testing.expect(at.ok);
    try std.testing.expectEqual(@as(u64, 0), at.drafted);
    // 2 active, threshold 3: below it, drafts are kept
    const below = try runPlain(four[0..2], .{ .plain_streams = 3 }, fours[0..2]);
    defer freeRun(@constCast(&below));
    try std.testing.expect(below.ok);
    try std.testing.expect(below.drafted > 0);
    // 4 active, threshold 0 (off): drafts are kept
    const off = try runPlain(&four, .{}, &fours);
    defer freeRun(@constCast(&off));
    try std.testing.expect(off.ok);
    try std.testing.expect(off.drafted > 0);
}

test "plain rounds skip the draft pass; the replies stay the serial ones; drafting resumes below the threshold" {
    const four = [_][]const u32{ &a1, &a1, &a1, &a1 };
    const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08 };
    const fours = [_]u32{ 40, 40, 40, 40 };
    // off, 4 streams: every round drafts, every round runs the pass
    const off = try runPlain(&four, .{}, &fours);
    defer freeRun(@constCast(&off));
    try std.testing.expect(off.passes > 0);
    // at the threshold: shared rounds run no pass (the engine asks depth 0); the four opening asks stay
    const on = try runPlain(&four, .{ .plain_streams = 3 }, &fours);
    defer freeRun(@constCast(&on));
    try std.testing.expectEqual(@as(u64, 0), on.drafted);
    try std.testing.expectEqual(@as(u64, 4), on.passes);
    // tokens identical drafted or plain
    for (off.out, on.out) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    // serial reference
    {
        const t = try twin.Twin.init(gpa, dims, 1);
        defer t.deinit();
        for (on.out, four) |got, p| {
            const want = try t.serial(0, p, 40, null, null);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, got);
        }
    }
    // three streams, the first stops early: its last rounds run with 2 active, drafts resume (a pass again)
    const threes = [_]u32{ 6, 40, 40 };
    const tail = [_][]const u32{ &a1, &a1, &a1 };
    const mixed = try runPlain(&tail, .{ .plain_streams = 3 }, &threes);
    defer freeRun(@constCast(&mixed));
    try std.testing.expect(mixed.ok);
    try std.testing.expect(mixed.drafted > 0); // after a stream finished, 2 active: drafts asked again
    try std.testing.expect(mixed.passes > 4); // the three opening asks plus the resumed rounds
    // stream 0 stops at 6: its 6 tokens must equal the off run's opening prefix; the others match in full
    try std.testing.expectEqualSlices(u32, off.out[0][0..6], mixed.out[0]);
    for (off.out[1..3], mixed.out[1..]) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    {
        const t = try twin.Twin.init(gpa, dims, 1);
        defer t.deinit();
        for (mixed.out, tail, &threes) |got, p, *n| {
            const want = try t.serial(0, p, n.*, null, null);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, got);
        }
    }
}
