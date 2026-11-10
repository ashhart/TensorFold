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
    Env.pairs = &.{.{ "TF_DSV41_DRAFT_PLAIN_STREAMS", "4" }};
    try std.testing.expectEqual(@as(usize, 4), (try calib_env.depthSettings(&Env.get)).plain_streams);
    Env.pairs = &.{.{ "TF_DSV41_DRAFT_PLAIN_STREAMS", "oops" }};
    try std.testing.expectError(error.InvalidCharacter, calib_env.depthSettings(&Env.get));
    Env.pairs = &.{};
}

fn runPlain(cases: []const []const u32, setup: Setup) !Run {
    const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08 }; // a confident drafter (lanes_test's): drafts are chosen
    const t = try twin.Twin.init(gpa, dims, cases.len + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, cases.len + 1);
    defer d.deinit();
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, t.target(), d.pass(), c, .{ .shape = d.shape, .slots = 4, .depth = setup });
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
    for (cases, streams) |p, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = 40 });
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| try e.addStream(s);
    while (e.activeCount() > 0) try e.step();
    var ok = true;
    for (streams) |*s| ok = ok and s.emitted().len > 0;
    return .{ .drafted = e.drafted, .ok = ok };
}

const Setup = depth.Settings;

const Run = struct { drafted: u64, ok: bool };

const a1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };

test "a round with plain_streams or more active windows decodes plain; below it drafts; 0 keeps drafts" {
    // identical prompts and greedy sampling keep every round a full 4-window round until the streams end together
    const four = [_][]const u32{ &a1, &a1, &a1, &a1 };
    // 4 active, threshold 4: every drafted window is forced to count 0
    const plain = try runPlain(&four, .{ .plain_streams = 4 });
    try std.testing.expect(plain.ok);
    try std.testing.expectEqual(@as(u64, 0), plain.drafted);
    // 3 active, threshold 4: below it, drafts are kept
    const below = try runPlain(four[0..3], .{ .plain_streams = 4 });
    try std.testing.expect(below.ok);
    try std.testing.expect(below.drafted > 0);
    // 4 active, threshold 0 (off): drafts are kept
    const off = try runPlain(&four, .{});
    try std.testing.expect(off.ok);
    try std.testing.expect(off.drafted > 0);
}
