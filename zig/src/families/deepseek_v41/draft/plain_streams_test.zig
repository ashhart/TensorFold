//! TF_DSV41_DRAFT_PLAIN_STREAMS: the env knob, and a round with enough decoding streams decodes plain.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const calib_env = @import("calib_env.zig");

const gpa = std.testing.allocator;
const t_ = std.testing;

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: [*:0]const u8) ?[]const u8 {
        for (pairs) |kv| if (std.mem.eql(u8, kv[0], std.mem.span(name))) return kv[1];
        return null;
    }
};

fn plainStreams(raw: []const u8) !usize {
    Env.pairs = &.{.{ "TF_DSV41_DRAFT_PLAIN_STREAMS", raw }};
    defer Env.pairs = &.{};
    return (try calib_env.depthSettings(&Env.get)).plain_streams;
}

test "TF_DSV41_DRAFT_PLAIN_STREAMS: 0 when unset, 0 to 16 taken, anything else refused by name" {
    Env.pairs = &.{};
    try t_.expectEqual(@as(usize, 0), (try calib_env.depthSettings(&Env.get)).plain_streams);
    for ([_][]const u8{ "0", "3", " 16 " }, [_]usize{ 0, 3, 16 }) |raw, want| try t_.expectEqual(want, try plainStreams(raw));
    for ([_][]const u8{ "17", "4294967296", "18446744073709551616", "-1", "oops", "3x" }) |raw|
        try t_.expectError(error.DraftPlainStreams, plainStreams(raw));
}

/// A run's settings: the switch, the target's history, the drafter's ring check, staggered arrivals, sampling.
pub const Opts = struct { set: depth.Settings = .{}, history: bool = true, check_ring: bool = false, stagger: bool = false, sampling: ?lanes.Sampling = null };

pub const Run = struct {
    drafted: u64,
    accepted: u64,
    ok: bool,
    passes: u64,
    ingests: u64,
    out: [][]u32,

    pub fn deinit(r: Run) void {
        for (r.out) |o| gpa.free(o);
        gpa.free(r.out);
    }
};

/// Streams of these prompts and lengths through the lanes over the twin; staggered: one added each step.
pub fn run(cases: []const []const u32, max_news: []const u32, o: Opts) !Run {
    const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08 }; // a confident drafter (lanes_test's): drafts are chosen
    const t = try twin.Twin.init(gpa, dims, cases.len + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, cases.len + 1);
    defer d.deinit();
    d.check_ring = o.check_ring;
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const tgt = if (o.history) t.target() else t.targetNoHistory();
    const x = try ln.Lanes.init(gpa, tgt, d.pass(), c, .{ .shape = d.shape, .slots = 4, .depth = o.set, .spec = .{ .on = true } });
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
    for (cases, streams, max_news) |p, *s, n| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = n, .sampling = o.sampling });
    defer for (streams) |*s| s.deinit(gpa);
    var added: usize = 0;
    if (!o.stagger) {
        for (streams) |*s| try e.addStream(s);
        added = streams.len;
    }
    var steps: usize = 0;
    while (e.activeCount() > 0 or added < streams.len) : (steps += 1) {
        if (added < streams.len) {
            try e.addStream(&streams[added]);
            added += 1;
        }
        try e.step();
        if (steps > 1000) return error.TooManySteps;
    }
    var ok = true;
    for (streams) |*s| ok = ok and s.emitted().len > 0;
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*q, *s| q.* = try gpa.dupe(u32, s.emitted());
    return .{ .drafted = e.drafted, .accepted = e.accepted, .ok = ok, .passes = d.passes, .ingests = d.ingests, .out = out };
}

/// Each reply is its prompt's serial decode at its own length.
pub fn expectSerial(r: Run, cases: []const []const u32, max_news: []const u32, sampling: ?lanes.Sampling) !void {
    const t = try twin.Twin.init(gpa, .{ .bigram = 8.0, .head = 0.08 }, 1);
    defer t.deinit();
    try t_.expect(r.ok);
    for (r.out, cases, max_news) |got, p, n| {
        const want = try t.serial(0, p, n, sampling, null);
        defer gpa.free(want);
        try t_.expectEqualSlices(u32, want, got);
    }
}

pub const a1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const four = [_][]const u32{ &a1, &a1, &a1, &a1 };
const fours = [_]u32{ 40, 40, 40, 40 };

test "a round with plain_streams or more decoding streams decodes plain; below it drafts; 0 keeps drafts" {
    // identical prompts and greedy sampling keep every round a full round until the streams end together
    const plain = try run(&four, &fours, .{ .set = .{ .plain_streams = 3 } });
    defer plain.deinit();
    try t_.expectEqual(@as(u64, 0), plain.drafted);
    const at = try run(four[0..3], fours[0..3], .{ .set = .{ .plain_streams = 3 } });
    defer at.deinit();
    try t_.expectEqual(@as(u64, 0), at.drafted);
    const below = try run(four[0..2], fours[0..2], .{ .set = .{ .plain_streams = 3 } });
    defer below.deinit();
    try t_.expect(below.drafted > 0);
    const off = try run(&four, &fours, .{});
    defer off.deinit();
    try t_.expect(off.drafted > 0);
    for ([_]Run{ plain, at, below, off }, [_]usize{ 4, 3, 2, 4 }) |r, n| try expectSerial(r, four[0..n], fours[0..n], null);
}

test "below the threshold the switch changes nothing: 1 and 2 streams draft from their first ask as with it off" {
    for ([_]usize{ 1, 2 }) |n| for ([_]bool{ false, true }) |stagger| {
        const on = try run(four[0..n], fours[0..n], .{ .set = .{ .plain_streams = 3 }, .stagger = stagger });
        defer on.deinit();
        const off = try run(four[0..n], fours[0..n], .{ .stagger = stagger });
        defer off.deinit();
        try t_.expect(on.drafted > 0);
        try t_.expectEqual(off.passes, on.passes);
        try t_.expectEqual(off.drafted, on.drafted);
        try t_.expectEqual(off.accepted, on.accepted);
        try t_.expectEqual(off.ingests, on.ingests);
        for (off.out, on.out) |a, b| try t_.expectEqualSlices(u32, a, b);
    };
}

test "at the threshold only the asks made before plain_streams streams decode run a pass; replies unchanged" {
    const on = try run(&four, &fours, .{ .set = .{ .plain_streams = 3 } });
    defer on.deinit();
    try t_.expectEqual(@as(u64, 0), on.drafted);
    try t_.expect(on.passes <= 2); // the first two streams' opening asks, made while fewer than three decoded
    try expectSerial(on, &four, &fours, null);
}

test "staggered arrivals: streams draft until plain_streams decode, plain after, drafting again as they end" {
    const lens = [_]u32{ 40, 46, 52, 58 };
    const on = try run(&four, &lens, .{ .set = .{ .plain_streams = 3 }, .stagger = true, .check_ring = true });
    defer on.deinit();
    try t_.expect(on.passes > 0 and on.drafted > 0);
    try expectSerial(on, &four, &lens, null);
}
