//! The shared DSpark host against what Metal's dspark.zig and dspark_state.zig do today, over a recording backend.
const std = @import("std");
const lanes = @import("lanes");
const dl = @import("dspark_lanes.zig");
const macos = @import("builtin").os.tag == .macos;
const metal_state = if (macos) @import("metal/dspark_state.zig") else struct {};

const testing = std.testing;
const dr = lanes.drafter;

test "one tokenBytes prices shared fit, lanes and scratch; lane prices match Metal" {
    const t = dl.tokenBytes();
    try testing.expectEqual(@as(u64, 4096), t.lane);
    try testing.expectEqual(@as(u64, 768), t.shared);
    try testing.expectEqual(@as(usize, 1152), dl.local_ring);
    try testing.expectEqual(@as(u64, 9_437_184 + 4608), dl.laneFixedBytes());
    try testing.expectEqual(dl.laneFixedBytes() + 128 * 4096, dl.laneBytes(1));
    try testing.expectEqual(dl.laneFixedBytes() + 1152 * 4096, dl.laneBytes(1025));
    try testing.expectEqual(@as(u64, 768 * 1032), dl.scoreBytes(1024));
    if (macos) {
        try testing.expectEqual(metal_state.laneTokenBytes(), t.lane);
        try testing.expectEqual(metal_state.laneFixedBytes(), dl.laneFixedBytes());
        try testing.expectEqual(metal_state.local_ring, dl.local_ring);
    }
    for ([_]usize{ 1, 127, 128, 129, 1025, 4096, 262143, 1 << 20 }) |n| {
        if (macos) {
            try testing.expectEqual(metal_state.laneBytes(n), dl.laneBytes(n));
            try testing.expectEqual(metal_state.globalRing(n), dl.rings(n)[1]);
        }
        // Metal no longer holds score scratch; the shared host price is checked on its own terms.
        try testing.expectEqual(t.shared * (n + dl.block), dl.scoreBytes(n));
        // The fit never prices a stream under its lane, and over it by at most the ring's slack.
        const f = dl.fit();
        const bound = f.fixed + n * f.lane;
        try testing.expect(bound >= dl.laneBytes(n) and bound - dl.laneBytes(n) <= dl.ring_slack * f.lane);
        try testing.expectEqual(dl.scoreBytes(0) + n * f.shared, dl.scoreBytes(n));
    }
    if (macos) {
        try testing.expectEqual(@as(u32, metal_state.block), dl.block);
        try testing.expectEqual(@as(u32, metal_state.absorb_rows), dl.absorb_rows);
        try testing.expectEqual(@as(u32, metal_state.vocabulary), dl.vocabulary);
        try testing.expectEqual(@as(u32, metal_state.hidden), dl.hidden);
    }
}

/// Records what the shared host asks of a backend; its drafts are `pending + 1 + step`, or one past the vocabulary.
const Fake = struct {
    pub const Lane = struct { id: u32, rings: [3]usize };
    const Piece = struct { lane: u32, offset: u64, position: u64, rows: u32 };
    const Drafted = struct { lane: u32, from: u64, anchor: u64, pending: u32, depth: u32, sampled: bool };

    opened: u32 = 0,
    closed: u32 = 0,
    scores: u64 = 0,
    absorb_calls: u32 = 0,
    hold_calls: u32 = 0,
    nonfinite: bool = false,
    pieces: [8]Piece = undefined,
    n_pieces: usize = 0,
    holds: [4]Drafted = undefined,
    n_holds: usize = 0,

    pub fn openLane(f: *Fake, rings: [3]usize) !Lane {
        f.opened += 1;
        return .{ .id = f.opened, .rings = rings };
    }
    pub fn closeLane(f: *Fake, _: *Lane) void {
        f.closed += 1;
    }
    pub fn setScores(f: *Fake, bytes: u64) !void {
        f.scores = bytes;
    }
    pub fn policy(_: *const Fake) dl.Policy {
        return .{ .step_ms = 0.5, .plain_guard = false, .batched = true };
    }
    pub fn absorb(f: *Fake, pieces: []const dl.Absorb(Lane)) !void {
        f.absorb_calls += 1;
        f.n_pieces = pieces.len;
        for (pieces, 0..) |p, i| f.pieces[i] = .{ .lane = p.lane.id, .offset = p.offset, .position = p.position, .rows = p.rows };
    }
    pub fn hold(f: *Fake, items: []const dl.Hold(Lane), drafts: [][dl.depth]u32) !void {
        f.hold_calls += 1;
        f.n_holds = items.len;
        for (items, drafts, 0..) |h, *out, i| {
            f.holds[i] = .{ .lane = h.lane.id, .from = h.from, .anchor = h.anchor, .pending = h.pending, .depth = h.depth, .sampled = h.sampling != null };
            for (out[0..h.depth], 0..) |*t, s| t.* = if (f.nonfinite) dl.vocabulary else h.pending + 1 + @as(u32, @intCast(s));
        }
    }
};

fn features(rows: u32, offset: u64) lanes.backend.Features {
    return .{ .buffer = 0x1000, .offset = offset, .rows = rows, .row_bytes = dl.feature_row_bytes, .space = .device, .dtype = .bf16 };
}

fn stream(id: []const u8) !lanes.Stream {
    return lanes.Stream.init(testing.allocator, .{ .id = id, .prompt = &.{ 1, 2, 3 }, .max_new = 4 });
}

test "a stream's rows commit in order: chunks of 64, a gap starts the context anew, a step back is refused" {
    var fake: Fake = .{};
    var host = dl.Drafter(Fake).init(testing.allocator, &fake);
    defer host.deinit();
    try host.setContext(300);
    try testing.expectEqual(dl.scoreBytes(300), fake.scores);
    const d = host.drafter();
    try testing.expectEqual(dl.laneBytes(300), d.facts().stream_bytes);
    try testing.expect(d.facts().batched and !d.facts().plain_guard and d.facts().depth == 7);
    var s = try stream("a");
    defer s.deinit(testing.allocator);
    try d.open(&s);
    try testing.expectError(error.DraftStreamOpen, d.open(&s));
    try d.absorb(&.{.{ .stream = &s, .features = features(130, 64), .start = 0, .follow = &.{} }});
    try testing.expectEqual(@as(usize, 3), fake.n_pieces);
    const row = dl.feature_row_bytes;
    try testing.expectEqual(Fake.Piece{ .lane = 1, .offset = 64 + 64 * row, .position = 64, .rows = 64 }, fake.pieces[1]);
    try testing.expectEqual(Fake.Piece{ .lane = 1, .offset = 64 + 128 * row, .position = 128, .rows = 2 }, fake.pieces[2]);
    try d.hold(&.{.{ .stream = &s, .pending = 9, .position = 131, .depth = 12 }});
    try testing.expectEqual(Fake.Drafted{ .lane = 1, .from = 0, .anchor = 130, .pending = 9, .depth = 7, .sampled = false }, fake.holds[0]);
    var got: [7]u32 = undefined;
    try d.held(&.{&s}, &.{&got});
    try testing.expectEqualSlices(u32, &.{ 10, 11, 12, 13, 14, 15, 16 }, &got);
    var more: [8]u32 = undefined;
    try testing.expectError(error.DraftHeldFewer, d.held(&.{&s}, &.{&more}));
    try testing.expectError(error.DraftContextGap, d.absorb(&.{.{ .stream = &s, .features = features(1, 0), .start = 129, .follow = &.{} }}));
    // A gap: the context starts at 200 and the held drafts are gone, as Metal's absorb leaves them.
    try d.absorb(&.{.{ .stream = &s, .features = features(5, 0), .start = 200, .follow = &.{} }});
    try testing.expectError(error.DraftHeldFewer, d.held(&.{&s}, &.{got[0..1]}));
    try d.hold(&.{.{ .stream = &s, .pending = 1, .position = 206, .depth = 3, .sampling = .{ .seed = 1, .temperature = 0.5 } }});
    try testing.expectEqual(Fake.Drafted{ .lane = 1, .from = 200, .anchor = 205, .pending = 1, .depth = 3, .sampled = true }, fake.holds[0]);
    try testing.expectError(error.DraftFeaturesUnsupported, d.absorb(&.{.{ .stream = &s, .features = .{ .buffer = 1, .rows = 1, .row_bytes = dl.feature_row_bytes, .space = .host, .dtype = .bf16 }, .start = 205, .follow = &.{} }}));
    try testing.expectError(error.TreesNotBuilt, d.absorb(&.{.{ .stream = &s, .features = features(1, 0), .start = 205, .follow = &.{}, .rows = &.{0} }}));
    try testing.expectError(error.DSparkContextActive, host.setContext(400));
    d.release(&s);
    try testing.expectEqual(@as(u32, 1), fake.closed);
    try host.setContext(400);
    try testing.expectEqual(dl.scoreBytes(400), fake.scores);
}

test "one backend call drafts every stream's hold, and a refused hold reaches no backend" {
    var fake: Fake = .{};
    var host = dl.Drafter(Fake).init(testing.allocator, &fake);
    defer host.deinit();
    try host.setContext(1024);
    const d = host.drafter();
    var a = try stream("a");
    defer a.deinit(testing.allocator);
    var b = try stream("b");
    defer b.deinit(testing.allocator);
    var c = try stream("c");
    defer c.deinit(testing.allocator);
    try d.open(&a);
    try d.open(&b);
    try d.absorb(&.{ .{ .stream = &a, .features = features(3, 0), .start = 0, .follow = &.{} }, .{ .stream = &b, .features = features(5, 0), .start = 0, .follow = &.{} } });
    try testing.expectEqual(@as(u32, 1), fake.absorb_calls);
    try testing.expectEqual(@as(usize, 2), fake.n_pieces);
    try d.hold(&.{ .{ .stream = &a, .pending = 100, .position = 4, .depth = 7 }, .{ .stream = &b, .pending = 200, .position = 6, .depth = 2 } });
    try testing.expectEqual(@as(u32, 1), fake.hold_calls);
    try testing.expectEqual(@as(usize, 2), fake.n_holds);
    var ga: [7]u32 = undefined;
    var gb: [2]u32 = undefined;
    try d.held(&.{ &a, &b }, &.{ &ga, &gb });
    try testing.expectEqualSlices(u32, &.{ 101, 102, 103, 104, 105, 106, 107 }, &ga);
    try testing.expectEqualSlices(u32, &.{ 201, 202 }, &gb);
    const refused = [_]struct { anyerror, []const dr.Hold }{
        .{ error.DraftStreamUnknown, &.{ .{ .stream = &a, .pending = 1, .position = 4, .depth = 1 }, .{ .stream = &c, .pending = 1, .position = 1, .depth = 1 } } },
        .{ error.DraftContextGap, &.{ .{ .stream = &a, .pending = 1, .position = 4, .depth = 1 }, .{ .stream = &b, .pending = 1, .position = 5, .depth = 1 } } },
        .{ error.DraftDepth, &.{.{ .stream = &a, .pending = 1, .position = 4, .depth = 0 }} },
    };
    for (refused) |r| try testing.expectError(r[0], d.hold(r[1]));
    try testing.expectEqual(@as(u32, 1), fake.hold_calls);
    fake.nonfinite = true;
    try testing.expectError(error.DraftNonfinite, d.hold(&.{.{ .stream = &a, .pending = 1, .position = 4, .depth = 2 }}));
    try testing.expectError(error.DraftStreamUnknown, d.held(&.{&c}, &.{gb[0..1]}));
    try testing.expectEqual(@as(u32, 2), fake.opened);
}
