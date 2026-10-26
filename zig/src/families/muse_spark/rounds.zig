//! The lane backend's host side of a round (native.zig): verify rows, kept paths, draft entries and held drafts.
const std = @import("std");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const be = lanes.backend;

/// A stream's cache slot, the drafts the backend holds for it, and how far its prompt pass stands.
pub const Lane = struct { slot: u32, held: [16]u32 = undefined, nheld: u32 = 0, filled: u32 = 0 };
pub const Map = std.AutoHashMapUnmanaged(*const lanes.Stream, Lane);

/// Every window's rows in order: its pending token, then its held drafts or host tokens, each at its position - 1.
pub fn verifyRows(map: *Map, windows: []const be.Window, rows: []fwd.Row) !usize {
    var n: usize = 0;
    for (windows) |w| {
        if (w.parents != null) return error.TreesNotBuilt;
        const lane = map.getPtr(w.stream) orelse return error.UnknownStream;
        if (n + w.rows() > rows.len) return error.WindowTooWide;
        if (w.held > lane.nheld) return error.NoHeldDrafts;
        lane.filled = 0; // decoding: no prompt mark stands
        for (0..w.rows()) |r| {
            const tok: u32 = if (r == 0) w.pending else if (w.held > 0) lane.held[r - 1] else w.tokens[r - 1];
            rows[n] = .{ .slot = lane.slot, .pos = @intCast(w.positions[r] - 1), .token = tok };
            n += 1;
        }
    }
    return n;
}

/// Whether every window's stream decodes greedily (the forward then returns ids, not logits).
pub fn allGreedy(windows: []const be.Window) bool {
    for (windows) |w| if (w.stream.sampling) |sm| if (sm.temperature > 0) return false;
    return true;
}

/// The drafts a window's forward verified: the ones the backend held, else the host's.
pub fn verifiedDrafts(lane: *const Lane, w: be.Window, out: []u32) void {
    if (w.held > 0) @memcpy(out, lane.held[0..w.held]) else @memcpy(out, w.tokens);
}

/// Kept paths are chains from row 0: the backend builds no trees.
pub fn checkPaths(paths: []const []const u32) !void {
    for (paths) |path| {
        if (path.len == 0) return error.EmptyPath;
        for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
    }
}

/// A token fed by value, or by its handle into the ring of drawn tokens.
pub fn fed(drawn: []const u32, feed: be.Feed) u32 {
    return switch (feed) {
        .handle => |h| drawn[h % drawn.len],
        .value => |v| v,
    };
}

/// One stream's place in a draft pass.
pub const Entry = struct { token: i64, pos0: u32, slot: u32, lane: *Lane, depth: u32 };

/// A request's entry (null: it wants none), its held drafts cleared either way; `block`: the drafter's, null for none.
pub fn draftEntry(map: *Map, r: be.DraftRequest, block: ?usize, drawn: []const u32) !?Entry {
    const lane = map.getPtr(r.stream) orelse return error.UnknownStream;
    lane.nheld = 0;
    if (r.depth == 0) return null;
    const b = block orelse return error.NoDraftHead;
    const token = if (r.follow.len > 0) r.follow[r.follow.len - 1] else fed(drawn, r.first orelse return error.NoFirstToken);
    return .{ .token = token, .pos0 = @intCast(r.position - 1), .slot = lane.slot, .lane = lane, .depth = @min(r.depth, @as(u32, @intCast(b - 1))) };
}

/// A draft pass's output (`per` tokens a stream, in pass order) into each stream's held drafts, `depth` of them.
pub fn hold(entries: []const Entry, out: []const i32, per: usize) void {
    for (entries, 0..) |e, i| {
        for (0..e.depth) |j| e.lane.held[j] = @intCast(out[i * per + j]);
        e.lane.nheld = e.depth;
    }
}

fn stream(sampling: ?lanes.Sampling) lanes.Stream {
    var s: lanes.Stream = undefined;
    s.sampling = sampling;
    return s;
}

test "verify rows: pending then held or host drafts, a row a position, each stream on its slot" {
    const gpa = std.testing.allocator;
    var map: Map = .empty;
    defer map.deinit(gpa);
    var a = stream(null);
    var b = stream(null);
    try map.put(gpa, &a, .{ .slot = 3, .held = @splat(0), .nheld = 2, .filled = 40 });
    try map.put(gpa, &b, .{ .slot = 1 });
    map.getPtr(&a).?.held[0..2].* = .{ 71, 72 };
    const windows = [_]be.Window{
        .{ .stream = &a, .pending = 70, .held = 2, .tokens = &.{}, .parents = null, .positions = &.{ 41, 42, 43 } },
        .{ .stream = &b, .pending = 90, .held = 0, .tokens = &.{91}, .parents = null, .positions = &.{ 8, 9 } },
    };
    var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
    const n = try verifyRows(&map, &windows, &rows);
    try std.testing.expectEqual(@as(usize, 5), n);
    const want = [_]fwd.Row{
        .{ .slot = 3, .pos = 40, .token = 70 }, .{ .slot = 3, .pos = 41, .token = 71 }, .{ .slot = 3, .pos = 42, .token = 72 },
        .{ .slot = 1, .pos = 7, .token = 90 },  .{ .slot = 1, .pos = 8, .token = 91 },
    };
    for (want, rows[0..n]) |w, r| try std.testing.expectEqual(w, r);
    try std.testing.expectEqual(@as(u32, 0), map.get(&a).?.filled);
    var drafts: [2]u32 = undefined;
    verifiedDrafts(map.getPtr(&a).?, windows[0], &drafts);
    try std.testing.expectEqualSlices(u32, &.{ 71, 72 }, &drafts);
    verifiedDrafts(map.getPtr(&b).?, windows[1], drafts[0..1]);
    try std.testing.expectEqual(@as(u32, 91), drafts[0]);
}

test "verify refuses trees, unknown streams, windows past the rows and drafts it does not hold" {
    const gpa = std.testing.allocator;
    var map: Map = .empty;
    defer map.deinit(gpa);
    var a = stream(null);
    var other = stream(null);
    try map.put(gpa, &a, .{ .slot = 0, .nheld = 1 });
    var rows: [4]fwd.Row = undefined;
    const pos = [_]u64{ 1, 2, 3, 4, 5 };
    const tree = [_]be.Window{.{ .stream = &a, .pending = 1, .held = 0, .tokens = &.{2}, .parents = &.{ -1, 0 }, .positions = pos[0..2] }};
    try std.testing.expectError(error.TreesNotBuilt, verifyRows(&map, &tree, &rows));
    const unknown = [_]be.Window{.{ .stream = &other, .pending = 1, .held = 0, .tokens = &.{}, .parents = null, .positions = pos[0..1] }};
    try std.testing.expectError(error.UnknownStream, verifyRows(&map, &unknown, &rows));
    const wide = [_]be.Window{.{ .stream = &a, .pending = 1, .held = 0, .tokens = &.{ 2, 3, 4, 5 }, .parents = null, .positions = &pos }};
    try std.testing.expectError(error.WindowTooWide, verifyRows(&map, &wide, &rows));
    const unheld = [_]be.Window{.{ .stream = &a, .pending = 1, .held = 2, .tokens = &.{}, .parents = null, .positions = pos[0..3] }};
    try std.testing.expectError(error.NoHeldDrafts, verifyRows(&map, &unheld, &rows));
}

test "greedy rounds only when no window's stream samples above temperature 0" {
    var g = stream(null);
    var cold = stream(.{ .seed = 1, .temperature = 0 });
    var hot = stream(.{ .seed = 1, .temperature = 0.7 });
    const w = struct {
        fn of(s: *lanes.Stream) be.Window {
            return .{ .stream = s, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{1} };
        }
    }.of;
    try std.testing.expect(allGreedy(&.{ w(&g), w(&cold) }));
    try std.testing.expect(!allGreedy(&.{ w(&g), w(&hot) }));
}

test "keep takes chains from row 0 only" {
    try checkPaths(&.{ &.{0}, &.{ 0, 1, 2 } });
    try std.testing.expectError(error.EmptyPath, checkPaths(&.{&.{}}));
    try std.testing.expectError(error.TreesNotBuilt, checkPaths(&.{&.{ 0, 2 }}));
    try std.testing.expectError(error.TreesNotBuilt, checkPaths(&.{&.{1}}));
}

test "draft entries: the last follow token or the fed first, at position - 1, depth capped by the block" {
    const gpa = std.testing.allocator;
    var map: Map = .empty;
    defer map.deinit(gpa);
    var a = stream(null);
    var b = stream(null);
    try map.put(gpa, &a, .{ .slot = 2, .nheld = 5 });
    try map.put(gpa, &b, .{ .slot = 6, .nheld = 3 });
    const drawn = [_]u32{ 10, 11, 12, 13 };
    const ea = (try draftEntry(&map, .{ .stream = &a, .follow = &.{ 5, 6, 7 }, .rows = null, .start = 0, .position = 101, .depth = 12 }, 8, &drawn)).?;
    try std.testing.expectEqual(@as(i64, 7), ea.token);
    try std.testing.expectEqual(@as(u32, 100), ea.pos0);
    try std.testing.expectEqual(@as(u32, 2), ea.slot);
    try std.testing.expectEqual(@as(u32, 7), ea.depth);
    try std.testing.expectEqual(@as(u32, 0), map.get(&a).?.nheld);
    const eb = (try draftEntry(&map, .{ .stream = &b, .follow = &.{}, .first = .{ .handle = 6 }, .rows = null, .start = 0, .position = 9, .depth = 3 }, 8, &drawn)).?;
    try std.testing.expectEqual(@as(i64, 12), eb.token); // handle 6 in a ring of 4
    try std.testing.expectEqual(@as(u32, 3), eb.depth);
    try std.testing.expectEqual(@as(?Entry, null), try draftEntry(&map, .{ .stream = &a, .follow = &.{1}, .rows = null, .start = 0, .position = 2, .depth = 0 }, null, &drawn));
    try std.testing.expectError(error.NoDraftHead, draftEntry(&map, .{ .stream = &a, .follow = &.{1}, .rows = null, .start = 0, .position = 2, .depth = 1 }, null, &drawn));
    try std.testing.expectError(error.NoFirstToken, draftEntry(&map, .{ .stream = &a, .follow = &.{}, .rows = null, .start = 0, .position = 2, .depth = 1 }, 8, &drawn));
    var c = stream(null);
    try std.testing.expectError(error.UnknownStream, draftEntry(&map, .{ .stream = &c, .follow = &.{1}, .rows = null, .start = 0, .position = 2, .depth = 1 }, 8, &drawn));
    const out = [_]i32{ 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47 };
    hold(&.{ ea, eb }, &out, 7);
    try std.testing.expectEqualSlices(u32, &.{ 31, 32, 33, 34, 35, 36, 37 }, map.get(&a).?.held[0..map.get(&a).?.nheld]);
    try std.testing.expectEqualSlices(u32, &.{ 41, 42, 43 }, map.get(&b).?.held[0..map.get(&b).?.nheld]);
}
