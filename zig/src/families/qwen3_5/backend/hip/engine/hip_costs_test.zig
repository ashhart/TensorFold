//! The cost table HIP hands the lane core, built through `hip_costs.forwards` on a rig that records instead of timing.
const std = @import("std");
const lanes = @import("lanes");
const costs = @import("hip_costs.zig");
const plan = @import("../forward/plan.zig");
const T = std.testing;

/// Records each forward's windows; a forward's ms is its call number, so a price names the forward it came from.
const Rig = struct {
    calls: std.ArrayList([]const usize) = .empty,

    fn deinit(r: *Rig) void {
        for (r.calls.items) |c| T.allocator.free(c);
        r.calls.deinit(T.allocator);
    }

    pub fn time(r: *Rig, widths: []const usize) !f64 {
        try r.calls.append(T.allocator, try T.allocator.dupe(usize, widths));
        return @floatFromInt(r.calls.items.len);
    }

    /// The windows of the forward that `ms` came from (none: a price no single forward gave).
    fn of(r: *const Rig, ms: f64) ![]const usize {
        if (ms != @round(ms) or ms < 1 or ms > @as(f64, @floatFromInt(r.calls.items.len))) return error.NotOneForward;
        return r.calls.items[@as(usize, @intFromFloat(ms)) - 1];
    }
};

fn rowsOf(widths: []const usize) usize {
    var rows: usize = 0;
    for (widths) |w| rows += w;
    return rows;
}

/// The shared table the lane core builds from the timed costs, as its rounds read it.
fn coreTable(c: *const costs.Costs, batch_rows: u32) !lanes.Config {
    return lanes.Config.init(T.allocator, .{
        .exact_width = 16,
        .hidden_rows = true,
        .batch_rows = batch_rows,
        .window_costs = c.window[0..c.windows],
        .shared_costs = c.shared[0..c.shareds],
    }, batch_rows, 3);
}

test "every total row count prices at one forward of the row bucket it pads to, over the served streams" {
    var r: Rig = .{};
    defer r.deinit();
    var c: costs.Costs = .{};
    try costs.forwards(&r, 4, 64, &c);
    var cfg = try coreTable(&c, 64);
    defer cfg.deinit(T.allocator);
    for (1..65) |total| {
        const ms = cfg.shared_costs.get(@intCast(total)).?;
        const timed = try r.of(ms);
        try T.expectEqual(plan.bucketOf(total, 64), rowsOf(timed));
        try T.expectEqual(@as(usize, @min(4, rowsOf(timed))), timed.len);
        try T.expectEqual(ms, cfg.shared_costs.get(@intCast(plan.bucketOf(total, 64))).?);
    }
}

test "a total past the rows the served windows hold prices at the forward of the bucket it pads to" {
    var r: Rig = .{};
    defer r.deinit();
    var c: costs.Costs = .{};
    // four streams of 16-row windows hold 64 rows: a 65-row total pads to the 128-row bucket
    try costs.forwards(&r, 4, 128, &c);
    var cfg = try coreTable(&c, 128);
    defer cfg.deinit(T.allocator);
    try T.expectEqualSlices(usize, &.{ 32, 32, 32, 32 }, try r.of(cfg.shared_costs.get(65).?));
    for (1..129) |total| {
        const timed = try r.of(cfg.shared_costs.get(@intCast(total)).?);
        try T.expectEqual(plan.bucketOf(total, 128), rowsOf(timed));
        try T.expectEqual(@as(usize, @min(4, rowsOf(timed))), timed.len);
    }
}

test "shared timings split a bucket's rows evenly over the served streams, past eight of them" {
    var r: Rig = .{};
    defer r.deinit();
    var c: costs.Costs = .{};
    try costs.forwards(&r, 3, 48, &c);
    const three = [_][]const usize{ &.{1}, &.{ 1, 1 }, &.{ 2, 1, 1 }, &.{ 3, 3, 2 }, &.{ 6, 5, 5 }, &.{ 11, 11, 10 }, &.{ 16, 16, 16 } };
    try T.expectEqual(16 + three.len, r.calls.items.len);
    for (three, r.calls.items[16..]) |want, got| try T.expectEqualSlices(usize, want, got);
    // past the 32-row bucket a round pads to the limit: 48 rows
    for (33..49) |total| try T.expectEqualSlices(usize, &.{ 16, 16, 16 }, try r.of(c.shared[total - 1].ms));
    try T.expectEqual(@as(usize, 48), c.shareds);

    var wide: Rig = .{};
    defer wide.deinit();
    try costs.forwards(&wide, 12, 192, &c);
    const twelve: [12]usize = @splat(16);
    try T.expectEqualSlices(usize, &twelve, wide.calls.items[wide.calls.items.len - 1]);
    try T.expectEqual(@as(usize, 192), c.shareds);
}

test "a table past its entries still prices every total of the last bucket at that bucket's forward" {
    var r: Rig = .{};
    defer r.deinit();
    var c: costs.Costs = .{};
    try costs.forwards(&r, 20, 320, &c);
    try T.expectEqual(c.shared.len, c.shareds);
    var cfg = try coreTable(&c, 320);
    defer cfg.deinit(T.allocator);
    const twenty: [20]usize = @splat(16);
    for (65..321) |total| try T.expectEqualSlices(usize, &twenty, try r.of(cfg.shared_costs.get(@intCast(total)).?));
}

test "the lone window table stays one window of each width to the widest" {
    for ([_]usize{ 1, 4 }) |streams| {
        var r: Rig = .{};
        defer r.deinit();
        var c: costs.Costs = .{};
        try costs.forwards(&r, streams, 16 * streams, &c);
        try T.expectEqual(@as(usize, 16), c.windows);
        for (c.window[0..c.windows], 1..) |w, width| {
            try T.expectEqual(@as(u32, @intCast(width)), w.width);
            try T.expectEqualSlices(usize, &.{width}, try r.of(w.ms));
        }
    }
}

test "a lone window times one repeated token; shared rows carry a token a row inside the vocabulary" {
    var tokens: [256]u32 = undefined;
    costs.fillTokens(&tokens, 1, 248320);
    for (tokens) |t| try T.expectEqual(@as(u32, 1), t);

    costs.fillTokens(&tokens, 4, 248320);
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(T.allocator);
    for (tokens) |t| {
        try T.expect(t > 0 and t < 248320);
        try T.expect(!(try seen.getOrPut(T.allocator, t)).found_existing);
    }

    costs.fillTokens(tokens[0..16], 2, 1000);
    for (tokens[0..16]) |t| try T.expect(t > 0 and t < 1000);
}

test "shared rows stay distinct over any vocabulary, spans 7919 or 7927 divide included" {
    // 15839 and 15855 ids: spans of twice 7919 and twice 7927
    for ([_]usize{ 15839, 15855, 65537, 151936, 248320 }) |vocab| {
        var tokens: [256]u32 = undefined;
        costs.fillTokens(&tokens, 2, vocab);
        var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer seen.deinit(T.allocator);
        for (tokens) |t| try T.expect(!(try seen.getOrPut(T.allocator, t)).found_existing);
    }
}

test "a timed round's tokens and greedy draws follow its windows: one repeated token alone, a token a row shared" {
    const draw = @import("hip_serve").draw;
    var tokens: [8]u32 = undefined;
    var reqs: [8]draw.Request = undefined;

    try T.expectEqual(@as(usize, 5), costs.fillRound(&tokens, &reqs, &.{5}, 248320));
    for (tokens[0..5]) |t| try T.expectEqual(@as(u32, 1), t);
    for (reqs[0..5], 0..) |q, j| {
        try T.expect(q.sampling == null);
        try T.expectEqual(reqs[0].position + @as(u64, j), q.position);
    }

    try T.expectEqual(@as(usize, 5), costs.fillRound(&tokens, &reqs, &.{ 2, 3 }, 248320));
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(T.allocator);
    for (tokens[0..5]) |t| try T.expect(!(try seen.getOrPut(T.allocator, t)).found_existing);
    // each window's draws start at the same position
    for ([_][2]usize{ .{ 0, 2 }, .{ 1, 3 } }) |pair| try T.expectEqual(reqs[pair[0]].position, reqs[pair[1]].position);
    try T.expectEqual(reqs[2].position + 2, reqs[4].position);
    for (reqs[0..5]) |q| try T.expect(q.sampling == null);
}
