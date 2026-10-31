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
    try costs.forwards(&r, 4, 16, 64, &c);
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

test "shared timings split a bucket's rows evenly over the served streams, past eight of them" {
    var r: Rig = .{};
    defer r.deinit();
    var c: costs.Costs = .{};
    try costs.forwards(&r, 3, 16, 48, &c);
    const three = [_][]const usize{ &.{1}, &.{ 1, 1 }, &.{ 2, 1, 1 }, &.{ 3, 3, 2 }, &.{ 6, 5, 5 }, &.{ 11, 11, 10 }, &.{ 16, 16, 16 } };
    try T.expectEqual(16 + three.len, r.calls.items.len);
    for (three, r.calls.items[16..]) |want, got| try T.expectEqualSlices(usize, want, got);
    // past the 32-row bucket a round pads to the limit: 48 rows
    for (33..49) |total| try T.expectEqualSlices(usize, &.{ 16, 16, 16 }, try r.of(c.shared[total - 1].ms));
    try T.expectEqual(@as(usize, 48), c.shareds);

    var wide: Rig = .{};
    defer wide.deinit();
    try costs.forwards(&wide, 12, 16, 192, &c);
    const twelve: [12]usize = @splat(16);
    try T.expectEqualSlices(usize, &twelve, wide.calls.items[wide.calls.items.len - 1]);
    try T.expectEqual(@as(usize, 192), c.shareds);
}

test "a table past its entries still prices every total of the last bucket at that bucket's forward" {
    var r: Rig = .{};
    defer r.deinit();
    var c: costs.Costs = .{};
    try costs.forwards(&r, 20, 16, 320, &c);
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
        try costs.forwards(&r, streams, 16, 16 * streams, &c);
        try T.expectEqual(@as(usize, 16), c.windows);
        for (c.window[0..c.windows], 1..) |w, width| {
            try T.expectEqual(@as(u32, @intCast(width)), w.width);
            try T.expectEqualSlices(usize, &.{width}, try r.of(w.ms));
        }
    }
}
