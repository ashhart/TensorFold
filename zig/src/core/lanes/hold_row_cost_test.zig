//! A learned row cost below the table and a measuring round's least rows, priced under a timed hold grid.
const std = @import("std");
const alloc = @import("allocate.zig");
const depth = @import("depth.zig");
const hold_offers = @import("hold_offers.zig");
const price = @import("hold_price.zig");
const Fixture = @import("hold_prefix_test.zig").Fixture;
const Table = @import("table.zig").Table;

const gpa = std.testing.allocator;

// SYNTHETIC costs below pin the arithmetic of the combined price, not any device's timings.
fn tenPerRow() !Table {
    return alloc.extendCosts(gpa, &.{ .{ .width = 2, .ms = 20.0 }, .{ .width = 8, .ms = 80.0 } }, 8);
}

const grid = [_]price.HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 1.0 }, .{ .streams = 1, .depth = 2, .ms = 2.0 },
    .{ .streams = 2, .depth = 1, .ms = 1.0 }, .{ .streams = 2, .depth = 2, .ms = 2.0 },
};

fn expectCounts(want: []const u32, head: alloc.HeadCost) !void {
    var costs = try tenPerRow();
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.5, 0.25 }, &.{ 0.5, 0.25 } };
    const got = try alloc.allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 8, head);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u32, want, got);
}

test "a hold grid prices the passes while a learned row cost below the table prices the rows" {
    const batched: alloc.HeadCost = .{ .batched = true, .hold_costs = &grid };
    // the table's 10 ms a row and the passes' 1 ms: no draft pays
    try expectCounts(&.{ 0, 0 }, batched);
    // each row 8 ms below the table: both first drafts pay, the second drafts' deeper pass does not
    var cheap = batched;
    cheap.row_ms = -8.0;
    try expectCounts(&.{ 1, 1 }, cheap);
    // a fall past the table's own slope stops at the streams' plain width: every draft then pays only its pass
    var wild = batched;
    wild.row_ms = -25.0;
    try expectCounts(&.{ 2, 2 }, wild);
}

test "a measuring round reaches its rows under a hold grid, then keeps only rows that pay" {
    var head: alloc.HeadCost = .{ .batched = true, .hold_costs = &grid };
    head.least_rows = 4;
    try expectCounts(&.{ 1, 1 }, head);
    head.least_rows = 7; // beyond every node on offer: all of them
    try expectCounts(&.{ 2, 2 }, head);
    // a learned fall that pays four rows reaches six when the round must measure them
    head.row_ms = -8.0;
    head.least_rows = 0;
    try expectCounts(&.{ 1, 1 }, head);
    head.least_rows = 6;
    try expectCounts(&.{ 2, 2 }, head);
}

test "the round's forward bounds a learned fall at the plain width, and a rise is linear" {
    var costs = try tenPerRow();
    defer costs.deinit(gpa);
    const base: price.Base = .{ .streams = 2, .rows = 2, .tokens = 2 };
    var wild = base;
    wild.row_ms = -25.0;
    try std.testing.expectEqual(@as(?f64, 20.0), wild.forwardMs(&costs, 2));
    try std.testing.expectEqual(@as(?f64, 20.0), wild.forwardMs(&costs, 8));
    var cheap = base;
    cheap.row_ms = -8.0;
    try std.testing.expectEqual(@as(?f64, 24.0), cheap.forwardMs(&costs, 4));
    try std.testing.expectEqual(@as(?f64, 32.0), cheap.forwardMs(&costs, 8));
    var dear = base;
    dear.row_ms = 3.0;
    try std.testing.expectEqual(@as(?f64, 46.0), dear.forwardMs(&costs, 4));
    try std.testing.expectEqual(@as(?f64, null), dear.forwardMs(&costs, 9));
}

test "a reserved hold is funded by rows cheaper than the table, and by a fall no further than the plain width" {
    var f: Fixture = .{};
    try f.init(.{
        .drafts = 1,
        .forward = .{ 10, 20, 30, 40, 50, 60, 70, 80 },
        .holds = &.{.{ .streams = 1, .depth = 1, .ms = 1.0 }},
        .prior = &.{0.9},
    });
    defer f.deinit();
    var state: ?depth.State = null;
    const whos = [_]depth.Who{.{ .state = &state, .draft_room = 64, .finished = false, .forced = false, .granted = 1, .gain = 0.9, .prices = &.{0.9} }};
    // 1.9 tokens in 21 ms lose to 1 in 10; 12 ms of verify win; a fall past the plain width still verifies at 10 ms
    const cases = [_]struct { row_ms: f64, held: i64 }{ .{ .row_ms = 0, .held = 0 }, .{ .row_ms = -8, .held = 1 }, .{ .row_ms = -100, .held = 1 } };
    for (cases) |c| {
        var base = price.Base.pending(1);
        base.row_ms = c.row_ms;
        var out = [_]i64{1};
        try hold_offers.pay(&f.engine.rule, &whos, false, &out, base);
        try std.testing.expectEqual(c.held, out[0]);
    }
}
