//! DSpark cost consumption, repricing boundaries and invalid measurements.
const std = @import("std");
const block = @import("dspark_block.zig");
const ds = @import("dspark.zig");
const lanes = @import("lanes");

test "lfm2 dspark: timing consumes CPU and GPU once without consuming pending EMA metadata" {
    var d: ds.Drafter = undefined;
    d.encode_cpu_ms = 0.25;
    d.hold_spent_ms = 0;
    d.pass_encode_cpu_ms = 0.25;
    d.pass_context_rows = 8;
    try std.testing.expectEqual(@as(f64, 0.25), block.takeHoldMs(&d));
    try std.testing.expectEqual(@as(f64, 0), d.encode_cpu_ms);
    try std.testing.expectEqual(@as(f64, 0.25), d.pass_encode_cpu_ms);
    try std.testing.expectEqual(@as(u32, 8), d.pass_context_rows);
    try std.testing.expectEqual(@as(f64, 0), block.takeHoldMs(&d));
    d.hold_spent_ms = 1.5;
    try std.testing.expectEqual(@as(f64, 1.5), block.takeHoldMs(&d));
    try std.testing.expectEqual(@as(f64, 0.25), d.pass_encode_cpu_ms);
    try std.testing.expectEqual(@as(f64, 0), block.takeHoldMs(&d));
    d.encode_cpu_ms = 0.5;
    d.hold_spent_ms = 2;
    try std.testing.expectEqual(@as(f64, 2.5), block.takeHoldMs(&d));
    try std.testing.expectEqual(@as(f64, 0), block.takeHoldMs(&d));
}

test "lfm2 dspark: a completed hold moves its block counts a tenth of the way to CPU-encode plus GPU ms" {
    var d: ds.Drafter = undefined;
    const grid = [_]lanes.hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1 }, .{ .streams = 4, .depth = 1, .ms = 2 }, .{ .streams = 4, .depth = 4, .ms = 4 }, .{ .streams = 8, .depth = 1, .ms = 3 } };
    @memcpy(d.holds[0..grid.len], &grid);
    d.holds_len = grid.len;
    d.contexts_len = 0;
    d.priceHold(4, 1, 0, 4); // priced 2 ms: the 4-block passes rise by a tenth of the doubling
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), d.holds[1].ms, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.4), d.holds[2].ms, 1e-12);
    try std.testing.expectEqual(@as(f64, 1), d.holds[0].ms);
    try std.testing.expectEqual(@as(f64, 3), d.holds[3].ms);
    d.priceHold(6, 1, 0, 0); // between 4 and 8 blocks, no measured time: nothing learned
    try std.testing.expectEqual(@as(f64, 2.2), d.holds[1].ms);
    d.priceHold(6, 1, 0, 2.6 * 2); // both neighbors move
    try std.testing.expect(d.holds[1].ms > 2.2 and d.holds[3].ms > 3 and d.holds[0].ms == 1);
    const before = d.holds[1].ms;
    d.priceHold(0, 1, 0, 5); // a pass of no blocks prices nothing
    try std.testing.expectEqual(before, d.holds[1].ms);
    d.holds_len = 0; // no grid (the load's timing holds): nothing to move
    d.priceHold(4, 1, 0, 100);
    try std.testing.expectEqual(before, d.holds[1].ms);
}

test "lfm2 dspark: hold repricing subtracts context and refuses nonpositive or nonfinite residuals" {
    var d: ds.Drafter = undefined;
    d.holds[0] = .{ .streams = 4, .depth = 1, .ms = 2 };
    d.holds_len = 1;
    d.contexts[0] = .{ .streams = 4, .rows = 8, .ms = 3 };
    d.contexts_len = 1;
    d.priceHold(4, 1, 5, 7); // seven CPU-encode + GPU ms, three estimated context ms: clean moves toward four
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), d.holds[0].ms, 1e-12);
    for ([_]f64{ 3, 2, 0, -1, std.math.inf(f64), std.math.nan(f64) }) |ms| {
        d.priceHold(4, 1, 5, ms);
        try std.testing.expectEqual(@as(f64, 2.2), d.holds[0].ms);
    }
    d.priceHold(4, 1, 9, 7); // debt beyond measured coverage must not teach a fabricated clean time
    try std.testing.expectEqual(@as(f64, 2.2), d.holds[0].ms);
}
