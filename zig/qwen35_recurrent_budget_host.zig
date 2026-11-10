//! Exact global scratch bounds are checked against the same helper used by native admission.
const std = @import("std");
const budget = @import("qwen35").engine.recurrent.budget;

test "Shared verification: recurrent budget: exact frame maxima retain the expensive one-row rollback copies" {
    const state = 153944064;
    const row = 4334592;
    try std.testing.expectEqual(@as(usize, 158278656), try budget.perWindow(state, row, 1, 64));
    try std.testing.expectEqual(@as(usize, 277413888), try budget.perWindow(state, row, 33, 64));
    for ([_][2]usize{ .{ 1, 277413888 }, .{ 4, 990520320 }, .{ 8, 1623634944 }, .{ 16, 2889864192 } }) |pair|
        try std.testing.expectEqual(pair[1], try budget.frame(state, row, pair[0], 128, 64));
    try std.testing.expectEqual(@as(usize, 16 * (state + row)), try budget.frame(state, row, 16, 16, 16));
    try std.testing.expectEqual(@as(usize, 13 * (state + row) + 3 * 64 * row), try budget.frame(state, row, 16, 128, 64));
}

test "Shared verification: recurrent budget: invalid geometry and overflowing byte totals fail closed" {
    try std.testing.expectEqual(@as(usize, 0), try budget.frame(1, 1, 0, 128, 64));
    try std.testing.expectError(error.InvalidRecurrentBudget, budget.frame(1, 1, 1, 0, 64));
    try std.testing.expectError(error.InvalidRecurrentBudget, budget.frame(1, 1, 1, 129, 64));
    try std.testing.expectError(error.InvalidRecurrentBudget, budget.frame(1, 1, 1, 32, 64));
    try std.testing.expectError(error.Overflow, budget.frame(std.math.maxInt(usize), 1, 1, 128, 64));
}

test "Shared verification: recurrent budget: committed roots need only exact rounded log classes" {
    const row = 4334592;
    for ([_][2]usize{ .{ 1, 277413888 }, .{ 4, 970948608 }, .{ 8, 1057640448 }, .{ 16, 1057640448 } }) |pair|
        try std.testing.expectEqual(pair[1], try budget.frame(0, row, pair[0], 128, 64));
    var worst: usize = 0;
    for ([_]usize{ 3, 9, 17, 33, 33, 33 }) |rows| worst += try budget.perWindow(0, row, rows, 64);
    try std.testing.expectEqual(worst, try budget.frame(0, row, 16, 128, 64));
    for (1..65) |rows| try std.testing.expect(try budget.perWindow(153944064, row, rows, 64) <= try budget.frame(0, row, 4, 128, 64));
}
