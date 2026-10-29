//! A trainer stores every final hidden row and only answer logits; both ranges have independent identity checks.
const std = @import("std");

pub fn check(row: usize, start: usize, hidden: []const u16, served_hidden: []const u16, logits: []const u16, served_logits: []const u16) !void {
    if (!std.mem.eql(u16, hidden, served_hidden)) return error.TaughtTrainerHiddenDiffers;
    if (row + 1 >= start and !std.mem.eql(u16, logits, served_logits)) return error.TaughtTrainerForwardDiffers;
}

test "qwen27 identity: poisoned skipped prompt logits are not outputs; every hidden and answer row is checked" {
    const hidden = [_]u16{ 0x3f80, 0x4000 };
    const logits = [_]u16{ 0x4040, 0x4080 };
    const skipped = [_]u16{ 0x7fc1, 0x7fc1 };
    try check(0, 2, &hidden, &hidden, &skipped, &logits);
    try check(1, 2, &hidden, &hidden, &logits, &logits);
    try std.testing.expectError(error.TaughtTrainerHiddenDiffers, check(0, 2, &skipped, &hidden, &skipped, &logits));
    try std.testing.expectError(error.TaughtTrainerForwardDiffers, check(1, 2, &hidden, &hidden, &skipped, &logits));
    // A one-token prompt has no skipped head row.
    try std.testing.expectError(error.TaughtTrainerForwardDiffers, check(0, 1, &hidden, &hidden, &skipped, &logits));
}
