//! Host checks for Gemma CUDA grammar-constrained greedy selection.
const std = @import("std");
const backend = @import("cuda_backend.zig");
const pickAllowed = backend.pickAllowed;

test "a greedy pick under a grammar: the first largest allowed token, refusals by name" {
    const bf = struct {
        fn of(f: f32) u16 {
            return @intCast(@as(u32, @bitCast(f)) >> 16);
        }
    }.of;
    var row: [64]u16 = @splat(bf(0.5));
    row[3] = bf(9);
    row[40] = bf(2);
    row[41] = bf(2);
    var mask = [2]u32{ 0, (1 << 8) | (1 << 9) }; // tokens 40 and 41
    try std.testing.expectEqual(@as(u32, 40), try pickAllowed(&row, &mask));
    mask[0] = 1 << 3;
    try std.testing.expectEqual(@as(u32, 3), try pickAllowed(&row, &mask));
    try std.testing.expectError(error.InvalidGrammarMasks, pickAllowed(&row, &.{ 0, 0 }));
    row[10] = 0xFF80; // -inf, even where the grammar refuses it
    try std.testing.expectError(error.NonFiniteLogits, pickAllowed(&row, &mask));
}
