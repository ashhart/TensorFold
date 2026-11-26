//! Host unit tests for Inkling 2 Mini Metal backend ring handles, grammar masks, and logprob helpers.
const std = @import("std");
const lanes = @import("lanes");
const backend = @import("backend.zig");

const ring = 4096;

test "a window's masks step by the padded vocabulary's words" {
    const words = backend.maskWords(201024);
    try std.testing.expectEqual(@as(usize, 6282), words);
    var masks: [3 * 6282]u32 = @splat(0);
    masks[2 * 6282] = 1;
    try std.testing.expectEqual(@as(usize, 0), backend.maskRow(&.{}, words, 2).len);
    try std.testing.expectEqual(@as(u32, 0), backend.maskRow(&masks, words, 1)[0]);
    try std.testing.expect(lanes.grammar.allows(backend.maskRow(&masks, words, 2), 0));
    try std.testing.expect(!lanes.grammar.allows(backend.maskRow(&masks, words, 1), 0));
}

test "sampled first-token logprob uses the last written handle, not argmax" {
    const logits = [_]f32{ 0, -1, -2, -3 };
    var tokens: [ring]u32 = @splat(0);
    tokens[2] = 3;
    const row = backend.firstLogprobRow(&logits, &tokens, 3, 0);
    try std.testing.expectEqual(@as(u32, 3), row.token);
    try std.testing.expect(std.math.isFinite(row.logprob));
    const lse = @log(@as(f64, 1) + @exp(-@as(f64, 1)) + @exp(-@as(f64, 2)) + @exp(-@as(f64, 3)));
    try std.testing.expectApproxEqAbs(@as(f32, -3) - @as(f32, @floatCast(lse)), row.logprob, 1e-6);
}

test "a token handle expires once the ring has wrapped past it" {
    var tokens: [ring]u32 = @splat(0);
    tokens[0] = 7;
    try std.testing.expectEqual(@as(u32, 7), try backend.ringToken(&tokens, ring, 0));
    try std.testing.expectError(error.InvalidInklingTokenHandle, backend.ringToken(&tokens, ring + 1, 0));
    try std.testing.expectEqual(@as(u32, 7), try backend.ringToken(&tokens, ring + 1, ring));
    try std.testing.expectError(error.InvalidInklingTokenHandle, backend.ringToken(&tokens, ring, ring));
}
