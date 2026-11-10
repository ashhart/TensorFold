//! One row of the Flash Next router. v0.6.6 `src/tensorfold/cuda/moe.py` `_topk_rows`: fp32 logits.

const std = @import("std");

/// `logits` is `experts + 1` fp32 values.
pub fn select(
    logits: []const f32,
    topk: usize,
    picks: []u16,
    weights: []f32,
    scratch: []f32,
) error{Length}!void {
    if (logits.len < 2) return error.Length;
    const experts = logits.len - 1;
    if (experts > std.math.maxInt(u16)) return error.Length;
    if (topk == 0 or topk > experts) return error.Length;
    if (picks.len != topk + 1 or weights.len != topk + 1 or scratch.len != experts) return error.Length;

    @memcpy(scratch, logits[0..experts]);
    var top = scratch[0];
    for (scratch[1..]) |v| top = @max(top, v);

    var total: f32 = 0;
    for (0..topk) |k| {
        var best_i: usize = 0;
        var best = scratch[0];
        for (scratch[1..], 1..) |v, i| {
            if (v > best) {
                best = v;
                best_i = i;
            }
        }
        const ex = @exp(best - top);
        picks[k] = @intCast(best_i);
        weights[k] = ex;
        total += ex;
        scratch[best_i] = -std.math.inf(f32);
    }
    for (weights[0..topk]) |*w| w.* = bf16(w.* / total);

    const shared: u16 = @intCast(experts);
    const gate = bf16(logits[experts]);
    picks[topk] = shared;
    weights[topk] = bf16(1.0 / (1.0 + @exp(-gate)));
}

/// fp32 through bf16 and back. NaN stays a NaN. Same rounding as the contract helper.
fn bf16(v: f32) f32 {
    const u: u32 = @bitCast(v);
    const h: u16 = if (std.math.isNan(v))
        @truncate((u >> 16) | 0x40)
    else
        @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
    return @bitCast(@as(u32, h) << 16);
}

test "ties take the lower id, and the shared expert is the last slot" {
    const logits = [_]f32{ 1, 3, 3, 0, 0 };
    var picks = [_]u16{ 9, 9, 9 };
    var weights = [_]f32{ 9, 9, 9 };
    var scratch: [4]f32 = undefined;
    try select(&logits, 2, &picks, &weights, &scratch);

    try std.testing.expectEqual(@as(u16, 1), picks[0]);
    try std.testing.expectEqual(@as(u16, 2), picks[1]);
    try std.testing.expectEqual(@as(u16, 4), picks[2]);
    try std.testing.expectEqual(@as(f32, 0.5), weights[0]);
    try std.testing.expectEqual(@as(f32, 0.5), weights[1]);
    try std.testing.expectEqual(@as(f32, 0.5), weights[2]);
}

test "one winner keeps the whole routed weight" {
    const logits = [_]f32{ 5, 5, 1, 2 };
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 9, 9 };
    var scratch: [3]f32 = undefined;
    try select(&logits, 1, &picks, &weights, &scratch);
    try std.testing.expectEqual(@as(u16, 0), picks[0]);
    try std.testing.expectEqual(@as(u16, 3), picks[1]);
    try std.testing.expectEqual(@as(f32, 1), weights[0]);
    const gate = bf16(2);
    const want = bf16(1.0 / (1.0 + @exp(-gate)));
    try std.testing.expectEqual(want, weights[1]);
}

test "a short buffer is refused before picks change" {
    const logits = [_]f32{ 1, 2, 0 };
    var picks = [_]u16{ 7, 7 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [2]f32 = undefined;
    try std.testing.expectError(error.Length, select(&logits, 2, picks[0..1], &weights, &scratch));
    try std.testing.expectEqual(@as(u16, 7), picks[0]);
    try std.testing.expectEqual(@as(f32, 3), weights[0]);
}
