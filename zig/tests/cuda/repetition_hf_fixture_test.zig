//! Host validation for the generated source-pinned repetition fixture.
const std = @import("std");
const fixture = @import("repetition_hf_fixture.zig");

fn contains(history: []const u32, token: usize) bool {
    for (history) |id| if (id == token) return true;
    return false;
}

test "pinned Hugging Face repetition cases preserve transformed FP32 values and greedy assertions" {
    try std.testing.expectEqual(@as(usize, 10), fixture.cases.len);
    for (fixture.cases) |case| {
        try std.testing.expectEqual(case.logits_bits.len, case.expected_bits.len);
        const factor: f32 = @bitCast(case.penalty_bits);
        for (case.logits_bits, case.expected_bits, 0..) |raw_bits, expected_bits, token| {
            const raw: f32 = @bitCast(raw_bits);
            const actual = if (contains(case.history, token))
                if (raw < 0) raw * factor else raw / factor
            else
                raw;
            const expected: f32 = @bitCast(expected_bits);
            if (std.math.isNan(expected))
                try std.testing.expect(std.math.isNan(actual))
            else
                try std.testing.expectEqual(expected_bits, @as(u32, @bitCast(actual)));
        }
        if (case.expected_greedy) |expected| {
            var best = -std.math.inf(f32);
            var actual: u32 = 0;
            for (case.expected_bits, 0..) |value_bits, token| {
                const value: f32 = @bitCast(value_bits);
                if (value > best) {
                    best = value;
                    actual = @intCast(token);
                }
            }
            if (best == -std.math.inf(f32)) actual = 0;
            try std.testing.expectEqual(expected, actual);
        }
    }
}
