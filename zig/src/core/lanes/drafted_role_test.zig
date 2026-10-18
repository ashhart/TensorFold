//! The production wrapper distinguishes prompt ingestion from accepted live rows.
const std = @import("std");
const fixture = @import("drafted_test.zig");
const Sampling = @import("sampling.zig").Sampling;
const p1 = fixture.p1;
const sampled = fixture.sampled;
const run = fixture.run;
const freeOut = fixture.freeOut;

test "an external drafter's rounds commit the one-token decode, greedy and sampled, and its drafts land" {
    for ([_]?Sampling{ null, sampled }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }}, .wrapped, false);
        defer freeOut(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .plain, false);
        defer freeOut(plain);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], drafted.tokens[0]);
        // four in five of its drafts are the target's, so far more than chance (1 in 97) land
        try std.testing.expect(drafted.drafted > 0 and drafted.accepted * 3 > drafted.drafted);
        try std.testing.expect(drafted.head.rows >= p1.len - 1 + drafted.tokens[0].len - 1);
        try std.testing.expectEqual(@as(u64, p1.len), drafted.head.prompt_rows);
        try std.testing.expect(drafted.head.live_rows > 0);
    }
}
