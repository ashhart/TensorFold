//! Drafted.held_on_target through the engine: a target reading held drafts from the drafter commits the plain decode.
const std = @import("std");
const fixture = @import("drafted_test.zig");
const Sampling = @import("sampling.zig").Sampling;
const run = fixture.run;
const freeOut = fixture.freeOut;
const p1 = fixture.p1;
const long = fixture.long;
const sampled = fixture.sampled;

test "held_on_target: windows reach the target held, its own readbacks draft and commit the one-token decode" {
    for ([_]?Sampling{ null, sampled }) |s| {
        const through = try run(&.{.{ .prompt = &p1, .sampling = s }}, .wrapped_held, false);
        defer freeOut(through);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .plain, false);
        defer freeOut(plain);
        const hosted = try run(&.{.{ .prompt = &p1, .sampling = s }}, .wrapped, false);
        defer freeOut(hosted);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], through.tokens[0]);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], hosted.tokens[0]);
        // the drafts landed, and every readback was the target's: the wrapper translated no window
        try std.testing.expect(through.drafted > 0 and through.accepted > 0);
        try std.testing.expect(through.seen.reads > 0);
        try std.testing.expectEqual(through.head.helds, through.seen.reads);
        // the host route (flag off) reads back in the wrapper and hands the target no held window
        try std.testing.expectEqual(@as(u32, 0), hosted.seen.reads);
        try std.testing.expect(hosted.head.helds > 0);
    }
}

test "held_on_target: fillers of a wholly restored prompt reach the target as host tokens, then held reads resume" {
    for ([_]?Sampling{ null, sampled }) |s| {
        const through = try run(&.{.{ .prompt = &long, .sampling = s, .restored = long.len }}, .wrapped_held, false);
        defer freeOut(through);
        const plain = try run(&.{.{ .prompt = &long, .sampling = s, .drafts = false }}, .plain, false);
        defer freeOut(plain);
        try std.testing.expectEqual(@as(u32, long.len), through.cached);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], through.tokens[0]);
        try std.testing.expect(through.seen.fillers >= 1); // no row to draft from: the pending token, on the host
        try std.testing.expect(through.seen.after_filler >= 1); // the filler is spent: the next rounds read held again
        try std.testing.expectEqual(through.head.helds, through.seen.reads);
    }
}
