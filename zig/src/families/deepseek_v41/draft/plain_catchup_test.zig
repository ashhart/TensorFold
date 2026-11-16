//! Plain rounds absorb nothing into the drafter; the next ask catches it up from the target's history, exactly.
const std = @import("std");
const lanes = @import("lanes");
const ps = @import("plain_streams_test.zig");

const t_ = std.testing;
const warm: lanes.Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.02 };
const four = [_][]const u32{ &ps.a1, &ps.a1, &ps.a1, &ps.a1 };

test "plain rounds run no ingest when the target keeps history; without it each round absorbs its rows" {
    const lens = [_]u32{ 40, 40, 40, 40 };
    const kept = try ps.run(&four, &lens, .{ .set = .{ .plain_streams = 3 } });
    defer kept.deinit();
    const each = try ps.run(&four, &lens, .{ .set = .{ .plain_streams = 3 }, .history = false });
    defer each.deinit();
    try t_.expectEqual(@as(u64, 4), kept.ingests); // the four prompts' rows only
    try t_.expect(each.ingests > 4 * 30);
    for (kept.out, each.out) |a, b| try t_.expectEqualSlices(u32, a, b);
    try ps.expectSerial(kept, &four, &lens, null);
}

test "dropping below the threshold catches the drafter up: the same drafts as a drafter fed every round" {
    // two streams end early: rounds turn plain to drafting; the long pair's gap is past the drafter's window (16)
    const cases = [_][4]u32{ .{ 6, 6, 40, 40 }, .{ 30, 30, 70, 70 } };
    for (&cases) |*lens| for ([_]?lanes.Sampling{ null, warm }) |smp| {
        const kept = try ps.run(&four, lens, .{ .set = .{ .plain_streams = 3 }, .check_ring = true, .sampling = smp });
        defer kept.deinit();
        const each = try ps.run(&four, lens, .{ .set = .{ .plain_streams = 3 }, .history = false, .check_ring = true, .sampling = smp });
        defer each.deinit();
        try t_.expect(kept.drafted > 0);
        try t_.expectEqual(each.drafted, kept.drafted);
        try t_.expectEqual(each.accepted, kept.accepted);
        try t_.expectEqual(each.passes, kept.passes);
        try t_.expect(kept.ingests < each.ingests);
        for (kept.out, each.out) |a, b| try t_.expectEqualSlices(u32, a, b);
        try ps.expectSerial(kept, &four, lens, smp);
    };
}

test "a never-plain run and a run that went plain give the same replies, greedy and sampled" {
    const lens = [_]u32{ 8, 12, 50, 50 };
    for ([_]?lanes.Sampling{ null, warm }) |smp| {
        const never = try ps.run(&four, &lens, .{ .sampling = smp, .check_ring = true });
        defer never.deinit();
        const went = try ps.run(&four, &lens, .{ .set = .{ .plain_streams = 3 }, .sampling = smp, .check_ring = true });
        defer went.deinit();
        try t_.expect(never.drafted > 0 and went.drafted > 0);
        for (never.out, went.out) |a, b| try t_.expectEqualSlices(u32, a, b);
    }
}
