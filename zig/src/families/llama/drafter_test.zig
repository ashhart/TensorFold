//! The Llama drafter's choice between its request's copies, its own drafts and a plain round.
const std = @import("std");
const lanes = @import("lanes");
const drafter = @import("drafter.zig");
const Drafter = drafter.Drafter;
const Request = drafter.Request;
const prior_kept = drafter.prior_kept;

test "llama drafter: a stream stands aside where drafts cost, and a probe drafts again" {
    var slots = [_]Drafter.Slot{.{ .taken = true }};
    var d: Drafter = undefined;
    d.slots = &slots;
    d.step_ms = 30; // a dear drafter step
    d.kept = 0;
    d.tried = 0;
    d.rounds = .{};
    for (0..20) |i| d.rounds.add(@floatFromInt(2 + i % 6), 60 + 5 * @as(f64, @floatFromInt(2 + i % 6)));
    var r: Request = .{ .d = &d, .gpa = std.testing.allocator, .slot = 0, .eos = &.{}, .kept = 0.5, .tried = 5 };
    try std.testing.expectEqual(@as(usize, 0), r.want(15, false)); // one in ten lands: a plain round lands more
    try std.testing.expect(r.want(15, true) >= 1); // a probe drafts all the same
    // a stood-aside round is the lane core's plain step: no acceptance history
    r.asked = .{ .len = 10, .max_draft = 15 };
    r.round(1, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), r.kept, 1e-9);
    try std.testing.expectEqual(@as(u64, 0), r.offered);
    // a landing drafter drafts on its own
    r.kept = 9;
    r.tried = 10;
    d.step_ms = 3;
    try std.testing.expect(r.want(15, false) >= 4);
}

test "llama drafter: a request's own copies answer first where the text repeats, judged as copies, not as drafts" {
    var d: Drafter = undefined;
    d.options.copy_enter_match = 4;
    var r: Request = .{ .d = &d, .gpa = std.testing.allocator, .slot = 0, .eos = &.{}, .copies = try lanes.SuffixLookup.init(std.testing.allocator, .{ .min_match = 4 }) };
    defer {
        r.out.deinit(r.gpa);
        r.copies.deinit();
    }
    const context = [_]u32{ 7, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4 };
    const drafts = try r.propose(&context, 4);
    try std.testing.expect(r.copied);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 8, 9 }, drafts);
    r.round(5, 4);
    try std.testing.expectEqual(@as(i64, 4), r.copies.accepted_tokens);
    try std.testing.expectEqual(@as(u64, 0), r.offered); // the drafter's own history untouched
    try std.testing.expectApproxEqAbs(prior_kept, r.kept, 1e-9);
}
