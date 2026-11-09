//! An input-fit allocation failure resumes the open output fit before any live router upload.
const std = @import("std");
const host = @import("host_test.zig");
const learner = @import("learner.zig");
const Example = @import("lesson_examples.zig").Example;

test "round wordings: a failed router fit continues the existing outputs and Adam" {
    const gpa = std.testing.allocator;
    const H = host.Host(@import("families.zig").nemotron_h);
    var seen: H.Seen = .{};
    var l = learner.Learner(H).init(gpa, std.testing.io, &seen);
    defer l.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const train = [_]Example{try host.qa(a, 7, 0)};
    const old = [_]Example{try host.qa(a, 7, 1)};
    const fresh = [_]Example{try host.qa(a, 7, 2)};
    try l.begin(.{ .train = &train, .asked = &old });
    try host.finishRound(&l);
    const p = &l.pockets;
    const weights = @import("pocket_mark.zig").hash(p);
    const optimizer = try a.dupe(f32, p.v.adam.items);
    const owned = l.owned;
    try l.begin(.{ .train = &train, .asked = &fresh, .more = true, .refresh_wordings = true });
    for (0..10_000) |_| {
        if (p.phase == .fit) break;
        const step = l.step();
        if (step.report) |report| if (report == .failed) return error.ReadFailed;
    } else return error.ReadDidNotFinish;
    var continued = p.v;
    continued.restart();
    const expected = host.optimizerHash(&continued);
    // The staging copies succeed; fitLayer's first temporary input-row allocation fails.
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 3 });
    p.gpa = failing.allocator();
    const step = l.step();
    p.gpa = gpa;
    try std.testing.expect(failing.has_induced_failure);
    if (step.report) |report| try std.testing.expect(report != .failed);
    try std.testing.expect(!step.done and p.phase == .values and !p.refreshing);
    try std.testing.expectEqual(weights, @import("pocket_mark.zig").hash(p));
    try std.testing.expectEqualSlices(f32, optimizer, p.v.adam.items);
    try std.testing.expectEqual(expected, host.optimizerHash(&p.v));
    try std.testing.expect(l.owned.train.ptr == owned.train.ptr and l.owned.held.ptr == owned.held.ptr);
    try host.finishRound(&l);
    try l.begin(.{ .undo = true });
    try host.finishRound(&l);
    try std.testing.expect(p.empty());
}
