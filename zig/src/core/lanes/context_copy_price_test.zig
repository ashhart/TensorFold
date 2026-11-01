//! A full head proposal must not hide a useful copy after its weak tail is priced away.
const std = @import("std");
const lanes = @import("lanes.zig");
const win = @import("windows.zig");
const Probe = struct {
    calls: usize = 0,
    fn chances(ptr: *anyopaque, _: *lanes.Stream, out: []f64) !bool {
        const p: *Probe = @ptrCast(@alignCast(ptr));
        p.calls += 1;
        const values = [_]f64{ 0.9, 0.005, 0.001 };
        @memcpy(out, values[0..out.len]);
        return true;
    }
};
test "context copy: price away a full head's weak tail before offering a useful continuation" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 4, .batch_rows = 4, .drafts = 3, .mtp = true, .speculate = true, .draft_probabilities = true, .context_copies = true, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 2, .ms = 10.1 }, .{ .width = 3, .ms = 10.3 }, .{ .width = 4, .ms = 10.5 } } }, 4, 3);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var probe: Probe = .{};
    var vtable = target.backend().vtable.*;
    vtable.probabilities = Probe.chances;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, .{ .ptr = &probe, .vtable = &vtable }, clock.clock());
    defer e.deinit();
    e.alone = true;
    var stream = try lanes.Stream.init(a, .{ .id = "copy-room", .prompt = &.{ 1, 2 }, .max_new = 32 });
    defer stream.deinit(a);
    stream.proposal_depth = 3;
    stream.next = .{ .count = 3, .tokens = try a.dupe(u32, &.{ 10, 11, 12 }), .parents = try a.dupe(i32, &.{ -1, 0, 1 }) };
    var plans = [_]win.Plan{try win.plan(&e, &stream, .{ .tokens = &.{ 10, 20, 21 } })};
    try std.testing.expectEqual(@as(usize, 3), plans[0].count());
    try win.allocate(&e, &plans);
    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 21 }, plans[0].tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, plans[0].parents.?);
    try std.testing.expect(plans[0].context_copy and plans[0].branch_from == 2);
    try std.testing.expectEqual(@as(usize, 2), probe.calls);
    try std.testing.expect(plans[0].count() + 1 <= cfg.family_width);
}
