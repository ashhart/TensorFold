//! A shared override reaches both budgeting and overhead subtraction without changing lone depth prices.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const depth = @import("depth.zig");
const fake = @import("fake.zig");
const a = std.testing.allocator;
const NoCopy = struct {
    fn copy(_: *anyopaque, _: *const depth.Who) bool {
        return false;
    }
};

test "shared head price: a model override reaches shared budgets and overhead, lone rounds keep their measured step" {
    var costs: [8]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 10 };
    for ([_]bool{ false, true }) |override| {
        var cfg = try Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 3, .draft_prior = &.{1}, .mtp_step_ms = 100, .shared_step_ms = if (override) 0 else null, .window_costs = &costs }, 8, 7);
        defer cfg.deinit(a);
        var target: fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        try std.testing.expectEqual(@as(?f64, 110), try e.rule.roundCost(1, null));
        try std.testing.expectEqual(@as(f64, if (override) 0 else 500), e.rule.headMs(&.{ 2, 3 }));
        var states: [2]?depth.State = @splat(null);
        defer for (&states) |*state| if (state.*) |*s| s.deinit(a);
        var who = [_]depth.Who{
            .{ .state = &states[0], .draft_room = 64, .finished = false, .forced = false },
            .{ .state = &states[1], .draft_room = 64, .finished = false, .forced = false },
        };
        var out: [2]i64 = undefined;
        var unused: u8 = 0;
        try e.rule.budgets(&who, .{ .ptr = &unused, .copy = NoCopy.copy }, false, &out);
        const want: [2]i64 = if (override) .{ 3, 3 } else .{ 1, 1 };
        try std.testing.expectEqualSlices(i64, &want, &out);
    }
}
