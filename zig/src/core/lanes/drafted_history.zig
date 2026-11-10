//! A restored target can supply the drafter's retained history before newly computed prompt rows.
const std = @import("std");

pub fn from(x: anytype, stream: anytype) u64 {
    const history = if (x.target.restored_features) x.drafter.facts().history_rows else 0;
    return stream.cached -| history;
}

test "cached draft history: ordinary targets never ask for history and restored reads saturate at zero" {
    const Probe = struct {
        calls: usize = 0,
        pub fn facts(self: *@This()) struct { history_rows: u32 } {
            self.calls += 1;
            return .{ .history_rows = 4 };
        }
    };
    var head: Probe = .{};
    var wrapper: struct { target: struct { restored_features: bool }, drafter: *Probe } = .{
        .target = .{ .restored_features = false },
        .drafter = &head,
    };
    for ([_]u32{ 0, 1, 4, 8 }) |cached| try std.testing.expectEqual(@as(u64, cached), from(&wrapper, .{ .cached = cached }));
    try std.testing.expectEqual(@as(usize, 0), head.calls);
    wrapper.target.restored_features = true;
    for ([_]u32{ 0, 1, 4, 8 }) |cached| try std.testing.expectEqual(@as(u64, cached -| 4), from(&wrapper, .{ .cached = cached }));
    try std.testing.expectEqual(@as(usize, 4), head.calls);
}
