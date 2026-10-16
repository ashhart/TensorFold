//! A page read's parsing budget: readers check it as they go, and once it runs out they stop and keep what they read.
const std = @import("std");

/// How long one page's parsing may take; with no clock and no steps it never runs out.
pub const Budget = struct {
    io: ?std.Io = null,
    deadline: i96 = 0,
    steps: ?u32 = null, // tests and the fuzz harness: run out after this many checks
    calls: u32 = 0,
    spent: bool = false,

    /// `ms` milliseconds of parsing from now.
    pub fn init(io: std.Io, ms: u32) Budget {
        return .{ .io = io, .deadline = now(io) + @as(i96, ms) * std.time.ns_per_ms };
    }

    /// Whether the budget has run out; the clock is read every 64th check, and once out it stays out.
    pub fn out(b: *Budget) bool {
        if (b.spent) return true;
        b.calls +%= 1;
        if (b.steps) |n| b.spent = b.calls > n;
        if (b.io) |io| {
            if (b.calls % 64 == 1 and now(io) >= b.deadline) b.spent = true;
        }
        return b.spent;
    }
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

test "a budget runs out by its clock or its steps, and stays out" {
    var never: Budget = .{};
    for (0..1000) |_| try std.testing.expect(!never.out());
    var spent: Budget = .init(std.testing.io, 0);
    try std.testing.expect(spent.out() and spent.out());
    var three: Budget = .{ .steps = 3 };
    for (0..3) |_| try std.testing.expect(!three.out());
    try std.testing.expect(three.out() and three.out());
    var long: Budget = .init(std.testing.io, 60_000);
    for (0..1000) |_| try std.testing.expect(!long.out());
}
