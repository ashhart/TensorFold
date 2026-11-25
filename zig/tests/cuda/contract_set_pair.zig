//! Two contract sets set up in order: the first one is released when the second one fails to set up.
const std = @import("std");

/// A.init(args) then B.init(args); a failing B leaves nothing of A behind.
pub fn pair(comptime A: type, comptime B: type, args: anytype) !struct { *A, *B } {
    const x = try @call(.auto, A.init, args);
    return .{ x, try @call(.auto, B.init, args) };
}

const Held = struct {
    gpa: std.mem.Allocator,
    fn init(gpa: std.mem.Allocator) !*Held {
        const h = try gpa.create(Held);
        h.gpa = gpa;
        return h;
    }
    fn deinit(h: *Held) void {
        h.gpa.destroy(h);
    }
};

const Refused = struct {
    fn init(gpa: std.mem.Allocator) !*Refused {
        _ = gpa;
        return error.SetupFailed;
    }
};

test "a set that fails to set up releases the one before it (the testing allocator sees no leak)" {
    try std.testing.expectError(error.SetupFailed, pair(Held, Refused, .{std.testing.allocator}));
    const a, const b = try pair(Held, Held, .{std.testing.allocator});
    a.deinit();
    b.deinit();
}
