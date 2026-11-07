//! Developer-only BF16 residual boundaries from the production prompt forward.
const std = @import("std");
pub const Trace = struct {
    ctx: *anyopaque,
    capture: *const fn (*anyopaque, usize, u64, usize) anyerror!void,
};

/// Boundary 0 is embedding; 1..layers are residuals; layers+1 is final normalization.
pub fn emit(e: anytype, boundary: usize, ptr: u64, rows: usize) !void {
    if (e.hidden_trace) |trace| try trace.capture(trace.ctx, boundary, ptr, rows);
}

test "BF16 scoring: hidden trace is opt-in and propagates capture failure" {
    const Recorder = struct {
        hidden_trace: ?Trace = null,
        calls: usize = 0,
        fn capture(ctx: *anyopaque, boundary: usize, ptr: u64, rows: usize) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            try std.testing.expectEqual(@as(usize, 7), boundary);
            try std.testing.expectEqual(@as(u64, 123), ptr);
            try std.testing.expectEqual(@as(usize, 32), rows);
            self.calls += 1;
            return error.PlantedCaptureFailure;
        }
    };
    var e: Recorder = .{};
    try emit(&e, 7, 123, 32);
    try std.testing.expectEqual(@as(usize, 0), e.calls);
    e.hidden_trace = .{ .ctx = &e, .capture = Recorder.capture };
    try std.testing.expectError(error.PlantedCaptureFailure, emit(&e, 7, 123, 32));
    try std.testing.expectEqual(@as(usize, 1), e.calls);
}
