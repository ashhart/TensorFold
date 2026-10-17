//! Bounded frame work shared by generation backends; one ready lane advances at each round boundary.
const std = @import("std");
pub const capacity = 8;
pub const Stage = enum { free, prompt, frame };
pub const Work = struct { lane: usize, stage: Stage, cancelled: bool };

pub const Queue = struct {
    stages: [capacity]Stage = @splat(.free),
    cancelled: [capacity]bool = @splat(false),
    cursor: usize = 0,
    lanes: usize,

    pub fn init(lanes: usize) !Queue {
        if (lanes == 0 or lanes > capacity) return error.BadLaneCount;
        return .{ .lanes = lanes };
    }

    pub fn admit(q: *Queue) error{Busy}!usize {
        for (q.stages[0..q.lanes], 0..) |stage, lane| if (stage == .free) {
            q.stages[lane] = .prompt;
            q.cancelled[lane] = false;
            return lane;
        };
        return error.Busy;
    }

    pub fn next(q: *Queue) ?Work {
        for (0..q.lanes) |_| {
            const lane = q.cursor;
            q.cursor = (lane + 1) % q.lanes;
            if (q.stages[lane] != .free) return .{ .lane = lane, .stage = q.stages[lane], .cancelled = q.cancelled[lane] };
        }
        return null;
    }

    pub fn release(q: *Queue, lane: usize) void {
        q.stages[lane] = .free;
        q.cancelled[lane] = false;
    }
};

test "four and more requests share prompt and frame rounds, cancellation and reuse" {
    for ([_]usize{ 4, 8 }) |n| {
        var q = try Queue.init(n);
        for (0..n) |i| try std.testing.expectEqual(i, try q.admit());
        try std.testing.expectError(error.Busy, q.admit());
        for (0..n) |i| {
            const w = q.next().?;
            try std.testing.expectEqual(i, w.lane);
            try std.testing.expectEqual(Stage.prompt, w.stage);
            q.stages[i] = .frame;
        }
        q.cancelled[1] = true;
        for (0..n) |i| {
            const w = q.next().?;
            try std.testing.expectEqual(i, w.lane);
            try std.testing.expectEqual(i == 1, w.cancelled);
            q.release(i);
        }
        try std.testing.expect(q.next() == null);
        try std.testing.expectEqual(@as(usize, 0), try q.admit());
        try std.testing.expect(!q.next().?.cancelled);
    }
}
