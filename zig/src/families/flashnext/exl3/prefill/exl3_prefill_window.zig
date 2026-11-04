const std = @import("std");

pub const max_rows = 4096;

pub fn enabled(value: ?[]const u8) bool {
    return if (value) |text| std.mem.eql(u8, text, "1") else false;
}

pub fn admits(free: u64, protected: u64, extra: u64) bool {
    const required = std.math.add(u64, protected, extra) catch return false;
    return extra != 0 and free >= required;
}

pub const Segment = struct { offset: usize, rows: usize };
pub const Step = struct { layer: usize, chunk: usize };

pub const Plan = struct {
    segments: [2]Segment,
    count: usize,
    rows: usize,

    pub fn init(remaining: usize, chunk_rows: usize, alignment: usize) !Plan {
        if (remaining == 0 or alignment == 0 or chunk_rows == 0 or chunk_rows > max_rows or chunk_rows % alignment != 0) return error.InvalidPrefillWindow;
        const first = @min(remaining, chunk_rows);
        const second = @min(remaining - first, chunk_rows);
        return .{ .segments = .{ .{ .offset = 0, .rows = first }, .{ .offset = first, .rows = second } }, .count = if (second == 0) 1 else 2, .rows = first + second };
    }

    pub fn iterator(plan: Plan, layers: usize) Iterator {
        return .{ .count = plan.count, .layers = layers };
    }
};

pub const Iterator = struct {
    count: usize,
    layers: usize,
    layer: usize = 0,
    chunk: usize = 0,

    pub fn next(iterator: *Iterator) ?Step {
        if (iterator.layer >= iterator.layers) return null;
        const step: Step = .{ .layer = iterator.layer, .chunk = iterator.chunk };
        iterator.chunk += 1;
        if (iterator.chunk == iterator.count) {
            iterator.chunk = 0;
            iterator.layer += 1;
        }
        return step;
    }
};

test "exl3: two chunk windows require explicit opt-in and bounded aligned rows" {
    for ([_]?[]const u8{ null, "", "0", "true", "2" }) |value| try std.testing.expect(!enabled(value));
    try std.testing.expect(enabled("1"));
    for ([_]usize{ 0, 1, 255, 4097, 8192 }) |rows| try std.testing.expectError(error.InvalidPrefillWindow, Plan.init(16384, rows, 256));
    try std.testing.expectError(error.InvalidPrefillWindow, Plan.init(0, 4096, 256));
    try std.testing.expectError(error.InvalidPrefillWindow, Plan.init(4096, 4096, 0));
}

test "exl3: two chunk scheduling keeps each layer chronological and retains chunk boundaries" {
    for ([_]usize{ 1, 255, 256, 4095, 4096, 4097, 8192, 8193, 16384, 16641 }) |length| {
        var position: [4]usize = @splat(0);
        var cursor: usize = 0;
        var windows: usize = 0;
        while (cursor < length) {
            const plan = try Plan.init(length - cursor, 4096, 256);
            try std.testing.expect(plan.rows <= 8192);
            var steps = plan.iterator(position.len);
            var previous: ?Step = null;
            while (steps.next()) |step| {
                const segment = plan.segments[step.chunk];
                try std.testing.expectEqual(position[step.layer], cursor + segment.offset);
                if (previous) |before| {
                    if (before.layer == step.layer) try std.testing.expectEqual(before.chunk + 1, step.chunk) else {
                        try std.testing.expectEqual(before.layer + 1, step.layer);
                        try std.testing.expectEqual(@as(usize, 0), step.chunk);
                    }
                }
                position[step.layer] += segment.rows;
                previous = step;
            }
            cursor += plan.rows;
            windows += 1;
        }
        for (position) |end| try std.testing.expectEqual(length, end);
        try std.testing.expectEqual((length + 8191) / 8192, windows);
    }
}

test "exl3: two chunk admission protects all existing reservations and rejects overflow" {
    try std.testing.expect(admits(100, 60, 40));
    try std.testing.expect(!admits(99, 60, 40));
    try std.testing.expect(!admits(100, 101, 1));
    try std.testing.expect(!admits(100, 0, 0));
    try std.testing.expect(!admits(std.math.maxInt(u64), std.math.maxInt(u64), 1));
}
