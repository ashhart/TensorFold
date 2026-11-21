const std = @import("std");

pub const Slot = struct {
    rows: usize = 0,
    used: usize = 0,
};

pub const Shape = struct { sub: usize, lo: usize, hi: usize };

pub fn prepare(history: []Slot, previous: *?Shape, shape: Shape) void {
    if (previous.* == null or !std.meta.eql(previous.*.?, shape)) @memset(history, .{});
    previous.* = shape;
}

pub fn want(history: Slot, rows: usize, current_used: usize, cap: usize) usize {
    const used = if (rows != 0 and history.rows == rows) history.used else current_used;
    return @min(cap, used +| 16);
}

test "exl3: prefetch history unseen and different rows retain the current-layer fallback" {
    try std.testing.expectEqual(@as(usize, 416), want(.{}, 4096, 400, 512));
    try std.testing.expectEqual(@as(usize, 416), want(.{ .rows = 2048, .used = 50 }, 4096, 400, 512));
    try std.testing.expectEqual(@as(usize, 416), want(.{}, 0, 400, 512));
}

test "exl3: prefetch history uses the target layer's own unique demand including valid zero" {
    var history: [4]Slot = @splat(.{});
    history[2] = .{ .rows = 4096, .used = 80 };
    history[0] = .{ .rows = 4096, .used = 300 };
    try std.testing.expectEqual(@as(usize, 316), want(history[1], 4096, 300, 512));
    try std.testing.expectEqual(@as(usize, 96), want(history[2], 4096, 300, 512));
    history[1] = .{ .rows = 4096, .used = 200 };
    try std.testing.expectEqual(@as(usize, 96), want(history[2], 4096, 200, 512));
    history[2] = .{ .rows = 4096, .used = 0 };
    try std.testing.expectEqual(@as(usize, 16), want(history[2], 4096, 200, 512));
    try std.testing.expectEqual(@as(usize, 216), want(history[2], 2048, 200, 512));
}

test "exl3: prefetch history caps both paths without overflow" {
    for ([_]usize{ 0, 1, 16, 256, 512, std.math.maxInt(usize) }) |cap| {
        try std.testing.expectEqual(cap, want(.{}, 4096, std.math.maxInt(usize), cap));
        try std.testing.expectEqual(cap, want(.{ .rows = 4096, .used = std.math.maxInt(usize) }, 4096, 0, cap));
    }
    for (0..513) |used| {
        try std.testing.expectEqual(@min(@as(usize, 512), used + 16), want(.{}, 4096, used, 512));
    }
}

test "exl3: prefetch history resets for sub-block or rank shape changes" {
    const shape: Shape = .{ .sub = 256, .lo = 0, .hi = 256 };
    var previous: ?Shape = null;
    var history: [4]Slot = @splat(.{ .rows = 4096, .used = 80 });
    prepare(&history, &previous, shape);
    for (history) |entry| try std.testing.expectEqual(@as(usize, 416), want(entry, 4096, 400, 512));
    history[2] = .{ .rows = 4096, .used = 80 };
    prepare(&history, &previous, shape);
    try std.testing.expectEqual(@as(usize, 96), want(history[2], 4096, 400, 512));
    prepare(&history, &previous, .{ .sub = 512, .lo = 0, .hi = 256 });
    try std.testing.expectEqual(@as(usize, 416), want(history[2], 4096, 400, 512));
    history[2] = .{ .rows = 4096, .used = 80 };
    prepare(&history, &previous, .{ .sub = 512, .lo = 256, .hi = 512 });
    try std.testing.expectEqual(@as(usize, 416), want(history[2], 4096, 400, 512));
}
