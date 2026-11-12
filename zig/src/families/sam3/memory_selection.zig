const std = @import("std");

fn List(comptime T: type, comptime capacity: usize) type {
    return struct {
        items: [capacity]T = undefined,
        len: usize = 0,

        fn appendAssumeCapacity(self: *@This(), item: T) void {
            std.debug.assert(self.len < capacity);
            self.items[self.len] = item;
            self.len += 1;
        }

        fn insertAssumeCapacity(self: *@This(), at: usize, item: T) void {
            std.debug.assert(at <= self.len and self.len < capacity);
            std.mem.copyBackwards(T, self.items[at + 1 .. self.len + 1], self.items[at..self.len]);
            self.items[at] = item;
            self.len += 1;
        }

        pub fn get(self: @This(), at: usize) T {
            return self.constSlice()[at];
        }

        pub fn constSlice(self: *const @This()) []const T {
            return self.items[0..self.len];
        }
    };
}

pub const Direction = enum { forward, reverse };
pub const Spatial = struct { history: usize, temporal: usize };
pub const Pointer = struct { history: usize, relative: usize };
pub const Plan = struct {
    spatial: List(Spatial, 10) = .{},
    pointers: List(Pointer, 19) = .{},
};

pub fn distance(a: usize, b: usize) usize {
    return @max(a, b) - @min(a, b);
}

fn past(frame: usize, index: usize, direction: Direction) bool {
    return if (direction == .forward) frame < index else frame > index;
}

fn previous(index: usize, frames: usize, direction: Direction, delta: usize) ?usize {
    return switch (direction) {
        .forward => if (delta <= index) index - delta else null,
        .reverse => if (index < frames and delta < frames - index) index + delta else null,
    };
}

fn find(history: anytype, index: usize, conditioning: []const usize) ?usize {
    for (history, 0..) |item, at| if (item.index == index) {
        return if (std.mem.indexOfScalar(usize, conditioning, at) == null) at else null;
    };
    return null;
}

fn conditioningFrames(history: anytype, index: usize) List(usize, 4) {
    var out: List(usize, 4) = .{};
    var count: usize = 0;
    for (history) |item| count += @intFromBool(item.conditioning);
    if (count <= 4) {
        for (history, 0..) |item, i| if (item.conditioning) out.appendAssumeCapacity(i);
        return out;
    }
    for ([_]bool{ true, false }) |before| {
        var closest: ?usize = null;
        for (history, 0..) |item, i| {
            if (!item.conditioning or (item.index < index) != before) continue;
            if (closest == null or distance(item.index, index) < distance(history[closest.?].index, index)) closest = i;
        }
        if (closest) |i| out.appendAssumeCapacity(i);
    }
    while (out.len < 4) {
        var closest: ?usize = null;
        for (history, 0..) |item, i| {
            if (!item.conditioning or std.mem.indexOfScalar(usize, out.constSlice(), i) != null) continue;
            if (closest == null or distance(item.index, index) < distance(history[closest.?].index, index)) closest = i;
        }
        out.appendAssumeCapacity(closest.?);
    }
    return out;
}

fn filtered(history: anytype, index: usize, frames: usize, direction: Direction) List(usize, 16) {
    var nearest: List(usize, 15) = .{};
    const maximum = @min(frames, 16) - 1;
    for (history) |item| {
        if (item.conditioning or !past(item.index, index, direction) or !(item.score > 0.01) or
            (direction == .forward and item.index == 0)) continue;
        var at: usize = 0;
        while (at < nearest.len and distance(nearest.get(at), index) < distance(item.index, index)) : (at += 1) {}
        if (at >= maximum) continue;
        if (nearest.len == maximum) nearest.len -= 1;
        nearest.insertAssumeCapacity(at, item.index);
    }
    var out: List(usize, 16) = .{};
    var i = nearest.len;
    while (i != 0) {
        i -= 1;
        out.appendAssumeCapacity(nearest.get(i));
    }
    if (previous(index, frames, direction, 1)) |last| {
        if (std.mem.indexOfScalar(usize, out.constSlice(), last) == null) out.appendAssumeCapacity(last);
    }
    return out;
}

pub fn select(history: anytype, index: usize, frames: usize, direction: Direction, multiplex: bool) !Plan {
    if (frames == 0 or index >= frames) return error.FrameOutOfRange;
    const conditioning = conditioningFrames(history, index);
    if (conditioning.len == 0) return error.MissingConditioningFrame;
    var out: Plan = .{};
    const valid: List(usize, 16) = if (multiplex) .{} else filtered(history, index, frames, direction);
    for (conditioning.constSlice()) |at| {
        const item = history[at];
        const delta = distance(item.index, index);
        const temporal = if (multiplex and past(item.index, index, direction) and delta < 7) 6 - delta else 6;
        if (item.spatial_live) out.spatial.appendAssumeCapacity(.{ .history = at, .temporal = temporal });
        if (past(item.index, index, direction) or item.index == index)
            out.pointers.appendAssumeCapacity(.{ .history = at, .relative = delta });
    }
    var rank: usize = 6;
    while (rank != 0) : (rank -= 1) {
        const frame = if (multiplex) previous(index, frames, direction, rank) else if (rank <= valid.len) valid.get(valid.len - rank) else null;
        const at = find(history, frame orelse continue, conditioning.constSlice()) orelse continue;
        if (history[at].spatial_live) out.spatial.appendAssumeCapacity(.{ .history = at, .temporal = rank - 1 });
    }
    const limit = @min(frames, 16);
    for (1..limit) |delta| {
        if (!multiplex and delta >= valid.len) break;
        const frame = if (multiplex) previous(index, frames, direction, delta) else valid.get(valid.len - delta);
        const at = find(history, frame orelse continue, conditioning.constSlice()) orelse continue;
        out.pointers.appendAssumeCapacity(.{ .history = at, .relative = delta });
    }
    return out;
}

pub const Retention = struct { keep: bool, spatial: bool };

pub fn retention(history: anytype, at: usize, index: usize, frames: usize, multiplex: bool) Retention {
    const item = history[at];
    if (item.conditioning or item.index == index) return .{ .keep = true, .spatial = true };
    const delta = distance(index, item.index);
    if (multiplex) return .{ .keep = delta < 16, .spatial = delta < 6 };
    if (!(item.score > 0.01)) return .{ .keep = false, .spatial = false };
    var newer: usize = 0;
    var latest_low: usize = 0;
    for (history) |other| {
        if (other.conditioning) continue;
        if (distance(index, other.index) < delta and other.score > 0.01) newer += 1;
        if (other.index == index and !(other.score > 0.01)) latest_low = 1;
    }
    return .{ .keep = newer < @min(frames, 16) - 1, .spatial = newer + latest_low < 6 };
}

const Sample = struct { index: usize, conditioning: bool = false, score: f32 = 1, spatial_live: bool = true };

test "SAM memory plans match captured pointer counts and temporal positions" {
    var history: [17]Sample = undefined;
    for (&history, 0..) |*item, i| item.* = .{ .index = i, .conditioning = i == 0 };
    const sam3 = try select(&history, 17, 120, .forward, false);
    const sam31 = try select(&history, 17, 120, .forward, true);
    try std.testing.expectEqual(@as(usize, 15), sam3.pointers.len);
    try std.testing.expectEqual(@as(usize, 16), sam31.pointers.len);
    for ([_]Plan{ sam3, sam31 }) |plan| {
        try std.testing.expectEqual(@as(usize, 7), plan.spatial.len);
        try std.testing.expectEqual(Spatial{ .history = 0, .temporal = 6 }, plan.spatial.get(0));
        for (plan.spatial.constSlice()[1..], 0..) |item, i| try std.testing.expectEqual(Spatial{ .history = 11 + i, .temporal = 5 - i }, item);
        try std.testing.expectEqual(Pointer{ .history = 0, .relative = 17 }, plan.pointers.get(0));
        for (plan.pointers.constSlice()[1..], 1..) |item, rank| try std.testing.expectEqual(Pointer{ .history = 17 - rank, .relative = rank }, item);
    }
    const early = try select(history[0..2], 2, 120, .forward, false);
    try std.testing.expectEqual(@as(usize, 1), early.pointers.len);
    try std.testing.expectEqual(Spatial{ .history = 1, .temporal = 0 }, early.spatial.get(1));
    const multiplex_early = try select(history[0..2], 2, 120, .forward, true);
    try std.testing.expectEqual(Spatial{ .history = 0, .temporal = 4 }, multiplex_early.spatial.get(0));
}

test "confidence gaps use selected ranks and always include the previous frame in either direction" {
    var history: [40]Sample = undefined;
    for ([_]Direction{ .forward, .reverse }) |direction| {
        for (&history, 0..) |*item, i| item.* = .{
            .index = if (direction == .forward) i else 119 - i,
            .conditioning = i == 0,
            .score = if (i < 15) 1 else 0,
        };
        const current: usize = if (direction == .forward) 40 else 79;
        const plan = try select(&history, current, 120, direction, false);
        try std.testing.expectEqual(@as(usize, 7), plan.spatial.len);
        const expected = [_]usize{ 0, 10, 11, 12, 13, 14, 39 };
        for (plan.spatial.constSlice(), expected) |item, at| try std.testing.expectEqual(at, item.history);
        try std.testing.expectEqual(Pointer{ .history = 39, .relative = 1 }, plan.pointers.get(1));
        try std.testing.expectEqual(Pointer{ .history = 14, .relative = 2 }, plan.pointers.get(2));
        const latest = history[39].index;
        try std.testing.expect(retention(&history, 1, latest, 120, false).keep);
        try std.testing.expect(!retention(&history, 1, latest, 120, true).keep);
        try std.testing.expect(retention(&history, 10, latest, 120, false).spatial);
        try std.testing.expect(!retention(&history, 9, latest, 120, false).spatial);
        try std.testing.expect(!retention(&history, 38, latest, 120, false).keep);
        try std.testing.expect(retention(&history, 39, latest, 120, false).keep);
    }
}

test "conditioning selection keeps the closest frames on both sides" {
    const history = [_]Sample{
        .{ .index = 0, .conditioning = true },
        .{ .index = 9, .conditioning = true },
        .{ .index = 11, .conditioning = true },
        .{ .index = 15, .conditioning = true },
        .{ .index = 12, .conditioning = true },
    };
    const plan = try select(&history, 10, 20, .forward, false);
    try std.testing.expectEqual(@as(usize, 4), plan.spatial.len);
    const expected = [_]usize{ 1, 2, 4, 3 };
    for (plan.spatial.constSlice(), expected) |item, at| try std.testing.expectEqual(at, item.history);
    try std.testing.expectEqual(@as(usize, 1), plan.pointers.len);
    try std.testing.expectEqual(@as(usize, 1), plan.pointers.get(0).history);
}

test "memory confidence threshold is strict and excludes nonfinite scores" {
    const history = [_]Sample{
        .{ .index = 0, .conditioning = true },
        .{ .index = 1, .score = 0.01 },
        .{ .index = 2, .score = std.math.nan(f32) },
        .{ .index = 3, .score = 0.02 },
        .{ .index = 4, .score = 0 },
    };
    const plan = try select(&history, 5, 20, .forward, false);
    try std.testing.expectEqual(@as(usize, 3), plan.spatial.len);
    try std.testing.expectEqual(Spatial{ .history = 3, .temporal = 1 }, plan.spatial.get(1));
    try std.testing.expectEqual(Spatial{ .history = 4, .temporal = 0 }, plan.spatial.get(2));
    try std.testing.expectEqual(@as(usize, 2), plan.pointers.len);
    try std.testing.expectEqual(Pointer{ .history = 4, .relative = 1 }, plan.pointers.get(1));
}

test "bounded history preserves unpruned memory plans through confidence gaps in either direction" {
    var full: [80]Sample = undefined;
    var kept: [80]Sample = undefined;
    for ([_]Direction{ .forward, .reverse }) |direction| for ([_]bool{ false, true }) |multiplex| {
        for (&full, 0..) |*item, i| item.* = .{
            .index = if (direction == .forward) i else full.len - 1 - i,
            .conditioning = i == 0,
            .score = if (i % 31 < 12) 0.5 else if (i % 7 == 0) std.math.nan(f32) else 0.01,
        };
        kept[0] = full[0];
        var count: usize = 1;
        for (1..full.len) |step| {
            const index = full[step].index;
            const expected = try select(full[0..step], index, full.len, direction, multiplex);
            const actual = try select(kept[0..count], index, full.len, direction, multiplex);
            try std.testing.expectEqual(expected.spatial.len, actual.spatial.len);
            try std.testing.expectEqual(expected.pointers.len, actual.pointers.len);
            for (expected.spatial.constSlice(), actual.spatial.constSlice()) |x, y| {
                try std.testing.expectEqual(full[x.history].index, kept[y.history].index);
                try std.testing.expectEqual(x.temporal, y.temporal);
            }
            for (expected.pointers.constSlice(), actual.pointers.constSlice()) |x, y| {
                try std.testing.expectEqual(full[x.history].index, kept[y.history].index);
                try std.testing.expectEqual(x.relative, y.relative);
            }
            kept[count] = full[step];
            count += 1;
            var at: usize = 0;
            while (at < count) {
                const keep = retention(kept[0..count], at, index, full.len, multiplex);
                if (keep.keep) {
                    kept[at].spatial_live = kept[at].spatial_live and keep.spatial;
                    at += 1;
                } else {
                    std.mem.copyForwards(Sample, kept[at .. count - 1], kept[at + 1 .. count]);
                    count -= 1;
                }
            }
            try std.testing.expect(count <= 17);
        }
    };
}
