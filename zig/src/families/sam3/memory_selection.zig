const std = @import("std");

fn Fixed(comptime T: type, comptime capacity: usize) type {
    return struct {
        items: [capacity]T = undefined,
        len: usize = 0,

        fn push(self: *@This(), value: T) void {
            std.debug.assert(self.len != capacity);
            self.items[self.len] = value;
            self.len += 1;
        }

        pub fn get(self: @This(), at: usize) T {
            std.debug.assert(at < self.len);
            return self.items[at];
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
    spatial: Fixed(Spatial, 10) = .{},
    pointers: Fixed(Pointer, 19) = .{},
};

pub fn distance(a: usize, b: usize) usize {
    return if (a >= b) a - b else b - a;
}

const Timeline = struct {
    current: usize,
    frames: usize,
    direction: Direction,

    fn age(self: Timeline, frame_index: usize) ?usize {
        const earlier = switch (self.direction) {
            .forward => frame_index < self.current,
            .reverse => frame_index > self.current,
        };
        return if (earlier) distance(frame_index, self.current) else null;
    }

    fn frame(self: Timeline, age_: usize) ?usize {
        return switch (self.direction) {
            .forward => if (age_ <= self.current) self.current - age_ else null,
            .reverse => if (age_ < self.frames - self.current) self.current + age_ else null,
        };
    }
};

fn anchors(history: anytype, current: usize) Fixed(usize, 4) {
    var all: usize = 0;
    for (history) |entry| all += @intFromBool(entry.conditioning);
    var selected: Fixed(usize, 4) = .{};
    if (all <= 4) {
        for (history, 0..) |entry, row| if (entry.conditioning) selected.push(row);
        return selected;
    }
    var sides: [2]?usize = @splat(null);
    for (history, 0..) |entry, row| {
        if (!entry.conditioning) continue;
        const side: usize = @intFromBool(entry.index >= current);
        const incumbent = sides[side];
        if (incumbent == null or distance(entry.index, current) < distance(history[incumbent.?].index, current))
            sides[side] = row;
    }
    for (sides) |row| if (row) |at| selected.push(at);
    while (selected.len < 4) {
        var next: ?usize = null;
        for (history, 0..) |entry, row| {
            if (!entry.conditioning or std.mem.indexOfScalar(usize, selected.constSlice(), row) != null) continue;
            if (next == null or distance(entry.index, current) < distance(history[next.?].index, current)) next = row;
        }
        selected.push(next.?);
    }
    return selected;
}

const Ranked = struct {
    row: usize,
    age: usize,

    fn older(_: void, a: Ranked, b: Ranked) bool {
        return a.age > b.age or (a.age == b.age and a.row < b.row);
    }
};

fn confidentFrames(history: anytype, timeline: Timeline) Fixed(usize, 16) {
    var pool: Fixed(Ranked, 15) = .{};
    const capacity = @min(timeline.frames, 16) - 1;
    for (history, 0..) |entry, row| {
        if (entry.conditioning or !(entry.score > 0.01)) continue;
        if (timeline.direction == .forward and entry.index == 0) continue;
        const age = timeline.age(entry.index) orelse continue;
        if (capacity == 0) break;
        const candidate = Ranked{ .row = row, .age = age };
        if (pool.len < capacity) {
            pool.push(candidate);
        } else if (Ranked.older({}, pool.items[0], candidate)) {
            pool.items[0] = candidate;
        }
        std.mem.sort(Ranked, pool.items[0..pool.len], {}, Ranked.older);
    }
    var selected: Fixed(usize, 16) = .{};
    for (pool.constSlice()) |entry| selected.push(history[entry.row].index);
    if (timeline.frame(1)) |last| {
        if (std.mem.indexOfScalar(usize, selected.constSlice(), last) == null) selected.push(last);
    }
    return selected;
}

fn historyRow(history: anytype, frame: ?usize, excluded: []const usize) ?usize {
    const wanted = frame orelse return null;
    for (history, 0..) |entry, row| {
        if (entry.index == wanted and std.mem.indexOfScalar(usize, excluded, row) == null) return row;
    }
    return null;
}

pub fn select(history: anytype, index: usize, frames: usize, direction: Direction, multiplex: bool) !Plan {
    if (frames == 0 or index >= frames) return error.FrameOutOfRange;
    const timeline = Timeline{ .current = index, .frames = frames, .direction = direction };
    const conditioning = anchors(history, index);
    if (conditioning.len == 0) return error.MissingConditioningFrame;
    var out: Plan = .{};
    const valid: Fixed(usize, 16) = if (multiplex) .{} else confidentFrames(history, timeline);
    for (conditioning.constSlice()) |at| {
        const item = history[at];
        const delta = distance(item.index, index);
        const temporal = if (multiplex and timeline.age(item.index) != null and delta < 7) 6 - delta else 6;
        if (item.spatial_live) out.spatial.push(.{ .history = at, .temporal = temporal });
        if (timeline.age(item.index) != null or item.index == index) out.pointers.push(.{ .history = at, .relative = delta });
    }
    var rank: usize = 6;
    while (rank != 0) : (rank -= 1) {
        const frame = if (multiplex) timeline.frame(rank) else if (rank <= valid.len) valid.get(valid.len - rank) else null;
        const at = historyRow(history, frame, conditioning.constSlice()) orelse continue;
        if (history[at].spatial_live) out.spatial.push(.{ .history = at, .temporal = rank - 1 });
    }
    const limit = @min(frames, 16);
    for (1..limit) |delta| {
        if (!multiplex and delta >= valid.len) break;
        const frame = if (multiplex) timeline.frame(delta) else valid.get(valid.len - delta);
        const at = historyRow(history, frame, conditioning.constSlice()) orelse continue;
        out.pointers.push(.{ .history = at, .relative = delta });
    }
    return out;
}

pub const Retention = struct { keep: bool, spatial: bool };

pub fn retention(history: anytype, at: usize, index: usize, frames: usize, multiplex: bool) Retention {
    const target = history[at];
    if (target.conditioning or target.index == index) return .{ .keep = true, .spatial = true };
    const age = distance(target.index, index);
    if (multiplex) return .{ .keep = age < 16, .spatial = age < 6 };
    if (!(target.score > 0.01) or frames == 0) return .{ .keep = false, .spatial = false };
    var rank: usize = 0;
    var reserve_latest = false;
    for (history) |entry| {
        if (entry.conditioning) continue;
        if (entry.index == index and !(entry.score > 0.01)) reserve_latest = true;
        rank += @intFromBool(entry.score > 0.01 and distance(entry.index, index) < age);
    }
    return .{ .keep = rank + 1 < @min(frames, 16), .spatial = rank + @intFromBool(reserve_latest) < 6 };
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
