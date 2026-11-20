const std = @import("std");

pub const Direction = enum { forward, reverse };
pub const GridSlot = struct { entry: usize, time_slot: usize };
pub const PointerSlot = struct { entry: usize, offset: usize };
pub const max_grid_slots = 10;
pub const max_pointer_slots = 19;

const prompted_cap = 4;
const recency = 6;
const far_time_slot = 6;
const pointer_horizon = 16;
const keep_distance = 16;
const confidence_threshold: f32 = 0.01;

pub const Plan = struct {
    grid: [max_grid_slots]GridSlot = undefined,
    grid_len: usize = 0,
    pointer: [max_pointer_slots]PointerSlot = undefined,
    pointer_len: usize = 0,

    pub fn gridSlots(self: *const Plan) []const GridSlot {
        return self.grid[0..self.grid_len];
    }

    pub fn pointerSlots(self: *const Plan) []const PointerSlot {
        return self.pointer[0..self.pointer_len];
    }

    fn addGrid(self: *Plan, entry: usize, time_slot: usize) void {
        self.grid[self.grid_len] = .{ .entry = entry, .time_slot = time_slot };
        self.grid_len += 1;
    }

    fn addPointer(self: *Plan, entry: usize, offset: usize) void {
        self.pointer[self.pointer_len] = .{ .entry = entry, .offset = offset };
        self.pointer_len += 1;
    }
};

pub const Keep = struct { entry: bool, grid: bool };

pub fn horizon(frames: usize) usize {
    return @min(frames, pointer_horizon);
}

fn distance(a: usize, b: usize) usize {
    return if (a > b) a - b else b - a;
}

fn confident(confidence: f32) bool {
    return confidence > confidence_threshold;
}

const Position = struct {
    current: usize,
    frames: usize,
    direction: Direction,

    fn past(self: Position, frame: usize) bool {
        return switch (self.direction) {
            .forward => frame < self.current,
            .reverse => frame > self.current,
        };
    }

    fn back(self: Position, steps: usize) ?usize {
        return switch (self.direction) {
            .forward => if (steps <= self.current) self.current - steps else null,
            .reverse => if (self.current + steps <= self.frames - 1) self.current + steps else null,
        };
    }
};

const Prompted = struct {
    items: [prompted_cap]usize = undefined,
    len: usize = 0,

    fn slice(self: *const Prompted) []const usize {
        return self.items[0..self.len];
    }

    fn has(self: *const Prompted, index: usize) bool {
        return std.mem.indexOfScalar(usize, self.slice(), index) != null;
    }

    fn add(self: *Prompted, index: usize) void {
        self.items[self.len] = index;
        self.len += 1;
    }
};

const Side = enum { before, after, any };

fn nearestPrompted(entries: anytype, current: usize, chosen: *const Prompted, side: Side) ?usize {
    var best: ?usize = null;
    var best_distance: usize = 0;
    for (entries, 0..) |entry, i| {
        if (!entry.prompted or chosen.has(i)) continue;
        switch (side) {
            .before => if (entry.frame >= current) continue,
            .after => if (entry.frame < current) continue,
            .any => {},
        }
        const d = distance(entry.frame, current);
        if (best == null or d < best_distance) {
            best = i;
            best_distance = d;
        }
    }
    return best;
}

fn promptedEntries(entries: anytype, current: usize) Prompted {
    var chosen: Prompted = .{};
    var count: usize = 0;
    for (entries) |entry| count += @intFromBool(entry.prompted);
    if (count <= prompted_cap) {
        for (entries, 0..) |entry, i| if (entry.prompted) chosen.add(i);
        return chosen;
    }
    if (nearestPrompted(entries, current, &chosen, .before)) |i| chosen.add(i);
    if (nearestPrompted(entries, current, &chosen, .after)) |i| chosen.add(i);
    while (chosen.len < prompted_cap) chosen.add(nearestPrompted(entries, current, &chosen, .any) orelse break);
    return chosen;
}

const Trail = struct {
    frames: [pointer_horizon]usize = undefined,
    len: usize = 0,

    fn contains(self: *const Trail, frame: usize) bool {
        return std.mem.indexOfScalar(usize, self.frames[0..self.len], frame) != null;
    }

    fn append(self: *Trail, frame: usize) void {
        self.frames[self.len] = frame;
        self.len += 1;
    }

    fn fromEnd(self: *const Trail, rank: usize) ?usize {
        return if (rank >= 1 and rank <= self.len) self.frames[self.len - rank] else null;
    }
};

fn confidentTrail(entries: anytype, at: Position) Trail {
    const cap = horizon(at.frames) -| 1;
    var nearest: [pointer_horizon - 1]usize = undefined;
    var count: usize = 0;
    for (entries) |entry| {
        if (entry.prompted or !confident(entry.confidence) or !at.past(entry.frame)) continue;
        if (at.direction == .forward and entry.frame == 0) continue;
        const d = distance(entry.frame, at.current);
        var position = count;
        while (position > 0 and distance(nearest[position - 1], at.current) > d) position -= 1;
        if (position >= cap) continue;
        count = @min(count + 1, cap);
        var j = count - 1;
        while (j > position) : (j -= 1) nearest[j] = nearest[j - 1];
        nearest[position] = entry.frame;
    }
    var trail: Trail = .{};
    var k = count;
    while (k > 0) {
        k -= 1;
        trail.append(nearest[k]);
    }
    if (at.back(1)) |frame| if (!trail.contains(frame)) trail.append(frame);
    return trail;
}

fn lookup(entries: anytype, chosen: *const Prompted, frame: usize) ?usize {
    for (entries, 0..) |entry, i| if (entry.frame == frame and !chosen.has(i)) return i;
    return null;
}

pub fn plan(entries: anytype, current: usize, frames: usize, direction: Direction, grouped: bool) error{ FrameOutOfRange, MissingPromptedFrame }!Plan {
    if (frames == 0 or current >= frames) return error.FrameOutOfRange;
    const at = Position{ .current = current, .frames = frames, .direction = direction };
    const chosen = promptedEntries(entries, current);
    if (chosen.len == 0) return error.MissingPromptedFrame;
    const trail: Trail = if (grouped) .{} else confidentTrail(entries, at);
    var result: Plan = .{};

    for (chosen.slice()) |i| {
        if (!entries[i].grid_live) continue;
        const d = distance(entries[i].frame, current);
        const near = grouped and at.past(entries[i].frame) and d <= far_time_slot;
        result.addGrid(i, if (near) far_time_slot - d else far_time_slot);
    }
    var u: usize = recency;
    while (u >= 1) : (u -= 1) {
        const frame = (if (grouped) at.back(u) else trail.fromEnd(u)) orelse continue;
        const i = lookup(entries, &chosen, frame) orelse continue;
        if (entries[i].grid_live) result.addGrid(i, u - 1);
    }

    for (chosen.slice()) |i| {
        const frame = entries[i].frame;
        if (at.past(frame) or frame == current) result.addPointer(i, distance(frame, current));
    }
    var v: usize = 1;
    while (v < horizon(frames)) : (v += 1) {
        if (!grouped and v >= trail.len) break;
        const frame = (if (grouped) at.back(v) else trail.fromEnd(v)) orelse continue;
        const i = lookup(entries, &chosen, frame) orelse continue;
        result.addPointer(i, v);
    }
    return result;
}

/// Whether entry `at` can feed any future plan after frame `current` was written, and whether its grid can.
pub fn keep(entries: anytype, at: usize, current: usize, frames: usize, grouped: bool) Keep {
    const entry = entries[at];
    if (entry.prompted or entry.frame == current) return .{ .entry = true, .grid = true };
    const a = distance(entry.frame, current);
    if (grouped) return .{ .entry = a < keep_distance, .grid = a < recency };
    if (!confident(entry.confidence)) return .{ .entry = false, .grid = false };
    var closer: usize = 0;
    var unconfident_current: usize = 0;
    for (entries) |other| {
        if (other.prompted) continue;
        if (confident(other.confidence) and distance(other.frame, current) < a) closer += 1;
        if (other.frame == current and !confident(other.confidence)) unconfident_current = 1;
    }
    return .{ .entry = closer + 1 < horizon(frames), .grid = closer + unconfident_current < recency };
}

const testing = std.testing;
const Light = struct { frame: usize, prompted: bool = false, confidence: f32 = 1, grid_live: bool = true };
const Pair = [2]usize;

fn chain(buffer: []Light, prompted: usize, first: usize, count: usize, direction: Direction) []Light {
    buffer[0] = .{ .frame = prompted, .prompted = true };
    for (0..count) |k| buffer[1 + k] = .{ .frame = if (direction == .forward) first + k else first - k };
    return buffer[0 .. 1 + count];
}

fn expectPlan(entries: []const Light, current: usize, frames: usize, direction: Direction, grouped: bool, grid: []const Pair, pointer: []const Pair) !void {
    const p = try plan(entries, current, frames, direction, grouped);
    try testing.expectEqual(grid.len, p.grid_len);
    try testing.expectEqual(pointer.len, p.pointer_len);
    for (grid, p.gridSlots()) |want, got| try testing.expectEqual(want, Pair{ entries[got.entry].frame, got.time_slot });
    for (pointer, p.pointerSlots()) |want, got| try testing.expectEqual(want, Pair{ entries[got.entry].frame, got.offset });
}

test "SAM3.1 recall plan uses recent frames and distance time slots" {
    var b: [20]Light = undefined;
    try expectPlan(chain(&b, 0, 1, 0, .forward), 1, 120, .forward, true, &.{.{ 0, 5 }}, &.{.{ 0, 1 }});
    try expectPlan(chain(&b, 0, 1, 1, .forward), 2, 120, .forward, true, &.{ .{ 0, 4 }, .{ 1, 0 } }, &.{ .{ 0, 2 }, .{ 1, 1 } });
    try expectPlan(chain(&b, 0, 1, 5, .forward), 6, 120, .forward, true, &.{ .{ 0, 0 }, .{ 1, 4 }, .{ 2, 3 }, .{ 3, 2 }, .{ 4, 1 }, .{ 5, 0 } }, &.{ .{ 0, 6 }, .{ 5, 1 }, .{ 4, 2 }, .{ 3, 3 }, .{ 2, 4 }, .{ 1, 5 } });
    const case4 = chain(&b, 0, 1, 6, .forward);
    try expectPlan(case4, 7, 120, .forward, true, &.{ .{ 0, 6 }, .{ 1, 5 }, .{ 2, 4 }, .{ 3, 3 }, .{ 4, 2 }, .{ 5, 1 }, .{ 6, 0 } }, &.{ .{ 0, 7 }, .{ 6, 1 }, .{ 5, 2 }, .{ 4, 3 }, .{ 3, 4 }, .{ 2, 5 }, .{ 1, 6 } });
    case4[3].grid_live = false;
    try expectPlan(case4, 7, 120, .forward, true, &.{ .{ 0, 6 }, .{ 1, 5 }, .{ 2, 4 }, .{ 4, 2 }, .{ 5, 1 }, .{ 6, 0 } }, &.{ .{ 0, 7 }, .{ 6, 1 }, .{ 5, 2 }, .{ 4, 3 }, .{ 3, 4 }, .{ 2, 5 }, .{ 1, 6 } });
    var pointers: [16]Pair = undefined;
    pointers[0] = .{ 0, 17 };
    for (1..16) |v| pointers[v] = .{ 17 - v, v };
    try expectPlan(chain(&b, 0, 1, 16, .forward), 17, 120, .forward, true, &.{ .{ 0, 6 }, .{ 11, 5 }, .{ 12, 4 }, .{ 13, 3 }, .{ 14, 2 }, .{ 15, 1 }, .{ 16, 0 } }, &pointers);
    try expectPlan(chain(&b, 47, 0, 0, .reverse), 46, 48, .reverse, true, &.{.{ 47, 5 }}, &.{.{ 47, 1 }});
    try expectPlan(chain(&b, 47, 46, 6, .reverse), 40, 48, .reverse, true, &.{ .{ 47, 6 }, .{ 46, 5 }, .{ 45, 4 }, .{ 44, 3 }, .{ 43, 2 }, .{ 42, 1 }, .{ 41, 0 } }, &.{ .{ 47, 7 }, .{ 41, 1 }, .{ 42, 2 }, .{ 43, 3 }, .{ 44, 4 }, .{ 45, 5 }, .{ 46, 6 } });
}

test "SAM3 recall plan follows the confident trail" {
    var b: [30]Light = undefined;
    try expectPlan(chain(&b, 0, 1, 0, .forward), 1, 120, .forward, false, &.{.{ 0, 6 }}, &.{.{ 0, 1 }});
    try expectPlan(chain(&b, 0, 1, 1, .forward), 2, 120, .forward, false, &.{ .{ 0, 6 }, .{ 1, 0 } }, &.{.{ 0, 2 }});
    try expectPlan(chain(&b, 0, 1, 2, .forward), 3, 120, .forward, false, &.{ .{ 0, 6 }, .{ 1, 1 }, .{ 2, 0 } }, &.{ .{ 0, 3 }, .{ 2, 1 } });
    try expectPlan(chain(&b, 0, 1, 6, .forward), 7, 120, .forward, false, &.{ .{ 0, 6 }, .{ 1, 5 }, .{ 2, 4 }, .{ 3, 3 }, .{ 4, 2 }, .{ 5, 1 }, .{ 6, 0 } }, &.{ .{ 0, 7 }, .{ 6, 1 }, .{ 5, 2 }, .{ 4, 3 }, .{ 3, 4 }, .{ 2, 5 } });
    var pointers: [15]Pair = undefined;
    pointers[0] = .{ 0, 17 };
    for (1..15) |v| pointers[v] = .{ 17 - v, v };
    try expectPlan(chain(&b, 0, 1, 16, .forward), 17, 120, .forward, false, &.{ .{ 0, 6 }, .{ 11, 5 }, .{ 12, 4 }, .{ 13, 3 }, .{ 14, 2 }, .{ 15, 1 }, .{ 16, 0 } }, &pointers);
    const case13 = chain(&b, 0, 1, 9, .forward);
    case13[9].confidence = 0.005;
    var trail: [9]Pair = undefined;
    trail[0] = .{ 0, 10 };
    for (1..9) |v| trail[v] = .{ 10 - v, v };
    try expectPlan(case13, 10, 120, .forward, false, &.{ .{ 0, 6 }, .{ 4, 5 }, .{ 5, 4 }, .{ 6, 3 }, .{ 7, 2 }, .{ 8, 1 }, .{ 9, 0 } }, &trail);
    try expectPlan(chain(&b, 119, 0, 0, .reverse), 118, 120, .reverse, false, &.{.{ 119, 6 }}, &.{.{ 119, 1 }});
    try expectPlan(chain(&b, 119, 118, 1, .reverse), 117, 120, .reverse, false, &.{ .{ 119, 6 }, .{ 118, 0 } }, &.{.{ 119, 2 }});
}

test "SAM3 recall plan thresholds confidence strictly and skips frame zero forward" {
    const case17 = [_]Light{
        .{ .frame = 4, .prompted = true },     .{ .frame = 5, .confidence = std.math.nan(f32) },
        .{ .frame = 6, .confidence = 0.0101 }, .{ .frame = 7, .confidence = 0.01 },
        .{ .frame = 8, .confidence = 0.3 },    .{ .frame = 9, .confidence = 0 },
    };
    try expectPlan(&case17, 10, 30, .forward, false, &.{ .{ 4, 6 }, .{ 6, 2 }, .{ 8, 1 }, .{ 9, 0 } }, &.{ .{ 4, 6 }, .{ 9, 1 }, .{ 8, 2 } });
    const case19 = [_]Light{ .{ .frame = 0 }, .{ .frame = 1, .prompted = true }, .{ .frame = 2 } };
    try expectPlan(&case19, 3, 10, .forward, false, &.{ .{ 1, 6 }, .{ 2, 0 } }, &.{.{ 1, 2 }});
    const case20 = [_]Light{ .{ .frame = 9 }, .{ .frame = 8, .prompted = true }, .{ .frame = 7 } };
    try expectPlan(&case20, 6, 10, .reverse, false, &.{ .{ 8, 6 }, .{ 9, 1 }, .{ 7, 0 } }, &.{ .{ 8, 2 }, .{ 7, 1 } });
}

test "SAM recall plan caps prompted entries around the current frame" {
    var case18: [6]Light = undefined;
    for (&case18, [_]usize{ 8, 30, 18, 25, 21, 14 }) |*e, f| e.* = .{ .frame = f, .prompted = true };
    try expectPlan(&case18, 20, 40, .forward, false, &.{ .{ 18, 6 }, .{ 21, 6 }, .{ 25, 6 }, .{ 14, 6 } }, &.{ .{ 18, 2 }, .{ 14, 6 } });
    try expectPlan(&case18, 20, 40, .forward, true, &.{ .{ 18, 4 }, .{ 21, 6 }, .{ 25, 6 }, .{ 14, 0 } }, &.{ .{ 18, 2 }, .{ 14, 6 }, .{ 8, 12 } });
}

test "SAM recall plan refuses missing prompts and frames outside the video" {
    const unprompted = [_]Light{ .{ .frame = 0 }, .{ .frame = 1 } };
    const prompted = [_]Light{.{ .frame = 0, .prompted = true }};
    for ([_]bool{ false, true }) |grouped| {
        try testing.expectError(error.MissingPromptedFrame, plan(&unprompted, 2, 10, .forward, grouped));
        try testing.expectError(error.FrameOutOfRange, plan(&prompted, 10, 10, .forward, grouped));
        try testing.expectError(error.FrameOutOfRange, plan(&prompted, 0, 0, .reverse, grouped));
    }
}

fn case16(buffer: *[26]Light, direction: Direction) []Light {
    const entries = chain(buffer, if (direction == .forward) 5 else 94, if (direction == .forward) 6 else 93, 25, direction);
    for (entries[1..], 1..) |*e, position| e.confidence = if (position <= 12) 1 else 0;
    return entries;
}

test "SAM3 recall plan ranks the trail by bank position in both directions" {
    for ([_]Direction{ .forward, .reverse }) |direction| {
        var b: [26]Light = undefined;
        const entries = case16(&b, direction);
        const p = try plan(entries, if (direction == .forward) 31 else 68, 100, direction, false);
        const grid = [_]Pair{ .{ 0, 6 }, .{ 8, 5 }, .{ 9, 4 }, .{ 10, 3 }, .{ 11, 2 }, .{ 12, 1 }, .{ 25, 0 } };
        try testing.expectEqual(grid.len, p.grid_len);
        for (grid, p.gridSlots()) |want, got| try testing.expectEqual(want, Pair{ got.entry, got.time_slot });
        var pointer: [13]Pair = undefined;
        pointer[0] = .{ 0, 26 };
        pointer[1] = .{ 25, 1 };
        for (2..13) |v| pointer[v] = .{ 14 - v, v };
        try testing.expectEqual(pointer.len, p.pointer_len);
        for (pointer, p.pointerSlots()) |want, got| try testing.expectEqual(want, Pair{ got.entry, got.offset });
    }
}

test "SAM recall keep rule drops entries and grids no future plan can use" {
    for ([_]Direction{ .forward, .reverse }) |direction| {
        var b: [26]Light = undefined;
        const entries = case16(&b, direction);
        const current = entries[25].frame;
        const both = Keep{ .entry = true, .grid = true };
        try testing.expectEqual(Keep{ .entry = true, .grid = false }, keep(entries, 1, current, 100, false));
        try testing.expectEqual(both, keep(entries, 8, current, 100, false));
        try testing.expectEqual(Keep{ .entry = true, .grid = false }, keep(entries, 7, current, 100, false));
        try testing.expectEqual(Keep{ .entry = false, .grid = false }, keep(entries, 24, current, 100, false));
        try testing.expectEqual(both, keep(entries, 25, current, 100, false));
        try testing.expectEqual(both, keep(entries, 0, current, 100, false));
        try testing.expectEqual(Keep{ .entry = false, .grid = false }, keep(entries, 1, current, 100, true));
        try testing.expectEqual(Keep{ .entry = true, .grid = false }, keep(entries, 15, current, 100, true));
        try testing.expectEqual(both, keep(entries, 20, current, 100, true));
    }
    const grouped = [_]Light{ .{ .frame = 0, .prompted = true }, .{ .frame = 4 }, .{ .frame = 5 }, .{ .frame = 14 }, .{ .frame = 15 }, .{ .frame = 20 } };
    try testing.expectEqual(Keep{ .entry = true, .grid = true }, keep(&grouped, 0, 20, 30, true));
    try testing.expectEqual(Keep{ .entry = false, .grid = false }, keep(&grouped, 1, 20, 30, true));
    try testing.expectEqual(Keep{ .entry = true, .grid = false }, keep(&grouped, 2, 20, 30, true));
    try testing.expectEqual(Keep{ .entry = true, .grid = false }, keep(&grouped, 3, 20, 30, true));
    try testing.expectEqual(Keep{ .entry = true, .grid = true }, keep(&grouped, 4, 20, 30, true));
    try testing.expectEqual(Keep{ .entry = true, .grid = true }, keep(&grouped, 5, 20, 30, true));
}

fn pattern(step: usize) f32 {
    const k = (step - 1) % 32;
    return if (k < 9) 0.8 else if (k < 14) 0.02 else if (k < 31) 0.01 else std.math.nan(f32);
}

fn pruneLight(entries: []Light, current: usize, frames: usize, grouped: bool) usize {
    var decisions: [128]Keep = undefined;
    for (0..entries.len) |i| decisions[i] = keep(entries, i, current, frames, grouped);
    var sequential: [128]Light = undefined;
    @memcpy(sequential[0..entries.len], entries);
    var remaining = entries.len;
    var i: usize = 0;
    while (i < remaining) {
        const k = keep(sequential[0..remaining], i, current, frames, grouped);
        if (!k.entry) {
            std.mem.copyForwards(Light, sequential[i .. remaining - 1], sequential[i + 1 .. remaining]);
            remaining -= 1;
            continue;
        }
        if (!k.grid) sequential[i].grid_live = false;
        i += 1;
    }
    var len: usize = 0;
    for (entries, decisions[0..entries.len]) |entry, k| {
        if (!k.entry) continue;
        entries[len] = entry;
        if (!k.grid) entries[len].grid_live = false;
        len += 1;
    }
    std.debug.assert(len == remaining);
    for (entries[0..len], sequential[0..len]) |a, b| std.debug.assert(a.frame == b.frame and a.grid_live == b.grid_live);
    return len;
}

fn expectSamePlan(full: []const Light, pruned: []const Light, current: usize, frames: usize, direction: Direction, grouped: bool) !void {
    const a = try plan(full, current, frames, direction, grouped);
    const b = try plan(pruned, current, frames, direction, grouped);
    try testing.expectEqual(a.grid_len, b.grid_len);
    try testing.expectEqual(a.pointer_len, b.pointer_len);
    for (a.gridSlots(), b.gridSlots()) |x, y| {
        try testing.expectEqual(full[x.entry].frame, pruned[y.entry].frame);
        try testing.expectEqual(x.time_slot, y.time_slot);
    }
    for (a.pointerSlots(), b.pointerSlots()) |x, y| {
        try testing.expectEqual(full[x.entry].frame, pruned[y.entry].frame);
        try testing.expectEqual(x.offset, y.offset);
    }
}

test "SAM recall plans on a pruned bank equal plans on the full bank" {
    const frames = 90;
    for ([_]bool{ false, true }) |grouped| for ([_]Direction{ .forward, .reverse }) |direction| {
        var full: [frames]Light = undefined;
        var pruned: [frames]Light = undefined;
        const first: usize = if (direction == .forward) 0 else frames - 1;
        full[0] = .{ .frame = first, .prompted = true };
        pruned[0] = full[0];
        var pruned_len: usize = 1;
        for (1..frames) |step| {
            const current = if (direction == .forward) step else frames - 1 - step;
            try expectSamePlan(full[0..step], pruned[0..pruned_len], current, frames, direction, grouped);
            full[step] = .{ .frame = current, .confidence = pattern(step) };
            pruned[pruned_len] = full[step];
            pruned_len = pruneLight(pruned[0 .. pruned_len + 1], current, frames, grouped);
            try testing.expect(pruned_len <= horizon(frames) + 1);
        }
    };
}
