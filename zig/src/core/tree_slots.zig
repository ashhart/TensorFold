//! Depth-first trees recycle a parent's register slot at its last child and preserve original output rows.
const std = @import("std");

pub const max_slots = 16;
pub const Step = extern struct { row: u32, read_slot: i32, write_slot: i32 };

pub fn build(gpa: std.mem.Allocator, parents: []const i32, steps: []Step, first: u32, limit: usize) !usize {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots or parents.len > std.math.maxInt(u32) - first) return error.BadSlotPlan;
    const sizes = try gpa.alloc(usize, parents.len);
    defer gpa.free(sizes);
    @memset(sizes, 1);
    for (parents, 0..) |parent, row| if (parent < -1 or parent >= row) return error.BadSlotParents;
    var remaining = parents.len;
    while (remaining > 0) {
        remaining -= 1;
        if (parents[remaining] >= 0) sizes[@intCast(parents[remaining])] += sizes[remaining];
    }
    var planner = Builder{ .parents = parents, .sizes = sizes, .steps = steps, .first = first, .limit = limit };
    for (parents, 0..) |parent, row| if (parent == -1) try planner.visit(row, -1, false);
    std.debug.assert(planner.at == steps.len and planner.live == 0);
    return planner.peak;
}

/// A schedule as flow words: row, then slots to restore and save into plus one, 0 for none; false for a forest.
pub fn flows(steps: []const Step, words: []u32) bool {
    for (steps, words, 0..) |step, *word, at| {
        if ((at > 0 and step.read_slot < 0) or step.row > 0xffff) return false;
        const continues = at == 0 or step.read_slot == steps[at - 1].write_slot;
        const restore: u32 = if (continues) 0 else @intCast(step.read_slot + 1);
        var save: u32 = 0;
        for (steps[at + 1 ..], at + 1..) |later, j| {
            if (j > at + 1 and later.read_slot == step.write_slot) {
                save = @intCast(step.write_slot + 1);
                break;
            }
            if (later.write_slot == step.write_slot) break;
        }
        word.* = step.row | (restore << 16) | (save << 24);
    }
    return true;
}

const Builder = struct {
    parents: []const i32,
    sizes: []const usize,
    steps: []Step,
    first: u32,
    limit: usize,
    used: [max_slots]bool = @splat(false),
    live: usize = 0,
    peak: usize = 0,
    at: usize = 0,

    fn take(b: *Builder) !i32 {
        for (b.used[0..b.limit], 0..) |*used, slot| if (!used.*) {
            used.* = true;
            b.live += 1;
            b.peak = @max(b.peak, b.live);
            return @intCast(slot);
        };
        return error.SlotOverflow;
    }

    fn visit(b: *Builder, row: usize, read: i32, reuse: bool) error{SlotOverflow}!void {
        const write = if (reuse) read else try b.take();
        b.steps[b.at] = .{ .row = b.first + @as(u32, @intCast(row)), .read_slot = read, .write_slot = write };
        b.at += 1;
        var last: ?usize = null;
        for (b.parents, 0..) |parent, child| if (parent == row) {
            if (last == null or b.sizes[child] >= b.sizes[last.?]) last = child;
        };
        if (last) |heavy| {
            for (b.parents, 0..) |parent, child| if (parent == row and child != heavy) try b.visit(child, write, false);
            try b.visit(heavy, write, true);
        } else {
            b.used[@intCast(write)] = false;
            b.live -= 1;
        }
    }
};

pub fn validate(parents: []const i32, steps: []const Step, first: u32, limit: usize) !void {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots) return error.BadSlotPlan;
    var owners: [max_slots]i32 = @splat(-2);
    for (steps, 0..) |step, at| {
        if (step.row < first or step.row - first >= parents.len or step.read_slot < -1 or step.read_slot >= limit or step.write_slot < 0 or step.write_slot >= limit) return error.BadSlotPlan;
        const row = step.row - first;
        const parent = parents[row];
        if (parent < -1 or parent >= row) return error.BadSlotParents;
        if (step.read_slot < 0) {
            if (parent != -1) return error.BadSlotPlan;
        } else if (owners[@intCast(step.read_slot)] != parent) return error.BadSlotPlan;
        for (steps[0..at]) |prior| if (prior.row == step.row) return error.BadSlotPlan;
        owners[@intCast(step.write_slot)] = @intCast(row);
    }
}

test "slot schedules retain each parent's state while moving the largest child last" {
    const parents = [_]i32{ -1, 0, 0, 1, 1, 3, 2, 6, 6 };
    var steps: [parents.len]Step = undefined;
    try std.testing.expectEqual(@as(usize, 3), try build(std.testing.allocator, &parents, &steps, 23, 16));
    try validate(&parents, &steps, 23, 16);
    try std.testing.expectEqualSlices(u32, &.{ 23, 24, 27, 26, 28, 25, 29, 30, 31 }, &rowsOf(steps));
    steps[2].read_slot = 15;
    try std.testing.expectError(error.BadSlotPlan, validate(&parents, &steps, 23, 16));
}

fn rowsOf(steps: anytype) [steps.len]u32 {
    var rows: [steps.len]u32 = undefined;
    for (steps, &rows) |step, *row| row.* = step.row;
    return rows;
}

test "flow words compute every row from the same parent as the slot schedule" {
    var random: u64 = 4099;
    for ([_]usize{ 2, 5, 16, 33, 64 }) |width| {
        var parents: [64]i32 = undefined;
        var steps: [64]Step = undefined;
        var words: [64]u32 = undefined;
        var expected: [64]u64 = undefined;
        for (0..32) |_| {
            for (parents[0..width], 0..) |*parent, row| {
                random = random *% 6364136223846793005 +% 1442695040888963407;
                parent.* = if (row == 0) -1 else @intCast(random % row);
                const value = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
                expected[row] = value *% 65537 +% row +% 1;
            }
            _ = try build(std.testing.allocator, parents[0..width], steps[0..width], 7, 16);
            try std.testing.expect(flows(steps[0..width], words[0..width]));
            var kept: [max_slots]u64 = @splat(0);
            var state: u64 = 23;
            for (words[0..width]) |word| {
                const restore = (word >> 16) & 0xff;
                if (restore != 0) state = kept[restore - 1];
                const row = (word & 0xffff) - 7;
                state = state *% 65537 +% row +% 1;
                try std.testing.expectEqual(expected[row], state);
                if (word >> 24 != 0) kept[(word >> 24) - 1] = state;
            }
        }
    }
    const forest = [_]i32{ -1, 0, -1 };
    var steps: [3]Step = undefined;
    var words: [3]u32 = undefined;
    _ = try build(std.testing.allocator, &forest, &steps, 0, 16);
    try std.testing.expect(!flows(&steps, &words));
}

test "long chains need one slot and stars need two independent of depth or width" {
    var parents: [512]i32 = undefined;
    var steps: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast(row - 1);
    try std.testing.expectEqual(@as(usize, 1), try build(std.testing.allocator, &parents, &steps, 0, 1));
    try validate(&parents, &steps, 0, 1);
    @memset(parents[1..], 0);
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 0, 1));
    try std.testing.expectEqual(@as(usize, 2), try build(std.testing.allocator, &parents, &steps, 0, 2));
    try validate(&parents, &steps, 0, 2);
}

test "balanced trees reuse ancestor slots and an explicit slot budget refuses overflow" {
    var parents: [127]i32 = undefined;
    var steps: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast((row - 1) / 2);
    try std.testing.expectEqual(@as(usize, 7), try build(std.testing.allocator, &parents, &steps, 0, 16));
    try validate(&parents, &steps, 0, 16);
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 0, 6));
}

test "forests reset to committed state and bad parents never make a schedule" {
    const parents = [_]i32{ -1, 0, -1, 2, 2, -1 };
    var steps: [parents.len]Step = undefined;
    _ = try build(std.testing.allocator, &parents, &steps, 73, 16);
    try validate(&parents, &steps, 73, 16);
    for ([_][]const i32{ &.{ -1, 1 }, &.{ -2, 0 }, &.{ -1, 2 } }) |bad| {
        try std.testing.expectError(error.BadSlotParents, build(std.testing.allocator, bad, steps[0..bad.len], 0, 16));
    }
}

test "irregular trees compute every row from the same parent as an independent topological replay" {
    var random: u64 = 1729;
    for ([_]usize{ 17, 32, 64, 128 }) |width| {
        var parents: [128]i32 = undefined;
        var steps: [128]Step = undefined;
        var expected: [128]u64 = undefined;
        var actual: [128]u64 = undefined;
        for (0..32) |_| {
            for (parents[0..width], 0..) |*parent, row| {
                random = random *% 6364136223846793005 +% 1442695040888963407;
                parent.* = if (row == 0) -1 else @intCast(random % row);
                const value = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
                expected[row] = value *% 65537 +% row +% 1;
            }
            _ = try build(std.testing.allocator, parents[0..width], steps[0..width], 11, 16);
            try validate(parents[0..width], steps[0..width], 11, 16);
            var registers: [max_slots]u64 = @splat(0);
            for (steps[0..width]) |step| {
                const row = step.row - 11;
                const value = if (step.read_slot < 0) @as(u64, 23) else registers[@intCast(step.read_slot)];
                actual[row] = value *% 65537 +% row +% 1;
                registers[@intCast(step.write_slot)] = actual[row];
            }
            try std.testing.expectEqualSlices(u64, expected[0..width], actual[0..width]);
        }
    }
}
