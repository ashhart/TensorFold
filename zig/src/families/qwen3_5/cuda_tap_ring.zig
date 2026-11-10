//! Target feature storage: retain a DFlash window and one complete per-stream verify, not a prompt-sized ring.
const std = @import("std");

/// Other drafters retain their original capacity; the caller selects only plain dense27 DFlash2 here.
pub fn targetCapacity(legacy: usize, window: usize, horizon: usize, compact: bool) !usize {
    if (!compact) return legacy;
    if (window == 0 or horizon == 0) return error.BadTapCapacity;
    const rows = std.math.add(usize, window, horizon) catch return error.BadTapCapacity;
    if (rows > legacy or rows > std.math.maxInt(c_int)) return error.BadTapCapacity;
    return rows;
}

/// qwen35_rows.cuh uses the same cut before writing: surviving rows must have distinct modulo slots.
pub fn firstWrite(rows: usize, capacity: usize) usize {
    std.debug.assert(capacity > 0);
    return rows -| capacity;
}

/// Absolute positions, including a selected verify branch, retain their modulus after a wrapped write.
pub fn slot(start: u64, row: u32, capacity: usize) !usize {
    if (capacity == 0) return error.BadTapCapacity;
    const position = std.math.add(u64, start, row) catch return error.BadTapPosition;
    return @intCast(position % capacity);
}

pub const Interval = struct {
    begin: u64 = 0,
    end: u64 = 0,

    pub fn contains(self: Interval, start: u64, rows: usize) bool {
        const end = std.math.add(u64, start, rows) catch return false;
        return rows > 0 and start >= self.begin and end <= self.end;
    }
};

/// Only the newest write is promised: a short rollback cannot revive rows overwritten by its rejected suffix.
pub fn retained(start: u64, rows: usize, capacity: usize) !Interval {
    if (rows == 0 or capacity == 0) return error.BadTapCapacity;
    const end = std.math.add(u64, start, rows) catch return error.BadTapPosition;
    return .{ .begin = end - @min(rows, capacity), .end = end };
}

/// A prompt consumer may drop its final row after taking Features, then truncate to its DFlash window.
pub fn needed(start: u64, rows: usize, window: usize) !Interval {
    if (window == 0) return error.BadTapCapacity;
    const keep = std.math.add(usize, window, 1) catch return error.BadTapCapacity;
    return retained(start, rows, keep);
}

pub const Move = struct { source: usize, dest: usize, rows: usize };
pub const Copies = struct { items: [3]Move = undefined, count: usize = 0, range: Interval };

/// Two ring boundaries can split a retained tail at most three ways; every move is a contiguous byte copy.
pub fn copyTail(start: u64, rows: usize, source_capacity: usize, dest_capacity: usize) !Copies {
    const range = try retained(start, rows, @min(source_capacity, dest_capacity));
    var plan: Copies = .{ .range = range };
    var position = range.begin;
    while (position < range.end) {
        const source: usize = @intCast(position % source_capacity);
        const dest: usize = @intCast(position % dest_capacity);
        const count = @min(range.end - position, @min(source_capacity - source, dest_capacity - dest));
        std.debug.assert(plan.count < plan.items.len);
        plan.items[plan.count] = .{ .source = source, .dest = dest, .rows = @intCast(count) };
        plan.count += 1;
        position += count;
    }
    return plan;
}

test "Qwen compact taps: one DFlash window plus the per-stream verify, other drafters unchanged" {
    const t = std.testing;
    for ([_]usize{ 6144, 10240 }) |legacy| {
        try t.expectEqual(@as(usize, 2112), try targetCapacity(legacy, 2048, 64, true));
        try t.expectEqual(legacy, try targetCapacity(legacy, 0, 0, false));
    }
    try t.expectEqual(@as(usize, 108134400), try targetCapacity(10240, 2048, 64, true) * 5 * 5120 * 2);
    try t.expectError(error.BadTapCapacity, targetCapacity(2048, 2048, 64, true));
    try t.expectError(error.BadTapCapacity, targetCapacity(10240, 0, 64, true));
    try t.expectError(error.BadTapCapacity, targetCapacity(10240, 2048, 0, true));
    try t.expectError(error.BadTapCapacity, targetCapacity(std.math.maxInt(usize), std.math.maxInt(usize), 1, true));
    try t.expectError(error.BadTapCapacity, targetCapacity(std.math.maxInt(usize), std.math.maxInt(c_int), 1, true));
}

test "Qwen compact taps: 8192-row prefill has exactly one writer per retained slot" {
    const t = std.testing;
    const capacity = 2112;
    for ([_]usize{ 1, 64, 2048, capacity, capacity + 1, 4096, 8192 }) |rows| {
        var seen: [capacity]u16 = @splat(0);
        for (firstWrite(rows, capacity)..rows) |r| seen[try slot(73, @intCast(r), capacity)] += 1;
        var writes: usize = 0;
        for (seen) |n| {
            try t.expect(n <= 1);
            writes += n;
        }
        try t.expectEqual(@min(rows, capacity), writes);
    }
    // The original untrimmed launch is an actual counterexample: parallel rows alias the same output.
    var unsafe: [capacity]u16 = @splat(0);
    for (0..8192) |r| unsafe[try slot(0, @intCast(r), capacity)] += 1;
    try t.expect(unsafe[0] > 1);
}

test "Qwen compact taps: full prompt tail, append, wrap and selected verify preserve exact words" {
    const t = std.testing;
    const capacity = 2112;
    const unset = std.math.maxInt(u64);
    var compact: [capacity]u64 = @splat(unset);
    var absolute: [20000]u64 = @splat(unset);
    const chunks = [_]struct { start: u64, rows: usize }{
        .{ .start = 0, .rows = 8192 },
        .{ .start = 8192, .rows = 64 },
        .{ .start = 8200, .rows = 64 }, // a rejected suffix replaced by the next verify
        .{ .start = 8264, .rows = 4096 },
        .{ .start = 12360, .rows = 1 },
    };
    for (chunks, 0..) |chunk, ci| {
        for (0..chunk.rows) |r| absolute[chunk.start + r] = (chunk.start + r) * 17 + ci;
        for (firstWrite(chunk.rows, capacity)..chunk.rows) |r|
            compact[try slot(chunk.start, @intCast(r), capacity)] = absolute[chunk.start + r];
        const end = chunk.start + chunk.rows;
        const kept_rows = @min(end, 2048);
        // Initial absorption excludes the prompt's last row; the extra verify horizon retains it too.
        const read_end = if (ci == 0) end - 1 else end;
        for (read_end - kept_rows..read_end) |p|
            try t.expectEqual(absolute[p], compact[try slot(p, 0, capacity)]);
        if (chunk.rows <= 64) for ([_]u32{ 0, 3, 7, 15, 31, 63 }) |r| {
            if (r < chunk.rows) try t.expectEqual(absolute[chunk.start + r], compact[try slot(chunk.start, r, capacity)]);
        };
    }
    try t.expectError(error.BadTapCapacity, slot(0, 0, 0));
    try t.expectError(error.BadTapPosition, slot(std.math.maxInt(u64), 1, capacity));
}

test "Qwen shared taps: 65-row prompt retains the excluded final row's preceding 64 rows" {
    const range = try retained(0, 65, 65);
    try std.testing.expect(range.contains(0, 64));
    try std.testing.expect(!(try retained(0, 65, 64)).contains(0, 64));
    try std.testing.expect((try retained(0, 8192, 2112)).contains((try needed(0, 8192, 2048)).begin, 2049));
}

test "Qwen shared taps: rollback shorter writes never revive overwritten history" {
    const before = try retained(8192, 64, 65);
    const after = try retained(8193, 1, 65);
    try std.testing.expect(before.contains(8192, 64));
    try std.testing.expect(after.contains(8193, 1));
    try std.testing.expect(!after.contains(8130, 64));
    try std.testing.expect(!after.contains(8192, 1));
}

test "Qwen shared taps: copy plans match absolute positions at both ring boundaries" {
    var source: [2112]u64 = undefined;
    var dest: [65]u64 = undefined;
    for ([_]usize{ 1, 64, 65, 66, 2048, 2112, 2113, 8192 }) |rows| {
        for ([_]u64{ 0, 1, 64, 65, 2090, 2111, 2112, 8192 }) |start| {
            @memset(&source, std.math.maxInt(u64));
            @memset(&dest, std.math.maxInt(u64));
            const kept = try retained(start, rows, source.len);
            for (kept.begin..kept.end) |position| source[position % source.len] = position;
            const plan = try copyTail(start, rows, source.len, dest.len);
            for (plan.items[0..plan.count]) |move|
                @memcpy(dest[move.dest..][0..move.rows], source[move.source..][0..move.rows]);
            for (plan.range.begin..plan.range.end) |position| try std.testing.expectEqual(position, dest[position % dest.len]);
            try std.testing.expect(plan.count <= 3);
        }
    }
    try std.testing.expectError(error.BadTapCapacity, copyTail(0, 1, 0, 65));
    try std.testing.expectError(error.BadTapPosition, copyTail(std.math.maxInt(u64), 1, 2112, 65));
}
