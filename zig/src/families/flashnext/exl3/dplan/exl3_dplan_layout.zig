const std = @import("std");

pub const experts_n: usize = 512;
pub const recent_window: usize = 256;
pub const recent_words: usize = recent_window * (experts_n / 32);
pub const misc_words: usize = 10;
pub const static_bytes: usize = experts_n * 3 * (8 + 8 + 4);

pub fn recentCountsOffset(lines: usize) usize {
    return 2 * lines + 2 * experts_n;
}

pub fn recentOffset(lines: usize) usize {
    return 2 * lines + 3 * experts_n;
}

pub fn miscOffset(lines: usize) usize {
    return recentOffset(lines) + recent_words;
}

pub fn stateWords(lines: usize) usize {
    return miscOffset(lines) + misc_words;
}

pub const PoolState = struct {
    lines: usize,
    offset_words: usize,
    words: usize,

    pub fn init(lines: usize, offset_words: usize) !PoolState {
        if (lines == 0 or lines > (std.math.maxInt(c_int) - stateWords(0)) / 2) return error.Invalid;
        const words = stateWords(lines);
        const end = try std.math.add(usize, offset_words, words);
        _ = try std.math.mul(usize, end, @sizeOf(u32));
        return .{ .lines = lines, .offset_words = offset_words, .words = words };
    }

    pub fn endWords(self: PoolState) usize {
        return self.offset_words + self.words;
    }

    pub fn byteOffset(self: PoolState) usize {
        return self.offset_words * @sizeOf(u32);
    }

    pub fn slice(self: PoolState, all: []u32) []u32 {
        return all[self.offset_words..self.endWords()];
    }
};

pub fn deviceBytes(layers: u64, total_lines: u64, rows: u64) !u64 {
    const fixed = try std.math.mul(u64, layers, stateWords(0) * @sizeOf(u32) + static_bytes);
    const state_lines = try std.math.mul(u64, total_lines, 2 * @sizeOf(u32));
    const work = try std.math.add(u64, try std.math.mul(u64, rows, 10 * 9 * 8 + 2 * 11 * 4), 256 + (experts_n + 64) * 4);
    return std.math.add(u64, try std.math.add(u64, fixed, state_lines), work);
}

test "exl3: ragged planner offsets are contiguous and exactly bounded" {
    const counts = [_]usize{ 10, 73, 11, 32 };
    var end: usize = 0;
    for (counts) |lines| {
        const entry = try PoolState.init(lines, end);
        try std.testing.expectEqual(end, entry.offset_words);
        try std.testing.expectEqual(stateWords(lines), entry.words);
        try std.testing.expectEqual(end * 4, entry.byteOffset());
        try std.testing.expectEqual(entry.words, miscOffset(lines) + misc_words);
        end = entry.endWords();
    }
    try std.testing.expectEqual(counts.len * stateWords(0) + 2 * (10 + 73 + 11 + 32), end);
    try std.testing.expectError(error.Invalid, PoolState.init(0, 0));
    const max_lines = (std.math.maxInt(c_int) - stateWords(0)) / 2;
    _ = try PoolState.init(max_lines, 0);
    try std.testing.expectError(error.Invalid, PoolState.init(max_lines + 1, 0));
    try std.testing.expectError(error.Invalid, PoolState.init(std.math.maxInt(usize), 0));
    try std.testing.expectError(error.Overflow, PoolState.init(10, std.math.maxInt(usize)));
    try std.testing.expectError(error.Overflow, PoolState.init(10, std.math.maxInt(usize) / 4));
    var uniform_end: usize = 0;
    for (0..4) |index| {
        const entry = try PoolState.init(32, uniform_end);
        try std.testing.expectEqual(index * stateWords(32), entry.offset_words);
        uniform_end = entry.endWords();
    }
}

test "exl3: ragged planner state round trip isolates line maps history and counters" {
    const counts = [_]usize{ 10, 57, 13 };
    var entries: [counts.len]PoolState = undefined;
    var end: usize = 1;
    for (counts, &entries) |lines, *entry| {
        entry.* = try PoolState.init(lines, end);
        end = entry.endWords();
    }
    const host = try std.testing.allocator.alloc(u32, end + 1);
    defer std.testing.allocator.free(host);
    @memset(host, 0xfeedface);
    for (entries, 0..) |entry, index| {
        const state = entry.slice(host);
        @memset(state, 0);
        for (0..entry.lines) |line| {
            const expert = (line + index * 100) % experts_n;
            state[line] = @intCast(expert);
            state[entry.lines + line] = @intCast(line + 1);
            state[2 * entry.lines + expert] = @intCast(line);
        }
        state[recentCountsOffset(entry.lines)] = @intCast(index + 1);
        state[recentOffset(entry.lines)] = @intCast(index + 10);
        state[miscOffset(entry.lines)] = @intCast(index + 20);
        state[state.len - 1] = @intCast(index + 30);
    }
    const device = try std.testing.allocator.dupe(u32, host);
    defer std.testing.allocator.free(device);
    @memset(host[1..end], 0);
    for (entries) |entry| @memcpy(entry.slice(host), entry.slice(device));
    try std.testing.expectEqual(@as(u32, 0xfeedface), host[0]);
    try std.testing.expectEqual(@as(u32, 0xfeedface), host[end]);
    for (entries, 0..) |entry, index| {
        const state = entry.slice(host);
        for (0..entry.lines) |line| {
            const expert = state[line];
            try std.testing.expectEqual(@as(u32, @intCast(line)), state[2 * entry.lines + expert]);
            try std.testing.expectEqual(@as(u32, @intCast(line + 1)), state[entry.lines + line]);
        }
        try std.testing.expectEqual(@as(u32, @intCast(index + 1)), state[recentCountsOffset(entry.lines)]);
        try std.testing.expectEqual(@as(u32, @intCast(index + 10)), state[recentOffset(entry.lines)]);
        try std.testing.expectEqual(@as(u32, @intCast(index + 20)), state[miscOffset(entry.lines)]);
        try std.testing.expectEqual(@as(u32, @intCast(index + 30)), state[state.len - 1]);
    }
}
