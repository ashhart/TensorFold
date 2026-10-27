//! DPLAN v4: default off, little-endian u32 words, fixed 1 MiB per device pool.

const std = @import("std");
const layout = @import("exl3_dplan_layout.zig");

pub const magic: u32 = 0x44505431;
pub const version: u32 = 4;
pub const pool_bytes: usize = 1024 * 1024;
pub const pool_words: usize = pool_bytes / 4;
pub const header_words: usize = 16;
pub const record_header_words: usize = 4 + layout.misc_words + 1;
pub const checkpoint_header_words: usize = 4;

pub fn checkpointWords(state_words: usize) usize {
    return checkpoint_header_words + 2 * state_words;
}

pub fn seedFromState(comptime Seed: type, lines: usize, seed: []const u32) Seed {
    return .{ .line_ids = @ptrCast(seed[0..lines]), .stamps = seed[lines .. 2 * lines], .counts = seed[2 * lines + layout.experts_n ..][0..layout.experts_n].*, .now = seed[layout.miscOffset(lines)] };
}

pub const Checkpoint = struct {
    words: []const u32,

    pub fn roundIndex(self: Checkpoint) usize {
        return self.words[0];
    }
    pub fn before(self: Checkpoint) []const u32 {
        return self.words[checkpoint_header_words..][0..self.words[2]];
    }
    pub fn after(self: Checkpoint) []const u32 {
        return self.words[checkpoint_header_words + self.words[2] ..][0..self.words[2]];
    }
};

pub fn enabledValue(value: ?[]const u8) bool {
    return if (value) |value_text| std.mem.eql(u8, value_text, "1") else false;
}

pub fn header(pool_index: usize, layer: usize, lines: usize, max_rows: usize) ![header_words]u32 {
    if (max_rows == 0 or max_rows > pool_words / 11) return error.TraceShapeTooLarge;
    const entry = layout.PoolState.init(lines, 0) catch return error.TraceShapeTooLarge;
    if (entry.words > (pool_words - header_words) / 2) return error.TraceShapeTooLarge;
    const stride = record_header_words + max_rows * 11;
    const available = pool_words - header_words - 2 * entry.words;
    if (stride > available or pool_index > std.math.maxInt(u32) or layer > std.math.maxInt(u32)) return error.TraceShapeTooLarge;
    return .{ magic, version, @intCast(pool_index), @intCast(layer), @intCast(lines), @intCast(max_rows), @intCast(entry.words), @intCast(stride), @intCast(available / stride), 0, 0, 0, 0, 0, 0, 0 };
}

pub const Round = struct {
    words: []const u32,
    lines: usize,
    max_rows: usize,

    pub fn rows(self: Round) usize {
        return self.words[0];
    }
    pub fn validPicks(self: Round) u32 {
        return self.words[1];
    }
    pub fn uniqueCount(self: Round) usize {
        return self.words[2];
    }
    pub fn recentEnabled(self: Round) bool {
        return self.words[3] == 1;
    }
    pub fn misc(self: Round) []const u32 {
        return self.words[4..][0..layout.misc_words];
    }
    pub fn mappedPicks(self: Round) u32 {
        return self.words[4 + layout.misc_words];
    }
    pub fn picks(self: Round) []const u32 {
        return self.words[record_header_words..][0 .. self.rows() * 11];
    }

    fn validate(self: Round) !void {
        if (self.rows() == 0 or self.rows() > self.max_rows or self.uniqueCount() > layout.experts_n or self.words[3] > 1) return error.InvalidTrace;
        var seen: [layout.experts_n]bool = @splat(false);
        var unique: usize = 0;
        var valid: u32 = 0;
        for (self.picks()) |bits| {
            const expert: i32 = @bitCast(bits);
            if (expert < 0 or expert >= layout.experts_n) continue;
            const index: usize = @intCast(expert);
            if (!seen[index]) unique += 1;
            seen[index] = true;
            valid += 1;
        }
        if (unique != self.uniqueCount() or valid != self.validPicks() or self.mappedPicks() > valid) return error.InvalidTrace;
        if (self.misc()[8] >= layout.recent_window or self.misc()[9] > layout.recent_window) return error.InvalidTrace;
    }
};

pub const View = struct {
    words: []const u32,

    pub fn init(words: []const u32) !View {
        if (words.len < header_words or words.len > pool_words or words[0] != magic or words[1] != version) return error.InvalidTrace;
        const expected = header(words[2], words[3], words[4], words[5]) catch return error.InvalidTrace;
        if (!std.mem.eql(u32, words[0..9], expected[0..9]) or words[9] > words[8]) return error.InvalidTrace;
        if (words[12] > 1 or words[13] > 1) return error.InvalidTrace;
        const self: View = .{ .words = words };
        const stride = checkpointWords(words[6]);
        if (self.checkpoints() > (pool_words - self.recordsOffset()) / stride) return error.InvalidTrace;
        const records_end = self.recordsOffset() + self.recorded() * words[7];
        const checkpoints_start = pool_words - self.checkpoints() * stride;
        if (records_end > checkpoints_start) return error.InvalidTrace;
        if (words.len < self.usedWords()) return error.InvalidTrace;
        if (self.dropped() != 0 and self.droppedCheckpoints() == 0 and checkpoints_start - records_end >= words[7]) return error.InvalidTrace;
        if (self.droppedCheckpoints() != 0 and self.dropped() == 0 and checkpoints_start - records_end >= stride) return error.InvalidTrace;
        var previous_round: usize = 0;
        for (0..self.checkpoints()) |index| {
            const entry = try self.checkpoint(index);
            if (entry.words[1] != 1 or entry.words[2] != words[6] or entry.words[3] != index or entry.roundIndex() > self.recorded() or entry.roundIndex() < previous_round) return error.InvalidTrace;
            previous_round = entry.roundIndex();
        }
        for (0..self.recorded()) |index| try (try self.round(index)).validate();
        return self;
    }

    pub fn recorded(self: View) usize {
        return self.words[9];
    }
    pub fn dropped(self: View) u64 {
        return @as(u64, self.words[10]) | (@as(u64, self.words[11]) << 32);
    }
    pub fn checkpoints(self: View) usize {
        return self.words[14];
    }
    pub fn droppedCheckpoints(self: View) u32 {
        return self.words[15];
    }
    pub fn checkpoint(self: View, index: usize) !Checkpoint {
        if (index >= self.checkpoints()) return error.TraceCheckpointOutOfRange;
        const stride = checkpointWords(self.words[6]);
        return .{ .words = self.words[pool_words - (index + 1) * stride ..][0..stride] };
    }
    pub fn initialSeed(self: View) []const u32 {
        return self.words[header_words..][0..self.words[6]];
    }
    pub fn terminalSeed(self: View) []const u32 {
        return self.words[header_words + self.words[6] ..][0..self.words[6]];
    }
    pub fn seedState(self: View, comptime Seed: type) Seed {
        return seedFromState(Seed, self.words[4], self.initialSeed());
    }
    pub fn recordsOffset(self: View) usize {
        return header_words + 2 * @as(usize, self.words[6]);
    }
    pub fn round(self: View, index: usize) !Round {
        if (index >= self.recorded()) return error.TraceRoundOutOfRange;
        const stride = self.words[7];
        return .{ .words = self.words[self.recordsOffset() + index * stride ..][0..stride], .lines = self.words[4], .max_rows = self.words[5] };
    }
    fn usedWords(self: View) usize {
        if (self.checkpoints() != 0) return pool_words;
        return self.recordsOffset() + self.recorded() * self.words[7];
    }
    pub fn encode(self: View, allocator: std.mem.Allocator) ![]u8 {
        const bytes = try allocator.alloc(u8, self.usedWords() * 4);
        for (self.words[0..self.usedWords()], 0..) |word, index| std.mem.writeInt(u32, bytes[index * 4 ..][0..4], word, .little);
        return bytes;
    }
};

pub const Snapshot = struct {
    allocator: std.mem.Allocator,
    words: []u32,

    pub fn deinit(self: *Snapshot) void {
        self.allocator.free(self.words);
        self.* = undefined;
    }
    pub fn view(self: Snapshot) !View {
        return View.init(self.words);
    }
    pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) !Snapshot {
        if (bytes.len % 4 != 0 or bytes.len > pool_bytes or bytes.len < header_words * 4) return error.InvalidTrace;
        const words = try allocator.alloc(u32, bytes.len / 4);
        errdefer allocator.free(words);
        for (words, 0..) |*word, index| word.* = std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
        _ = try View.init(words);
        return .{ .allocator = allocator, .words = words };
    }
};
