const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");
const layout = @import("exl3_dplan_layout.zig");

pub fn emptySeed(allocator: std.mem.Allocator, lines: usize) ![]u32 {
    const seed = try allocator.alloc(u32, layout.stateWords(lines));
    @memset(seed, 0);
    @memset(seed[0..lines], std.math.maxInt(u32));
    @memset(seed[2 * lines ..][0..layout.experts_n], std.math.maxInt(u32));
    return seed;
}

pub fn make(allocator: std.mem.Allocator, lines: usize, rows: usize) ![]u32 {
    const words = try allocator.alloc(u32, trace.pool_words);
    errdefer allocator.free(words);
    @memset(words, 0);
    @memcpy(words[0..trace.header_words], &try trace.header(3, 48, lines, rows));
    return words;
}

pub fn append(words: []u32, before: []const u32, after: []const u32, picks: []const i32, recent: bool, mapped: u32) void {
    const records_end = trace.header_words + 2 * words[6] + words[9] * words[7];
    const checkpoints_start = trace.pool_words - words[14] * trace.checkpointWords(words[6]);
    if (checkpoints_start - records_end < words[7] or words[15] != 0) {
        const dropped = (@as(u64, words[11]) << 32 | words[10]) +| 1;
        words[10] = @truncate(dropped);
        words[11] = @truncate(dropped >> 32);
        return;
    }
    const seed = words[trace.header_words..][0..words[6]];
    const terminal = words[trace.header_words + words[6] ..][0..words[6]];
    if (words[9] == 0 and words[14] == 0) {
        @memcpy(seed, before);
    } else if (!std.mem.eql(u32, terminal, before)) {
        words[12] = 1;
    }
    const offset = trace.header_words + 2 * words[6] + words[9] * words[7];
    const record = words[offset..][0..words[7]];
    @memset(record, 0);
    record[0] = @intCast(picks.len / 11);
    record[3] = @intFromBool(recent);
    @memcpy(record[4..][0..layout.misc_words], after[layout.miscOffset(words[4])..]);
    record[14] = mapped;
    var seen: [layout.experts_n]bool = @splat(false);
    for (picks, 0..) |expert, index| {
        record[trace.record_header_words + index] = @bitCast(expert);
        if (expert < 0 or expert >= layout.experts_n) continue;
        const valid: usize = @intCast(expert);
        record[1] += 1;
        if (!seen[valid]) record[2] += 1;
        seen[valid] = true;
    }
    @memcpy(terminal, after);
    words[9] += 1;
}

pub fn checkpoint(words: []u32, incoming: []const u32) void {
    const stride = trace.checkpointWords(words[6]);
    const end = trace.header_words + 2 * words[6] + words[9] * words[7];
    const start = trace.pool_words - words[14] * stride;
    if (start - end < stride or words[10] != 0 or words[11] != 0 or words[15] != 0) {
        words[15] +|= 1;
        return;
    }
    const initial = words[trace.header_words..][0..words[6]];
    const terminal = words[trace.header_words + words[6] ..][0..words[6]];
    if (words[9] == 0 and words[14] == 0) {
        @memcpy(initial, incoming);
        @memcpy(terminal, incoming);
    }
    const entry = words[start - stride .. start];
    entry[0..4].* = .{ words[9], 1, words[6], words[14] };
    @memcpy(entry[4..][0..words[6]], terminal);
    @memcpy(entry[4 + words[6] ..], incoming);
    @memcpy(terminal, incoming);
    words[14] += 1;
}

test "exl3: checkpoints reject corrupt ordinals shapes order overlap and truncation" {
    const allocator = std.testing.allocator;
    const words = try make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try emptySeed(allocator, 2);
    defer allocator.free(seed);
    const picks: [11]i32 = @splat(-1);
    append(words, seed, seed, &picks, false, 0);
    checkpoint(words, seed);
    append(words, seed, seed, &picks, false, 0);
    checkpoint(words, seed);
    const view = try trace.View.init(words);
    try std.testing.expectError(error.TraceCheckpointOutOfRange, view.checkpoint(2));
    const stride = trace.checkpointWords(words[6]);
    const offset = trace.pool_words - stride;
    for (0..4) |field| {
        const saved = words[offset + field];
        words[offset + field] = std.math.maxInt(u32);
        try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
        words[offset + field] = saved;
    }
    words[offset - stride] = 0;
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
    words[offset - stride] = 2;
    const saved_count = words[14];
    words[14] = std.math.maxInt(u32);
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
    words[14] = @intCast((trace.pool_words - view.recordsOffset()) / stride);
    words[9] = words[8];
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
    words[14] = saved_count;
    words[9] = 2;
    const bytes = try view.encode(allocator);
    defer allocator.free(bytes);
    try std.testing.expectError(error.InvalidTrace, trace.Snapshot.decode(allocator, bytes[0 .. bytes.len - 4]));
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words[0 .. words.len - 1]));
    words[15] = 1;
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
}

test "exl3: checkpoint exhaustion saturates without overwriting retained records or states" {
    const allocator = std.testing.allocator;
    const words = try make(allocator, 2, 16);
    defer allocator.free(words);
    const seed = try emptySeed(allocator, 2);
    defer allocator.free(seed);
    const capacity = (trace.pool_words - trace.header_words - 2 * words[6]) / trace.checkpointWords(words[6]);
    for (0..capacity) |_| checkpoint(words, seed);
    const retained = try allocator.dupe(u32, words);
    defer allocator.free(retained);
    seed[layout.miscOffset(2)] += 20;
    checkpoint(words, seed);
    try std.testing.expectEqual(@as(u32, 1), (try trace.View.init(words)).droppedCheckpoints());
    try std.testing.expectEqualSlices(u32, retained[trace.header_words..], words[trace.header_words..]);
    words[15] = std.math.maxInt(u32);
    checkpoint(words, seed);
    try std.testing.expectEqual(std.math.maxInt(u32), (try trace.View.init(words)).droppedCheckpoints());
    append(words, seed, seed, &@as([11]i32, @splat(-1)), false, 0);
    try std.testing.expectEqual(@as(u64, 1), (try trace.View.init(words)).dropped());
}

test "exl3: v4 trace preserves raw row order seed checkpoint and post counters" {
    const allocator = std.testing.allocator;
    const words = try make(allocator, 2, 2);
    defer allocator.free(words);
    const seed = try emptySeed(allocator, 2);
    defer allocator.free(seed);
    seed[layout.miscOffset(2)] = 0xfffffffe;
    const picks = [_]i32{ 7, 2, 7, -1, 512, 4, 2, 2, 0, -9, 800, 4, 7, 2, 511, -1, 512, 4, 0, 0, 7, 511 };
    append(words, seed, seed, &picks, true, 3);
    const view = try trace.View.init(words);
    const bytes = try view.encode(allocator);
    defer allocator.free(bytes);
    var decoded = try trace.Snapshot.decode(allocator, bytes);
    defer decoded.deinit();
    const restored = try decoded.view();
    const record = try restored.round(0);
    try std.testing.expectEqual(@as(usize, 2), record.rows());
    try std.testing.expectEqual(@as(u32, 16), record.validPicks());
    try std.testing.expectEqual(@as(usize, 5), record.uniqueCount());
    try std.testing.expect(record.recentEnabled());
    try std.testing.expectEqual(@as(u32, 3), record.mappedPicks());
    try std.testing.expectEqualSlices(u32, seed, restored.initialSeed());
    try std.testing.expectEqualSlices(u32, seed, restored.terminalSeed());
    try std.testing.expectEqual(@as(u32, 0xfffffffe), record.misc()[0]);
    for (picks, record.picks()) |expert, bits| try std.testing.expectEqual(expert, @as(i32, @bitCast(bits)));
    try std.testing.expectEqual(trace.magic, std.mem.readInt(u32, bytes[0..4], .little));
    try std.testing.expectEqual((trace.header_words + 2 * seed.len + words[7]) * 4, bytes.len);
    try std.testing.expectError(error.TraceRoundOutOfRange, view.round(1));
}

test "exl3: v4 capacity exceeds 1000 with full 16-row records before ownership checkpoints" {
    for ([_]usize{ 1, 10, 73, 256, 512 }) |lines| {
        const shape = try trace.header(0, 0, lines, 16);
        try std.testing.expect(shape[8] >= 1000);
        const used = trace.header_words + 2 * shape[6] + shape[8] * shape[7];
        try std.testing.expect(used <= trace.pool_words);
        try std.testing.expect(used + shape[7] > trace.pool_words);
        try std.testing.expectEqual(@as(u32, 191), shape[7]);
    }
    try std.testing.expectError(error.TraceShapeTooLarge, trace.header(0, 0, trace.pool_words, 1));
    try std.testing.expectError(error.TraceShapeTooLarge, trace.header(0, 0, 2, std.math.maxInt(usize)));
    try std.testing.expectError(error.TraceShapeTooLarge, trace.header(0, 0, 2, 0));
    try std.testing.expect(!trace.enabledValue(null));
    try std.testing.expect(!trace.enabledValue("true"));
    try std.testing.expect(!trace.enabledValue("0"));
    try std.testing.expect(trace.enabledValue("1"));
}

test "exl3: v4 keeps initial seed detects discontinuity and saturates bounded drops" {
    const allocator = std.testing.allocator;
    const words = try make(allocator, 2, 16);
    defer allocator.free(words);
    const seed = try emptySeed(allocator, 2);
    defer allocator.free(seed);
    const picks: [11]i32 = @splat(-1);
    append(words, seed, seed, &picks, false, 0);
    seed[layout.miscOffset(2)] = 700;
    append(words, seed, seed, &picks, false, 0);
    try std.testing.expectEqual(@as(u32, 1), words[12]);
    try std.testing.expectEqual(@as(u32, 0), (try trace.View.init(words)).initialSeed()[layout.miscOffset(2)]);
    for (2..words[8] + 3) |_| append(words, seed, seed, &picks, false, 0);
    var view = try trace.View.init(words);
    try std.testing.expectEqual(@as(usize, words[8]), view.recorded());
    try std.testing.expectEqual(@as(u64, 3), view.dropped());
    words[10] = std.math.maxInt(u32);
    words[11] = 4;
    append(words, seed, seed, &picks, false, 0);
    try std.testing.expectEqual(@as(u64, 5) << 32, (try trace.View.init(words)).dropped());
    words[10] = std.math.maxInt(u32);
    words[11] = std.math.maxInt(u32);
    append(words, seed, seed, &picks, false, 0);
    view = try trace.View.init(words);
    try std.testing.expectEqual(std.math.maxInt(u64), view.dropped());
}

test "exl3: v4 rejects metadata corruption truncation old versions and false pick counts" {
    const allocator = std.testing.allocator;
    const words = try make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try emptySeed(allocator, 2);
    defer allocator.free(seed);
    append(words, seed, seed, &@as([11]i32, @splat(7)), false, 0);
    const view = try trace.View.init(words);
    const bytes = try view.encode(allocator);
    defer allocator.free(bytes);
    try std.testing.expectError(error.InvalidTrace, trace.Snapshot.decode(allocator, bytes[0 .. bytes.len - 4]));
    try std.testing.expectError(error.InvalidTrace, trace.Snapshot.decode(allocator, bytes[0 .. bytes.len - 1]));
    const offset = view.recordsOffset();
    for ([_]usize{ 1, 2, 3, 14 }) |field| {
        const saved = words[offset + field];
        words[offset + field] = std.math.maxInt(u32);
        try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
        words[offset + field] = saved;
    }
    for ([_]usize{ 1, 6, 7, 8, 9, 12, 13, 14, 15 }) |field| {
        const saved = words[field];
        words[field] = std.math.maxInt(u32);
        try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
        words[field] = saved;
    }
    words[1] = 2;
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
    words[1] = 3;
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
    words[1] = trace.version;
    words[10] = 1;
    try std.testing.expectError(error.InvalidTrace, trace.View.init(words));
}

test "exl3: v4 raw picks cover all 512 experts beyond the planner unique bound" {
    const allocator = std.testing.allocator;
    const words = try make(allocator, 2, 47);
    defer allocator.free(words);
    const seed = try emptySeed(allocator, 2);
    defer allocator.free(seed);
    var picks: [47 * 11]i32 = @splat(512);
    for (0..512) |index| picks[index] = @intCast(511 - index);
    append(words, seed, seed, &picks, false, 0);
    const record = try (try trace.View.init(words)).round(0);
    try std.testing.expectEqual(@as(usize, 512), record.uniqueCount());
    try std.testing.expectEqual(@as(u32, 512), record.validPicks());
    for (picks, record.picks()) |expert, raw| try std.testing.expectEqual(expert, @as(i32, @bitCast(raw)));
}
