const std = @import("std");
const arena = @import("arena.zig");
const reader = @import("reader.zig");
const Fake = struct {
    data: []const u8,
    reads: usize = 0,
    checks: usize = 0,
    bytes: usize = 0,
    fail: bool = false,
    change: bool = false,
    pub fn fileSize(s: *@This(), file: usize) !u64 {
        if (file != 0) return error.NoFile;
        s.checks += 1;
        return s.data.len + @intFromBool(s.change and s.reads > 0);
    }
    pub fn readAt(s: *@This(), _: usize, dst: []u8, at: u64) !usize {
        s.reads += 1;
        if (s.fail) return error.ReadFailed;
        @memcpy(dst, s.data[@intCast(at)..][0..dst.len]);
        s.bytes += dst.len;
        return dst.len;
    }
};

test "FSR full prompt union coalesces adjacent experts and validates files once per batch" {
    const a = std.testing.allocator;
    const count = 512;
    var data: [9 * count * 4]u8 = undefined;
    for (&data, 0..) |*byte, i| byte.* = @truncate(i * 17);
    var ex: [count]arena.Expert = undefined;
    for (&ex, 0..) |*expert, key| for (expert, 0..) |*span, part| {
        span.* = .{ .file = 0, .offset = (part * count + key) * 4, .length = 4 };
    };
    var storage: [9][count * 4]u8 = undefined;
    var slabs: [9][]u8 = undefined;
    for (&slabs, &storage) |*slab, *bytes| slab.* = bytes;
    var source: Fake = .{ .data = &data };
    var s = try arena.Arena(*Fake).init(a, &source, &ex, @splat(4), slabs);
    defer s.deinit();
    source.reads = 0;
    source.checks = 0;
    const keys = try a.alloc(usize, 8192 * 10);
    defer a.free(keys);
    for (keys, 0..) |*key, i| key.* = (i * 37) % count;
    try s.demand(keys);
    try std.testing.expectEqual(@as(usize, 9), source.reads);
    try std.testing.expectEqual(@as(usize, 2), source.checks);
    try std.testing.expectEqual(data.len, source.bytes);
    for (0..count) |key| {
        const slot: usize = @intCast(s.clock.map[key]);
        for (slabs, 0..) |slab, part|
            try std.testing.expectEqualSlices(u8, data[(part * count + key) * 4 ..][0..4], slab[slot * 4 ..][0..4]);
    }
    try s.demand(keys);
    try std.testing.expectEqual(@as(usize, 9), source.reads);
    try std.testing.expectEqual(@as(usize, 2), source.checks);
}

test "FSR batched reads exclude sparse experts and unused mixed-width stride tails" {
    var data: [9 * 4 * 8]u8 = undefined;
    for (&data, 0..) |*byte, i| byte.* = @truncate(i);
    var ex: [4]arena.Expert = undefined;
    for (&ex, 0..) |*expert, key| for (expert, 0..) |*span, part| {
        span.* = .{ .file = 0, .offset = (part * 4 + key) * 8, .length = if (key == 0) 4 else 8 };
    };
    var memory: [9][16]u8 = @splat(@splat(0xa5));
    var slabs: [9][]u8 = undefined;
    for (&slabs, &memory) |*slab, *bytes| slab.* = bytes;
    var source: Fake = .{ .data = &data };
    var s = try arena.Arena(*Fake).init(std.testing.allocator, &source, &ex, @splat(8), slabs);
    defer s.deinit();
    try s.demand(&.{ 2, 0, 2 });
    try std.testing.expectEqual(@as(usize, 9 * 12), source.bytes);
    try std.testing.expectEqual(@as(usize, 18), source.reads);
    const slot: usize = @intCast(s.clock.map[0]);
    for (slabs, 0..) |slab, part| {
        try std.testing.expectEqualSlices(u8, data[part * 32 ..][0..4], slab[slot * 8 ..][0..4]);
        try std.testing.expect(std.mem.allEqual(u8, slab[slot * 8 + 4 ..][0..4], 0xa5));
    }
}

test "FSR failed batch aborts all new publications and keeps the previous leased bytes exact" {
    var data: [9 * 3 * 4]u8 = @splat(7);
    var ex: [3]arena.Expert = undefined;
    for (&ex, 0..) |*expert, key| for (expert, 0..) |*span, part| {
        span.* = .{ .file = 0, .offset = (part * 3 + key) * 4, .length = 4 };
    };
    var memory: [9][12]u8 = undefined;
    var slabs: [9][]u8 = undefined;
    for (&slabs, &memory) |*slab, *bytes| slab.* = bytes;
    var source: Fake = .{ .data = &data };
    var s = try arena.Arena(*Fake).init(std.testing.allocator, &source, &ex, @splat(4), slabs);
    defer s.deinit();
    try s.demand(&.{0});
    const before = memory;
    const held: usize = @intCast(s.clock.map[0]);
    source.fail = true;
    try std.testing.expectError(error.ReadFailed, s.demand(&.{ 0, 1, 2 }));
    try std.testing.expect(s.clock.map[1] < 0 and s.clock.map[2] < 0);
    for (memory, before) |actual, expected|
        try std.testing.expectEqualSlices(u8, expected[held * 4 ..][0..4], actual[held * 4 ..][0..4]);
    source.fail = false;
    source.reads = 0;
    source.change = true;
    try std.testing.expectError(error.CheckpointChanged, s.demand(&.{ 1, 2 }));
    try std.testing.expect(s.clock.map[1] < 0 and s.clock.map[2] < 0);
}

test "FSR coalesced requests remain bounded at the same four MiB maximum" {
    const a = std.testing.allocator;
    const size = reader.max_read_bytes + 4;
    const data = try a.alloc(u8, size + 8 * 4);
    defer a.free(data);
    @memset(data, 13);
    var source: Fake = .{ .data = data };
    var expert: [1]arena.Expert = undefined;
    expert[0][0] = .{ .file = 0, .offset = 0, .length = size };
    for (1..9) |part| expert[0][part] = .{ .file = 0, .offset = size + (part - 1) * 4, .length = 4 };
    var slabs: [9][]u8 = undefined;
    slabs[0] = try a.alloc(u8, size);
    defer a.free(slabs[0]);
    var small: [8][4]u8 = undefined;
    for (&small, 1..) |*bytes, part| slabs[part] = bytes;
    var strides: [9]usize = @splat(4);
    strides[0] = size;
    var s = try arena.Arena(*Fake).init(a, &source, &expert, strides, slabs);
    defer s.deinit();
    try s.demand(&.{0});
    try std.testing.expectEqual(@as(usize, 10), source.reads);
    try std.testing.expectEqualSlices(u8, data[0..size], slabs[0]);
    try std.testing.expectEqual(@as(usize, 2), source.checks - 1);
}
