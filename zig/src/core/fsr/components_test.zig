const std = @import("std");
const arena = @import("arena.zig");
const reader = @import("reader.zig");
const plan = @import("plan.zig");
const cache_file = @import("file.zig");
const a = std.testing.allocator;

const Source = struct {
    data: []const u8,
    checks: usize = 0,
    calls: usize = 0,
    chunk: usize = std.math.maxInt(usize),
    stop: usize = std.math.maxInt(usize),
    change: bool = false,
    started: std.atomic.Value(bool) = .init(false),
    pub fn fileSize(s: *@This(), file: usize) !u64 {
        if (file != 0) return error.NoFile;
        s.checks += 1;
        return s.data.len + @intFromBool(s.change and s.calls > 0);
    }
    pub fn readAt(s: *@This(), _: usize, dst: []u8, offset: u64) !usize {
        s.started.store(true, .release);
        s.calls += 1;
        if (s.calls > s.stop) return 0;
        const n = @min(dst.len, s.chunk);
        @memcpy(dst[0..n], s.data[@intCast(offset)..][0..n]);
        return n;
    }
};
const Six = arena.ArenaOf(*Source, 6);

fn spans() [4]Six.Expert {
    var out: [4]Six.Expert = undefined;
    for (&out, 0..) |*expert, key| for (expert, 0..) |*span, part| {
        span.* = .{ .file = 0, .offset = (part * 4 + key) * 4, .length = 4 };
    };
    return out;
}

test "FSR six-component plan charges its descriptors and complete layer union" {
    const parts: [6]u64 = .{ 5898240, 368640, 5898240, 368640, 5898240, 368640 };
    const stride: u64 = 18800640;
    const p = try plan.poolOf(6, 384 * stride + 13, parts, 15744, 384);
    try std.testing.expectEqual(@as(usize, 384), p.slots);
    try std.testing.expectEqual(stride, p.stride);
    try std.testing.expectEqual(384 * stride, p.slabs);
    try std.testing.expectEqual(@as(u64, 13), p.unused);
    const per_slot = @sizeOf(@import("cache.zig").Slot) + @sizeOf(Six.Job) + @sizeOf(usize);
    const per_key = 6 * (@sizeOf(reader.Span) + @sizeOf(u64)) + 2 * @sizeOf(i32) + 1 + @sizeOf(u64);
    try std.testing.expectEqual(384 * per_slot + 15744 * per_key + arena.worker_stack_bytes + 4096, p.metadata);
    try std.testing.expectError(error.ExpertPoolTooSmall, plan.poolOf(6, 384 * stride - 1, parts, 15744, 384));
    var bad = parts;
    bad[5] = 0;
    try std.testing.expectError(error.InvalidExpertStride, plan.poolOf(6, 8 << 30, bad, 15744, 384));
    bad[5] = @as(u64, std.math.maxInt(u32)) + 1;
    try std.testing.expectError(error.InvalidExpertStride, plan.poolOf(6, 8 << 30, bad, 15744, 384));
    try std.testing.expectError(error.InvalidPoolShape, plan.poolOf(6, 8 << 30, parts, 10, 11));
}

test "FSR six-component arena coalesces full demand and retains forecast ownership" {
    const ex = spans();
    var data: [96]u8 = undefined;
    for (&data, 0..) |*v, i| v.* = @truncate(i * 17);
    var src: Source = .{ .data = &data, .chunk = 3 };
    var memory: [6][12]u8 = @splat(@splat(0xa5));
    var slabs: [6][]u8 = undefined;
    for (&slabs, &memory) |*slab, *bytes| slab.* = bytes;
    var s = try Six.init(a, &src, &ex, @splat(4), slabs);
    defer s.deinit();
    src.checks = 0;
    try s.demand(&.{ 1, 0, 1 });
    try std.testing.expectEqual(@as(usize, 2), src.checks);
    try std.testing.expectEqual(@as(u64, 48), s.io.bytes);
    try std.testing.expectEqual(@as(u64, 2), s.demand_misses);
    for ([_]usize{ 0, 1 }) |key| {
        const slot: usize = @intCast(s.clock.map[key]);
        for (slabs, 0..) |slab, part|
            try std.testing.expectEqualSlices(u8, data[(part * 4 + key) * 4 ..][0..4], slab[slot * 4 ..][0..4]);
    }
    const held = s.clock.map[0];
    const calls = src.calls;
    try s.demand(&.{0});
    try std.testing.expectEqual(calls, src.calls);
    src.started.store(false, .release);
    try s.forecast(&.{2});
    while (!src.started.load(.acquire)) std.Thread.yield() catch {};
    try s.join();
    try std.testing.expectEqual(held, s.clock.map[0]);
    try std.testing.expect(s.clock.map[2] >= 0);
    try s.demand(&.{ 0, 2 });
    try std.testing.expectEqual(@as(u64, 24), s.io.bytes - 48);
}

test "FSR six-component arena refuses partial or changed payload before publication" {
    const ex = spans();
    const data: [96]u8 = @splat(7);
    var src: Source = .{ .data = &data };
    var memory: [6][8]u8 = @splat(@splat(0xa5));
    var slabs: [6][]u8 = undefined;
    for (&slabs, &memory) |*slab, *bytes| slab.* = bytes;
    var s = try Six.init(a, &src, &ex, @splat(4), slabs);
    defer s.deinit();
    try s.demand(&.{0});
    const before = memory;
    const held: usize = @intCast(s.clock.map[0]);
    src.stop = src.calls + 2;
    try std.testing.expectError(error.ShortRead, s.demand(&.{ 0, 1 }));
    try std.testing.expect(s.clock.map[1] < 0);
    for (memory, before) |actual, expected|
        try std.testing.expectEqualSlices(u8, expected[held * 4 ..][0..4], actual[held * 4 ..][0..4]);
    src.stop = std.math.maxInt(usize);
    src.calls = 0;
    src.change = true;
    try std.testing.expectError(error.CheckpointChanged, s.demand(&.{1}));
    try std.testing.expect(s.clock.map[1] < 0);
    try std.testing.expectError(error.ActiveUnionExceedsPool, s.demand(&.{ 0, 1, 2 }));
    var bad_slabs = slabs;
    bad_slabs[5] = slabs[5][0..7];
    try std.testing.expectError(error.InvalidSlab, Six.init(a, &src, &ex, @splat(4), bad_slabs));
    var bad = ex;
    bad[3][5].length = 5;
    try std.testing.expectError(error.InvalidSlab, Six.init(a, &src, &bad, @splat(4), slabs));
}

test "FSR six-component owners require the same fail-closed uncached file policy" {
    const C = struct {
        pub const fd_t = c_int;
        pub const F = struct {
            pub const NOCACHE = 48;
        };
        var fail = false;
        var calls: usize = 0;
        pub fn fcntl(fd: c_int, command: c_int, value: c_int) c_int {
            std.debug.assert(fd == 7 and command == 48 and value == 1);
            calls += 1;
            return if (fail) -1 else 0;
        }
    };
    C.fail = false;
    C.calls = 0;
    try cache_file.requireCacheBypass(C, 7);
    C.fail = true;
    try std.testing.expectError(error.CacheBypassFailed, cache_file.requireCacheBypass(C, 7));
    try std.testing.expectEqual(@as(usize, 2), C.calls);
}
