const std = @import("std");
const r = @import("reader.zig");
const Job = struct { ticket: struct { key: usize, slot: usize } };
const Source = struct {
    data: []const u8,
    reads: [32]r.Span = undefined,
    count: usize = 0,
    groups: [8]usize = undefined,
    ngroups: usize = 0,
    pub fn fileSize(s: *@This(), _: usize) !u64 {
        return s.data.len;
    }
    pub fn readMany(s: *@This(), requests: []const r.Request, stats: *r.Stats) !void {
        std.debug.assert(requests.len <= 9);
        s.groups[s.ngroups] = requests.len;
        s.ngroups += 1;
        for (requests) |req| {
            s.reads[s.count] = req.span;
            s.count += 1;
            @memcpy(req.destination, s.data[@intCast(req.span.offset)..][0..req.span.length]);
            stats.calls += 1;
            stats.bytes += req.span.length;
        }
    }
};

test "FSR six-component reader merges only contiguous source and destination bytes in component order" {
    var data: [72]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i * 13);
    var experts: [3][6]r.Span = undefined;
    for (&experts, 0..) |*ex, key| for (ex, 0..) |*span, part| {
        span.* = .{ .file = 0, .offset = part * 12 + key * 4, .length = 4 };
    };
    var memory: [6][12]u8 = undefined;
    var slabs: [6][]u8 = undefined;
    for (&slabs, &memory) |*slab, *bytes| slab.* = bytes;
    const cases = [_][2]Job{
        .{ .{ .ticket = .{ .key = 0, .slot = 0 } }, .{ .ticket = .{ .key = 1, .slot = 1 } } },
        .{ .{ .ticket = .{ .key = 0, .slot = 0 } }, .{ .ticket = .{ .key = 1, .slot = 2 } } },
        .{ .{ .ticket = .{ .key = 0, .slot = 0 } }, .{ .ticket = .{ .key = 2, .slot = 1 } } },
    };
    for (cases, 0..) |jobs, ci| {
        for (&memory) |*bytes| @memset(bytes, 0xa5);
        var source: Source = .{ .data = &data };
        var sizes: [1]u64 = undefined;
        var stats: r.Stats = .{};
        try r.batchOf(6, &source, &jobs, &experts, @splat(4), slabs, &sizes, &stats);
        try std.testing.expectEqual(@as(usize, if (ci == 0) 6 else 12), source.count);
        try std.testing.expectEqual(@as(u64, 48), stats.logical_bytes);
        try std.testing.expectEqual(@as(u64, 48), stats.bytes);
        try std.testing.expectEqual(@as(u64, 2), stats.size_checks);
        try std.testing.expectEqualSlices(usize, if (ci == 0) &.{6} else &.{ 9, 3 }, source.groups[0..source.ngroups]);
        for (0..6) |part| {
            const at = part * (if (ci == 0) @as(usize, 1) else 2);
            try std.testing.expectEqual(@as(u64, part * 12), source.reads[at].offset);
            try std.testing.expectEqual(@as(usize, if (ci == 0) 8 else 4), source.reads[at].length);
            for (jobs) |job|
                try std.testing.expectEqualSlices(u8, data[part * 12 + job.ticket.key * 4 ..][0..4], slabs[part][job.ticket.slot * 4 ..][0..4]);
            if (ci == 1) try std.testing.expect(std.mem.allEqual(u8, slabs[part][4..8], 0xa5));
        }
    }
}

test "FSR six-component reader keeps padding unread and large requests bounded" {
    const a = std.testing.allocator;
    const large = r.max_read_bytes + 4;
    const data = try a.alloc(u8, large + 5 * 4);
    defer a.free(data);
    @memset(data, 13);
    var experts: [1][6]r.Span = undefined;
    experts[0][0] = .{ .file = 0, .offset = 0, .length = large };
    for (1..6) |part| experts[0][part] = .{ .file = 0, .offset = large + (part - 1) * 4, .length = 4 };
    var slabs: [6][]u8 = undefined;
    slabs[0] = try a.alloc(u8, large);
    defer a.free(slabs[0]);
    var small: [5][8]u8 = @splat(@splat(0xa5));
    for (&small, 1..) |*bytes, part| slabs[part] = bytes;
    var strides: [6]usize = @splat(8);
    strides[0] = large;
    var source: Source = .{ .data = data };
    var sizes: [1]u64 = undefined;
    var stats: r.Stats = .{};
    const jobs = [_]Job{.{ .ticket = .{ .key = 0, .slot = 0 } }};
    try r.batchOf(6, &source, &jobs, &experts, strides, slabs, &sizes, &stats);
    try std.testing.expectEqual(@as(usize, 7), source.count);
    try std.testing.expectEqual(r.max_read_bytes, source.reads[0].length);
    try std.testing.expectEqual(@as(usize, 4), source.reads[1].length);
    try std.testing.expectEqualSlices(u8, data[0..large], slabs[0]);
    for (small) |bytes| {
        try std.testing.expect(std.mem.allEqual(u8, bytes[0..4], 13));
        try std.testing.expect(std.mem.allEqual(u8, bytes[4..8], 0xa5));
    }
}
