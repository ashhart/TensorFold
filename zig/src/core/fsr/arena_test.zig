const std = @import("std");
const arena = @import("arena.zig");
const Fake = struct {
    data: []const u8,
    fail_at: usize = std.math.maxInt(usize),
    pub fn fileSize(f: Fake, _: usize) !u64 {
        return f.data.len;
    }
    pub fn readAt(f: Fake, _: usize, out: []u8, at: u64) !usize {
        if (at >= f.fail_at) return error.ReadFailed;
        const n = @min(out.len, @min(5, f.data.len - @as(usize, @intCast(at))));
        @memcpy(out[0..n], f.data[@intCast(at)..][0..n]);
        return n;
    }
};
const A = arena.Arena(Fake);

fn experts() [6]arena.Expert {
    var out: [6]arena.Expert = undefined;
    for (&out, 0..) |*expert, e| {
        for (expert, 0..) |*span, j| span.* = .{ .file = 0, .offset = e * 36 + j * 4, .length = 4 };
    }
    return out;
}

test "FSR arena forecasts behind demand and publishes the original nine components" {
    const a = std.testing.allocator;
    var data: [6 * 36]u8 = undefined;
    for (&data, 0..) |*v, i| v.* = @truncate(i);
    const ex = experts();
    var memory: [9][3 * 4]u8 = undefined;
    var slabs: [9][]u8 = undefined;
    for (&slabs, &memory) |*dst, *src| dst.* = src;
    var s = try A.init(a, .{ .data = &data }, &ex, @splat(4), slabs);
    defer s.deinit();
    try std.testing.expectError(error.InvalidExpert, s.forecast(&.{ 1, 6 }));
    try std.testing.expectEqual(@as(usize, 0), s.job_count);
    try std.testing.expectError(error.NoExperts, s.demand(&.{}));
    try s.demand(&.{ 0, 0 });
    const fixed = s.clock.map[0];
    try s.forecast(&.{ 1, 2 });
    try s.join();
    try std.testing.expectEqual(fixed, s.clock.map[0]);
    try s.demand(&.{ 1, 2, 1 });
    for ([_]usize{ 1, 2 }) |e| {
        const at: usize = @intCast(s.clock.map[e]);
        for (slabs, 0..) |slab, j| try std.testing.expectEqualSlices(u8, data[e * 36 + j * 4 ..][0..4], slab[at * 4 ..][0..4]);
    }
    try s.forecast(&.{ 3, 4, 5 });
    try s.join();
    try std.testing.expect(s.clock.map[1] >= 0 and s.clock.map[2] >= 0);
    try s.demand(&.{ 3, 4, 5 });
    for ([_]usize{ 3, 4, 5 }) |e| try std.testing.expect(s.clock.map[e] >= 0);
}

test "FSR failed component cannot publish an expert or overwrite an owned consumer" {
    const ex = experts();
    const data: [6 * 36]u8 = @splat(7);
    var memory: [9][2 * 4]u8 = undefined;
    var slabs: [9][]u8 = undefined;
    for (&slabs, &memory) |*dst, *src| dst.* = src;
    var s = try A.init(std.testing.allocator, .{ .data = &data, .fail_at = 40 }, &ex, @splat(4), slabs);
    defer s.deinit();
    try s.demand(&.{0});
    try std.testing.expectError(error.ReadFailed, s.demand(&.{1}));
    try std.testing.expectEqual(@as(i32, -1), s.clock.map[1]);
    try std.testing.expect(s.clock.map[0] >= 0);
    try std.testing.expectError(error.ActiveUnionExceedsPool, s.demand(&.{ 0, 1, 2 }));
}

test "FSR demand waits for its forecast reads and cancels unrelated reads between experts" {
    const Gate = struct {
        data: [6 * 36]u8 = @splat(7),
        enabled: bool = false,
        entered: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        bytes: usize = 0,
        pub fn fileSize(s: *@This(), _: usize) !u64 {
            return s.data.len;
        }
        pub fn readAt(s: *@This(), _: usize, dst: []u8, at: u64) !usize {
            if (s.enabled and !s.entered.swap(true, .acq_rel)) {
                while (!s.release.load(.acquire)) std.Thread.yield() catch {};
            }
            @memcpy(dst, s.data[@intCast(at)..][0..dst.len]);
            s.bytes += dst.len;
            return dst.len;
        }
    };
    const Owned = arena.Arena(*Gate);
    const Request = struct {
        pool: *Owned,
        failure: ?anyerror = null,
        fn run(s: *@This()) void {
            s.pool.demand(&.{3}) catch |err| {
                s.failure = err;
            };
        }
    };
    var source: Gate = .{};
    const ex = experts();
    var memory: [9][6 * 4]u8 = undefined;
    var slabs: [9][]u8 = undefined;
    for (&slabs, &memory) |*dst, *src| dst.* = src;
    var s = try Owned.init(std.testing.allocator, &source, &ex, @splat(4), slabs);
    defer s.deinit();
    try s.demand(&.{0});
    source.enabled = true;
    source.bytes = 0;
    try s.forecast(&.{ 1, 2, 3, 4, 5 });
    errdefer source.release.store(true, .release);
    while (!source.entered.load(.acquire)) std.Thread.yield() catch {};
    var request: Request = .{ .pool = &s };
    const caller = try std.Thread.spawn(.{}, Request.run, .{&request});
    while (!s.urgent.load(.acquire)) std.Thread.yield() catch {};
    source.release.store(true, .release);
    caller.join();
    try std.testing.expect(request.failure == null);
    try std.testing.expectEqual(@as(usize, 72), source.bytes);
    try std.testing.expectEqual(@as(u64, 3), s.cancelled_forecasts);
    try std.testing.expect(s.clock.map[1] >= 0 and s.clock.map[3] >= 0);
    for ([_]usize{ 2, 4, 5 }) |key| try std.testing.expect(s.clock.map[key] < 0 and s.clock.pending(key) == null);
}
