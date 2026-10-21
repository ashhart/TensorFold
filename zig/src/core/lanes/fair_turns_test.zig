//! Shared row packing preserves waiting age and the complete record of rounds that already fit.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const model = @import("engine_test.zig").modelWith;
const events = @import("events.zig");
const t = std.testing;
const a = t.allocator;
const names = [_][]const u8{ "a", "b", "c", "d", "e", "f", "g", "h" };

test "shared row packing: oversized forced windows retain age and run next round" {
    for ([_]usize{ 3, 8 }) |count| {
        var cfg = try model(false);
        defer cfg.deinit(a);
        cfg.batch_rows = 6;
        cfg.batch_streams = 4;
        cfg.base_width = 6;
        cfg.pipelined = false;
        cfg.timed = false;
        var target: fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        var streams: [8]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(a);
        for (streams[0..count], names[0..count]) |*s, name| {
            s.* = try Stream.init(a, .{ .id = name, .prompt = &.{ 3, 1, 4 }, .max_new = 64 });
            made += 1;
            try e.addStream(s);
        }
        try streams[0].force.append(a, 5);
        const forced = [_]u32{ 7, 9, 11, 13 };
        try streams[1].force.appendSlice(a, &forced);
        try e.step();
        try t.expectEqual(@as(?u64, 1), streams[0].served);
        try t.expectEqual(@as(?u64, null), streams[1].served);
        try t.expectEqual(@as(?u64, 1), streams[2].served);
        try t.expectEqual(@as(usize, if (count == 3) 3 else 5), target.last_rows);
        try t.expectEqualSlices(u32, &forced, streams[1].force.items);
        try e.step();
        try t.expectEqual(@as(?u64, 2), streams[1].served);
        try t.expectEqual(@as(usize, 6), target.last_rows);
        try t.expectEqual(@as(usize, 0), streams[1].force.items.len);
        try t.expect(std.mem.indexOf(u32, streams[1].emitted(), &forced) != null);
        for (0..count) |_| {
            var waiting = false;
            for (streams[0..count]) |s| waiting = waiting or s.served == null;
            if (!waiting) break;
            try e.step();
        }
        for (streams[0..count]) |s| try t.expect(s.served != null and s.served.? <= count + 2);
    }
}

fn fitRecord(sampling: ?@import("sampling.zig").Sampling) ![32]u8 {
    var cfg = try model(false);
    defer cfg.deinit(a);
    cfg.batch_rows = 8;
    cfg.base_width = 6;
    cfg.pipelined = false;
    cfg.timed = false;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var log = events.Log.init(a);
    defer log.deinit();
    e.log = &log;
    var streams: [3]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams, names[0..3]) |*s, name| {
        s.* = try Stream.init(a, .{ .id = name, .prompt = &.{ 3, 1, 4 }, .max_new = 32, .sampling = sampling });
        made += 1;
        try e.addStream(s);
    }
    try streams[0].force.append(a, 5);
    try streams[1].force.appendSlice(a, &.{ 7, 9, 11, 13 });
    for (0..4) |_| try e.step();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..log.events.items.len) |i| {
        const line = try log.line(a, i);
        defer a.free(line);
        hash.update(line);
        hash.update("\n");
    }
    for (&streams) |*s| for (s.emitted()) |token| {
        var word: [4]u8 = undefined;
        std.mem.writeInt(u32, &word, token, .little);
        hash.update(&word);
    };
    return hash.finalResult();
}
test "shared row packing: fitting greedy and sampled rounds preserve the preceding byte records" {
    const greedy = std.fmt.bytesToHex(try fitRecord(null), .lower);
    const sampled = std.fmt.bytesToHex(try fitRecord(.{ .seed = 1234 }), .lower);
    try t.expectEqualStrings("f33023f45dc9c6f63d0cb453cd067a92ffd4ca45212cb54bae04e8d376e88bb7", &greedy);
    try t.expectEqualStrings("ba3178cff18f4ddf2e9467f75a1046003736f4c7abcbcd35afbda82f35ba3fe2", &sampled);
}
