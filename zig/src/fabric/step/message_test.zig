//! The step message's sizes, its checked arithmetic, and a decoder that refuses by name and never reads past a bound.
const std = @import("std");
const message = @import("message.zig");
const section = @import("section.zig");
const fixtures = @import("fixtures.zig");
const t = std.testing;
const max32 = std.math.maxInt(u32);

const roomy: message.Limits = .{ .bytes = 1 << 20, .streams = 1 << 10, .rows = max32, .context = max32 };

test "step message: size is what write writes, for every good fixture" {
    inline for (fixtures.all) |f| if (f.want == null) {
        var buf: [1024]u8 = undefined;
        const bytes = try f.build(&buf);
        const s = try message.decode(bytes, fixtures.limits);
        try t.expectEqual(@as(u64, bytes.len), 64 + s.header.bytes);
    };
}

test "step message: the keep-alive is the header alone, and the ring's round reads back entry for entry" {
    var buf: [256]u8 = undefined;
    const ka = try message.encode(.{ .header = .{ .kind = .round, .round = 9 } }, fixtures.limits, &buf);
    try t.expectEqual(@as(usize, 64), ka.len);
    try t.expect((try message.decode(ka, fixtures.limits)).keepAlive());
    var buf2: [256]u8 = undefined;
    const p = try message.encode(.{ .header = .{ .kind = .round, .round = 3 }, .streams = &fixtures.streams, .cuts = &fixtures.cuts }, fixtures.limits, &buf2);
    const s = try message.decode(p, fixtures.limits);
    try t.expect(!s.keepAlive());
    try t.expectEqual(@as(u32, 7), s.header.rows);
    try t.expectEqual(@as(u32, 4), s.header.streams);
    const map = s.rowMap().?;
    for (fixtures.streams, 0..) |want, i| try t.expectEqual(want, map.stream(i));
    try t.expectEqual(@as(u32, 4), s.get(.cuts).?.word(0));
    try t.expectEqual(@as(usize, 64 + 8 + 4 * 28 + 8 + 8), p.len); // the design's 64 + 8 + 28n, a cut and its padding
}

test "step message: encode refuses what decode refuses, and writes nothing a receiver would not take" {
    var buf: [256]u8 = undefined;
    const bad = [_]section.Stream{.{ .id = 1, .op = .open, .first = 2, .rows = 1 }};
    try t.expectError(error.BadOpen, message.encode(.{ .header = .{ .kind = .round, .round = 1 }, .streams = &bad }, fixtures.limits, &buf));
    try t.expectError(error.NoRoom, message.encode(.{ .header = .{ .kind = .round, .round = 1 }, .streams = &fixtures.streams }, fixtures.limits, buf[0..100]));
}

test "step message: design section 2's sizes, a 39-drop header inside the 1,024-byte inline limit" {
    var drops: [80]section.Snapshot = undefined;
    for (&drops, 0..) |*d, i| d.* = .{ .op = .drop, .id = i };
    const m80: message.Message = .{ .header = .{ .kind = .state, .round = 1 }, .snapshots = &drops };
    try t.expectEqual(@as(u64, 1992), try message.size(m80));
    const m40: message.Message = .{ .header = .{ .kind = .state, .round = 1 }, .snapshots = drops[0..40] };
    try t.expectEqual(@as(u64, 1032), try message.size(m40));
    const m39: message.Message = .{ .header = .{ .kind = .state, .round = 1 }, .snapshots = drops[0..39] };
    try t.expect(try message.size(m39) <= 1024);
}

test "step message: maximum counts are refused by name, with no overflow" {
    var buf: [256]u8 = undefined;
    // a section whose count claims maxInt(u32) entries on a short payload
    const p = try message.write(.{ .header = .{ .kind = .round, .round = 1 }, .streams = &fixtures.streams }, &buf);
    std.mem.writeInt(u32, p[68..72], max32, .little);
    try t.expectError(error.BadLength, message.decode(p, roomy));
    // a page patch whose n claims maxInt(u32) ids
    const ids = [_]u32{ 1, 2 };
    const q = try message.write(.{ .header = .{ .kind = .state, .round = 1 }, .pages = &.{.{ .stream = 1, .first = 0, .ids = &ids }} }, &buf);
    std.mem.writeInt(u32, q[72 + 8 ..][0..4], max32, .little);
    try t.expectError(error.BadLength, message.decode(q, roomy));
    // two streams whose rows sum past u32: the writer refuses, and so does a receiver of planted bytes
    const two = [_]section.Stream{ .{ .id = 1, .op = .@"continue", .first = 0, .rows = max32 }, .{ .id = 2, .op = .@"continue", .first = 0, .rows = 2 } };
    try t.expectError(error.TooLarge, message.write(.{ .header = .{ .kind = .round, .round = 1 }, .streams = &two }, &buf));
    const one = [_]section.Stream{ two[0], .{ .id = 2, .op = .@"continue", .first = 0, .rows = 0 } };
    const r = try message.write(.{ .header = .{ .kind = .round, .round = 1 }, .streams = &one }, &buf);
    std.mem.writeInt(u32, r[72 + 28 + 12 ..][0..4], 2, .little); // the second stream's rows, planted
    try t.expectError(error.TooManyRows, message.decode(r, roomy)); // the u64 sum passes the largest u32 limit
}

test "step message: every truncation and every byte flip of each good fixture is taken or refused, never a panic" {
    inline for (fixtures.all) |f| if (f.want == null) {
        var buf: [1024]u8 = undefined;
        const good = try f.build(&buf);
        for (0..good.len) |n| _ = message.decode(good[0..n], fixtures.limits) catch {};
        var flip: [1024]u8 = undefined;
        for (0..good.len) |i| for ([_]u8{ 0x01, 0x80, 0xFF }) |x| {
            @memcpy(flip[0..good.len], good);
            flip[i] ^= x;
            _ = message.decode(flip[0..good.len], fixtures.limits) catch {};
        };
    };
}
