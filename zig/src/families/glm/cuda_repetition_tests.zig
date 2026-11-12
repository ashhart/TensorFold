const std = @import("std");
const rep = @import("cuda_repetition.zig");
const a = std.testing.allocator;

test "GLM repetition: table installation refuses gapped positions before replacing the installed table" {
    var table: rep.Table = .{};
    try table.set(&.{.{ .slot = 7, .position = 5, .factor = 1.1 }}, 65, 3, 8, 100);
    try std.testing.expectError(error.BadProcessorTable, table.set(&.{
        .{ .slot = 0, .position = 10, .factor = 1.1 },
        .{ .slot = 0, .position = 12, .factor = 1.1 },
    }, 65, 3, 8, 100));
    try std.testing.expectEqual(@as(?usize, 0), try table.find(7, 5, 1));
}

test "GLM repetition: table installation refuses interleaved slot groups" {
    var table: rep.Table = .{};
    try std.testing.expectError(error.BadProcessorTable, table.set(&.{
        .{ .slot = 0, .position = 10, .factor = 1.1 },
        .{ .slot = 1, .position = 10, .factor = 1.1 },
        .{ .slot = 0, .position = 11, .factor = 1.1 },
    }, 65, 3, 8, 100));
}

test "GLM repetition: table installation refuses a factor change within one slot" {
    var table: rep.Table = .{};
    try std.testing.expectError(error.BadProcessorTable, table.set(&.{
        .{ .slot = 0, .position = 10, .factor = 1.1 },
        .{ .slot = 0, .position = 11, .factor = 1.2 },
    }, 65, 3, 8, 100));
}

test "GLM repetition: row prefixes include pending and earlier held drafts without future draft pollution" {
    var h = try rep.History.init(a, 65, &.{ 0, 32 });
    defer h.deinit(a);
    var s = try rep.Stage.init(65);
    defer s.deinit(a);
    try s.add(a, 7, 3, 1.1, &h, &.{ 1, 64, 2 });
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 0, 3, 1, 1, 7, 1, 1 }, s.masks.items);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 0 }, h.bits);
    try std.testing.expectEqual(@as(u32, 3), s.rows[0].position);
    try std.testing.expectEqual(@as(u32, 5), s.rows[2].position);
}

test "GLM repetition: partial acceptance commits actual context and leaves rejected draft absent" {
    var h = try rep.History.init(a, 64, &.{0});
    defer h.deinit(a);
    var s = try rep.Stage.init(64);
    defer s.deinit(a);
    try s.add(a, 0, 2, 2, &h, &.{ 1, 2, 3 });
    try h.sync(a, &.{ 0, 1 });
    s.reset();
    try s.add(a, 0, 3, 2, &h, &.{4});
    try std.testing.expectEqualSlices(u32, &.{ 19, 0 }, s.masks.items);
    try std.testing.expectEqualSlices(u32, &.{ 3, 0 }, h.bits);
}

test "GLM repetition: shortened and replaced committed prefixes rebuild from actual IDs" {
    var h = try rep.History.init(a, 64, &.{ 0, 1, 2 });
    defer h.deinit(a);
    try h.sync(a, &.{ 0, 1, 2, 3 });
    try h.sync(a, &.{0});
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, h.bits);
    try h.sync(a, &.{32});
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, h.bits);
}

test "GLM repetition: mixed slots retain separate masks and factors including identity" {
    var h = try rep.History.init(a, 32, &.{4});
    defer h.deinit(a);
    var s = try rep.Stage.init(32);
    defer s.deinit(a);
    try s.add(a, 0, 2, 1, null, &.{1});
    try s.add(a, 7, 2, 0.5, &h, &.{2});
    try s.add(a, 3, 2, 1, null, &.{3});
    try std.testing.expectEqualSlices(u32, &.{ 0, 20, 0 }, s.masks.items);
    try std.testing.expectEqual(@as(f32, 1), s.rows[0].factor);
    try std.testing.expectEqual(@as(f32, 0.5), s.rows[1].factor);
    try std.testing.expectEqual(@as(usize, 3), s.descriptors().len);
}

test "GLM repetition: identity after active resets same slot and position without stale masks" {
    var h = try rep.History.init(a, 32, &.{4});
    defer h.deinit(a);
    var s = try rep.Stage.init(32);
    defer s.deinit(a);
    try s.add(a, 0, 8, 2, &h, &.{});
    var t: rep.Table = .{};
    try t.set(s.descriptors(), 32, 1, 8, 100);
    try std.testing.expectEqual(@as(?usize, 0), try t.find(0, 8, 1));
    s.reset();
    try s.add(a, 0, 8, 1, null, &.{});
    try t.set(s.descriptors(), 32, 1, 8, 100);
    try std.testing.expectEqual(@as(?usize, null), try t.find(0, 8, 1));
}

test "GLM repetition: wire rejects malformed length geometry slot position factor and duplicate keys" {
    const good = [_]u32{ 7, 100, 0x3f8ccccd, 0 };
    const t = try rep.decode(&good, 1, 65, 3, 8, 1000);
    try std.testing.expectEqual(@as(?usize, 0), try t.find(7, 100, 1));
    try std.testing.expectError(error.BadProcessorTable, rep.decode(good[0..3], 1, 65, 3, 8, 1000));
    try std.testing.expectError(error.BadProcessorTable, rep.decode(&good, 1, 65, 2, 8, 1000));
    try std.testing.expectError(error.BadProcessorTable, rep.decode(&good, 1, 65, 3, 7, 1000));
    try std.testing.expectError(error.BadProcessorTable, rep.decode(&good, 1, 65, 3, 8, 99));
    for ([_]u32{ 0, 0x80000000, 0xbf800000, 0x7f800000, 0x7fc00000 }) |bad| {
        const row = [_]u32{ 0, 1, bad, 0 };
        try std.testing.expectError(error.BadProcessorTable, rep.decode(&row, 1, 65, 3, 8, 1000));
    }
    try std.testing.expectError(error.BadProcessorTable, rep.decode(&(good ++ good), 2, 65, 3, 8, 1000));
    try std.testing.expectError(error.ProcessorPositionMismatch, t.find(7, 101, 1));
    try std.testing.expectError(error.ProcessorPositionMismatch, t.find(7, 100, 2));
    const huge = [_]u32{ 0, 0x80000000, 0x3f800000, 0 };
    try std.testing.expectError(error.BadProcessorTable, rep.decode(&huge, 1, 65, 3, 8, 0xffffffff));
}

test "GLM repetition: invalid token IDs are refused before modifying committed or staged state" {
    try std.testing.expectError(error.InvalidToken, rep.History.init(a, 32, &.{32}));
    var h = try rep.History.init(a, 32, &.{0});
    defer h.deinit(a);
    try std.testing.expectError(error.InvalidToken, h.sync(a, &.{ 0, 32 }));
    try std.testing.expectEqualSlices(u32, &.{1}, h.bits);
    var s = try rep.Stage.init(32);
    defer s.deinit(a);
    try std.testing.expectError(error.InvalidToken, s.add(a, 0, 2, 2, &h, &.{32}));
    try std.testing.expectEqual(@as(usize, 0), s.len);
}

test "GLM repetition: identity table avoids allocating history and membership storage" {
    var s = try rep.Stage.init(100000);
    defer s.deinit(a);
    try s.add(a, 0, 1000000, 1, null, &.{});
    try std.testing.expectEqual(@as(usize, 0), s.masks.capacity);
    try std.testing.expectEqual(@as(usize, 0), s.descriptors().len);
}

test "GLM repetition: eight slots fill a 32 row batch with independent masks and an exact bound" {
    var s = try rep.Stage.init(32);
    defer s.deinit(a);
    const expected = [_]u32{ 0xf0001, 0xf0002, 0xf0004, 0xf0008, 0xf0010, 0xf0020, 0xf0040, 0xf0080 };
    for (0..8) |slot| {
        var h = try rep.History.init(a, 32, &.{@intCast(slot)});
        defer h.deinit(a);
        try s.add(a, @intCast(slot), 10, if (slot % 2 == 0) 1.1 else 0.5, &h, &.{ 16, 17, 18, 19 });
        try std.testing.expectEqual(expected[slot], s.masks.items[slot * 4 + 3]);
    }
    var table: rep.Table = .{};
    try table.set(s.descriptors(), 32, 1, 8, 100);
    for (0..8) |slot| try std.testing.expectEqual(@as(?usize, slot * 4), try table.find(@intCast(slot), 10, 4));
    try std.testing.expectError(error.BadProcessorTable, s.add(a, 8, 10, 1, null, &.{}));
    try std.testing.expectEqual(@as(usize, 32), s.len);
}

test "GLM repetition: forced committed tokens enter history while original prompt membership stays immutable" {
    var h = try rep.History.init(a, 32, &.{ 0, 1 });
    defer h.deinit(a);
    try h.sync(a, &.{ 0, 1, 10 });
    try std.testing.expectEqualSlices(u32, &.{1027}, h.bits);
    try std.testing.expectEqualSlices(u32, &.{3}, h.prompt_bits);
}
