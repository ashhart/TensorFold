//! Repetition processor metadata, coordinated preparation and refusal before TP mask transport.
const std = @import("std");
const repetition = @import("cuda_repetition.zig");
const Header = @import("cuda_follow.zig").Header;
const Kind = @import("cuda_follow.zig").Kind;

pub fn decodeTable(h: Header, ids: []const u32, vocab: usize, slots: usize, capacity: usize) !repetition.Table {
    const words = (vocab + 31) / 32;
    if (h.flags != 1 or h.a != vocab or h.seed != words or h.n != ids.len) return error.BadProcessorTable;
    return repetition.decode(ids, h.depth, vocab, words, slots, capacity);
}

pub fn vote(e: anytype, refused: bool) !bool {
    if (try e.shared(@floatFromInt(@intFromBool(refused))) == 0) return true;
    e.processor_rows.table.len = 0;
    return false;
}

pub fn apply(e: anytype, link: anytype, h: Header, ids: []const u32) !bool {
    const words = (e.c.vocab + 31) / 32;
    const ready = preparation: {
        const table = decodeTable(h, ids, e.c.vocab, e.seqs.items.len, e.capacity) catch break :preparation false;
        e.prepareProcessors(table, words) catch break :preparation false;
        break :preparation true;
    };
    if (!try vote(e, !ready)) return true;
    if (e.processor_rows.table.len != 0) {
        try link.rows(e.processor_rows.masks.?.ptr, e.processor_rows.table.len * words * 4);
    }
    return true;
}

test "GLM repetition: processor refusal keeps followers receiving and never broadcasts an unprepared mask" {
    const Fake = struct {
        c: struct { vocab: usize = 65 } = .{},
        seqs: struct { items: [8]u8 = @splat(0) } = .{},
        capacity: usize = 100,
        processor_rows: struct {
            table: repetition.Table = .{},
            masks: ?struct { ptr: u64 = 1 } = .{},
        } = .{},
        allocation_failed: bool = false,
        peer_refused: bool = false,
        votes: usize = 0,
        payloads: usize = 0,
        pub fn prepareProcessors(e: *@This(), table: repetition.Table, _: usize) !void {
            if (e.allocation_failed) return error.OutOfMemory;
            e.processor_rows.table = table;
        }
        pub fn shared(e: *@This(), local: f64) !f64 {
            e.votes += 1;
            return @max(local, @as(f64, @floatFromInt(@intFromBool(e.peer_refused))));
        }
        pub fn rows(e: *@This(), _: u64, _: usize) !void {
            e.payloads += 1;
        }
    };
    const h: Header = .{ .kind = @backingInt(Kind.processor), .n = 4, .depth = 1, .flags = 1, .a = 65, .seed = 3 };
    const ids = [_]u32{ 7, 10, 0x3f8ccccd, 0 };
    for (0..6) |failure| {
        var e: Fake = .{ .allocation_failed = failure == 0, .peer_refused = failure == 1 };
        var header = h;
        if (failure == 2) header.flags = 0x101;
        var wire = [_]u32{ 0, 10, 0x3f8ccccd, 0, 0, 11, 0x3f8ccccd, 0, 0, 12, 0x3f8ccccd, 0 };
        if (failure >= 3) {
            header.depth = if (failure == 4) 3 else 2;
            header.n = header.depth * 4;
            if (failure == 3) wire[5] = 12;
            if (failure == 4) wire[4] = 1;
            if (failure == 5) wire[6] = @bitCast(@as(f32, 1.2));
        }
        const payload: []const u32 = if (failure >= 3) wire[0..header.n] else &ids;
        try std.testing.expect(try apply(&e, &e, header, payload));
        try std.testing.expectEqual(@as(usize, 1), e.votes);
        try std.testing.expectEqual(@as(usize, 0), e.payloads);
        try std.testing.expectEqual(@as(usize, 0), e.processor_rows.table.len);
        e.allocation_failed = false;
        e.peer_refused = false;
        try std.testing.expect(try apply(&e, &e, h, &ids));
        try std.testing.expectEqual(@as(usize, 1), e.payloads);
    }
}

test "GLM repetition: processor wire validates its version and geometry and supports an explicit empty reset" {
    const h: Header = .{ .kind = @backingInt(Kind.processor), .n = 4, .depth = 1, .flags = 1, .a = 65, .seed = 3 };
    const ids = [_]u32{ 7, 10, 0x3f8ccccd, 0 };
    const t = try decodeTable(h, &ids, 65, 8, 100);
    try std.testing.expectEqual(@as(?usize, 0), try t.find(7, 10, 1));
    var bad = h;
    bad.flags = 2;
    try std.testing.expectError(error.BadProcessorTable, decodeTable(bad, &ids, 65, 8, 100));
    bad = h;
    bad.a = 64;
    try std.testing.expectError(error.BadProcessorTable, decodeTable(bad, &ids, 65, 8, 100));
    bad = h;
    bad.n = 3;
    try std.testing.expectError(error.BadProcessorTable, decodeTable(bad, &ids, 65, 8, 100));
    const reset = try decodeTable(.{ .kind = h.kind, .flags = 1, .a = 65, .seed = 3 }, &.{}, 65, 8, 100);
    try std.testing.expectEqual(@as(?usize, null), try reset.find(7, 10, 1));
}
