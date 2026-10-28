//! The leader names a restored entry by logical identity before any rank mutates its slot.

const std = @import("std");

pub const Want = struct {
    pos: u64,
    key: [16]u8,

    pub fn encode(w: Want, words: []i64) void {
        words[0] = @intCast(w.pos);
        words[1] = std.mem.readInt(i64, w.key[0..8], .little);
        words[2] = std.mem.readInt(i64, w.key[8..16], .little);
    }

    pub fn decode(words: []const i64) !Want {
        if (words.len != 3 or words[0] < 0) return error.BadPlan;
        var w: Want = .{ .pos = @intCast(words[0]), .key = undefined };
        std.mem.writeInt(i64, w.key[0..8], words[1], .little);
        std.mem.writeInt(i64, w.key[8..16], words[2], .little);
        return w;
    }

    pub fn check(w: Want, index: anytype, id: u32) !void {
        if (!index.holds(id, w.pos, w.key)) return error.NotTheEntry;
    }
};

test "restore identity wire preserves high key bits and refuses old or negative positions" {
    const t = std.testing;
    const w: Want = .{ .pos = 80160, .key = @splat(255) };
    var words: [3]i64 = undefined;
    w.encode(&words);
    const got = try Want.decode(&words);
    try t.expectEqual(w.pos, got.pos);
    try t.expectEqualSlices(u8, &w.key, &got.key);
    try t.expectError(error.BadPlan, Want.decode(words[0..2]));
    words[0] = -1;
    try t.expectError(error.BadPlan, Want.decode(&words));
}
