//! A request's grammar as non-negative int64 words for another rank, so both ranks compile and fill the same masks.

const std = @import("std");
const grammars = @import("grammars.zig");

/// Words `encode` writes for a spec of `text_len` bytes.
pub fn len(text_len: usize) usize {
    return 3 + (text_len + 6) / 7;
}

/// [kind, active, n, bytes 7 a word...] into `out` (len(spec.text.len) words).
pub fn encode(spec: grammars.Spec, active: bool, out: []i64) void {
    out[0] = @backingInt(spec.kind);
    out[1] = @intFromBool(active);
    out[2] = @intCast(spec.text.len);
    var i: usize = 0;
    while (i < spec.text.len) : (i += 7) {
        var v: i64 = 0;
        const chunk = spec.text[i..@min(i + 7, spec.text.len)];
        for (chunk, 0..) |b, k| v |= @as(i64, b) << @intCast(8 * k);
        out[3 + i / 7] = v;
    }
}

pub const Decoded = struct { spec: grammars.Spec, active: bool, used: usize };

/// The spec `encode` wrote at the front of `words`; its text in `buf` (resized).
pub fn decode(gpa: std.mem.Allocator, words: []const i64, buf: *std.ArrayList(u8)) !Decoded {
    if (words.len < 3) return error.BadPlan;
    const kind = std.enums.fromInt(grammars.Kind, words[0]) orelse return error.BadPlan;
    const n: usize = @intCast(words[2]);
    const used = len(n);
    if (words.len < used) return error.BadPlan;
    try buf.resize(gpa, n);
    for (buf.items, 0..) |*b, i| b.* = @truncate(@as(u64, @bitCast(words[3 + i / 7])) >> @intCast(8 * (i % 7)));
    return .{ .spec = .{ .kind = kind, .text = buf.items }, .active = words[1] != 0, .used = used };
}

test "pack round trip" {
    const gpa = std.testing.allocator;
    const text = "{\"type\": \"object\", \"é\": \"｜DSML｜\"}";
    var words: [len(text.len)]i64 = undefined;
    encode(.{ .kind = .json_schema, .text = text }, true, &words);
    for (words) |w| try std.testing.expect(w >= 0);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    const d = try decode(gpa, &words, &buf);
    try std.testing.expectEqual(grammars.Kind.json_schema, d.spec.kind);
    try std.testing.expect(d.active);
    try std.testing.expectEqualStrings(text, d.spec.text);
    try std.testing.expectEqual(words.len, d.used);
}
