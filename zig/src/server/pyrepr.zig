//! Python's ``repr`` of decoded JSON, for refusal messages that quote a request's value.
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub fn repr(a: Allocator, v: ?Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try write(a, &out, v orelse .null);
    return out.items;
}

fn write(a: Allocator, out: *std.ArrayList(u8), v: Value) Allocator.Error!void {
    switch (v) {
        .null => try out.appendSlice(a, "None"),
        .bool => |b| try out.appendSlice(a, if (b) "True" else "False"),
        .int => |t| try out.appendSlice(a, t),
        .float => |f| {
            var buf: [40]u8 = undefined;
            if (std.math.isNan(f)) try out.appendSlice(a, "nan") else if (std.math.isInf(f)) try out.appendSlice(a, if (f > 0) "inf" else "-inf") else try out.appendSlice(a, json.floatRepr(&buf, f));
        },
        .string => |s| try str(a, out, s),
        .array => |items| {
            try out.append(a, '[');
            for (items, 0..) |item, i| {
                if (i > 0) try out.appendSlice(a, ", ");
                try write(a, out, item);
            }
            try out.append(a, ']');
        },
        .object => |o| {
            try out.append(a, '{');
            for (o.keys(), o.values(), 0..) |k, item, i| {
                if (i > 0) try out.appendSlice(a, ", ");
                try str(a, out, k);
                try out.appendSlice(a, ": ");
                try write(a, out, item);
            }
            try out.append(a, '}');
        },
    }
}

/// A str's repr: quotes chosen as Python chooses them, non-printable characters escaped.
pub fn str(a: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    const q: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try out.append(a, q);
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c < 0x80) {
            switch (c) {
                '\\' => try out.appendSlice(a, "\\\\"),
                '\t' => try out.appendSlice(a, "\\t"),
                '\n' => try out.appendSlice(a, "\\n"),
                '\r' => try out.appendSlice(a, "\\r"),
                else => if (c == q) {
                    try out.appendSlice(a, &.{ '\\', c });
                } else if (c < 0x20 or c == 0x7f) {
                    try out.print(a, "\\x{x:0>2}", .{c});
                } else try out.append(a, c),
            }
            i += 1;
            continue;
        }
        const n = std.unicode.utf8ByteSequenceLength(c) catch 1;
        const end = @min(s.len, i + n);
        const cp: u21 = decode(s[i..end]);
        if (printable(cp)) {
            try out.appendSlice(a, s[i..end]);
        } else if (cp <= 0xff) {
            try out.print(a, "\\x{x:0>2}", .{cp});
        } else if (cp <= 0xffff) {
            try out.print(a, "\\u{x:0>4}", .{cp});
        } else try out.print(a, "\\U{x:0>8}", .{cp});
        i = end;
    }
    try out.append(a, q);
}

fn decode(b: []const u8) u21 {
    return switch (b.len) {
        2 => (@as(u21, b[0] & 0x1f) << 6) | (b[1] & 0x3f),
        3 => (@as(u21, b[0] & 0x0f) << 12) | (@as(u21, b[1] & 0x3f) << 6) | (b[2] & 0x3f),
        4 => (@as(u21, b[0] & 0x07) << 18) | (@as(u21, b[1] & 0x3f) << 12) | (@as(u21, b[2] & 0x3f) << 6) | (b[3] & 0x3f),
        else => 0xfffd,
    };
}

/// ``str.isprintable`` for the characters requests carry: controls, format marks, separators and surrogates are not.
fn printable(cp: u21) bool {
    return switch (cp) {
        0x80...0xa0, 0xad, 0x600...0x605, 0x61c, 0x6dd, 0x70f, 0x180e, 0x2000...0x200f, 0x2028...0x202f, 0x205f...0x2064, 0x2066...0x206f, 0x3000, 0xd800...0xdfff, 0xe000...0xf8ff, 0xfeff, 0xfff9...0xfffb, 0xf0000...0x10ffff => false,
        else => true,
    };
}

test "reprs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = (try json.parse(a, "{\"a\": [1, 2.5, null, true, \"it's\"]}")).ok;
    try std.testing.expectEqualStrings("{'a': [1, 2.5, None, True, \"it's\"]}", try repr(a, v));
}
