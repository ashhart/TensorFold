//! ``json.dumps`` byte for byte: Python's separators, escapes and float repr.
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Writer = std.Io.Writer;

pub const Options = struct {
    ascii: bool = true, // ensure_ascii
    compact: bool = false, // separators=(",", ":")
};

pub fn write(w: *Writer, v: Value, o: Options) Writer.Error!void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .int => |t| try w.writeAll(t),
        .float => |f| {
            var buf: [40]u8 = undefined;
            try w.writeAll(floatJson(&buf, f));
        },
        .string => |s| try writeString(w, s, o),
        .array => |items| {
            try w.writeByte('[');
            for (items, 0..) |item, i| {
                if (i > 0) try w.writeAll(if (o.compact) "," else ", ");
                try write(w, item, o);
            }
            try w.writeByte(']');
        },
        .object => |obj| {
            try w.writeByte('{');
            for (obj.keys(), obj.values(), 0..) |k, item, i| {
                if (i > 0) try w.writeAll(if (o.compact) "," else ", ");
                try writeString(w, k, o);
                try w.writeAll(if (o.compact) ":" else ": ");
                try write(w, item, o);
            }
            try w.writeByte('}');
        },
    }
}

pub fn writeString(w: *Writer, s: []const u8, o: Options) Writer.Error!void {
    try w.writeByte('"');
    var i: usize = 0;
    var plain: usize = 0;
    while (i < s.len) {
        const c = s[i];
        const short: ?[]const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            8 => "\\b",
            12 => "\\f",
            else => null,
        };
        if (short == null and c >= 0x20 and !(o.ascii and c >= 0x7f)) {
            i += 1;
            continue;
        }
        try w.writeAll(s[plain..i]);
        if (short) |e| {
            try w.writeAll(e);
            i += 1;
        } else if (c < 0x80) {
            try w.print("\\u{x:0>4}", .{c});
            i += 1;
        } else {
            const n = std.unicode.utf8ByteSequenceLength(c) catch 1;
            const code = decodeWtf8(s[i..@min(s.len, i + n)]);
            if (code >= 0x10000) {
                const v = code - 0x10000;
                try w.print("\\u{x:0>4}\\u{x:0>4}", .{ 0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff) });
            } else try w.print("\\u{x:0>4}", .{code});
            i += n;
        }
        plain = i;
    }
    try w.writeAll(s[plain..]);
    try w.writeByte('"');
}

fn decodeWtf8(b: []const u8) u21 {
    if (b.len == 2) return (@as(u21, b[0] & 0x1f) << 6) | (b[1] & 0x3f);
    if (b.len == 3) return (@as(u21, b[0] & 0x0f) << 12) | (@as(u21, b[1] & 0x3f) << 6) | (b[2] & 0x3f);
    if (b.len == 4) return (@as(u21, b[0] & 0x07) << 18) | (@as(u21, b[1] & 0x3f) << 12) | (@as(u21, b[2] & 0x3f) << 6) | (b[3] & 0x3f);
    return 0xfffd;
}

/// Python's json float text: repr, with NaN and Infinity spelled as JavaScript does.
pub fn floatJson(buf: []u8, f: f64) []const u8 {
    if (std.math.isNan(f)) return "NaN";
    if (std.math.isInf(f)) return if (f > 0) "Infinity" else "-Infinity";
    return floatRepr(buf, f);
}

/// ``repr(float)``: the shortest round-trip digits, fixed between 1e-4 and 1e16, else exponent form.
pub fn floatRepr(buf: []u8, f: f64) []const u8 {
    if (std.math.isNan(f)) return "nan";
    if (std.math.isInf(f)) return if (f > 0) "inf" else "-inf";
    if (f == 0) return if (std.math.signbit(f)) "-0.0" else "0.0";
    var tmp: [64]u8 = undefined;
    const sci = std.fmt.float.render(&tmp, f, .{ .mode = .scientific }) catch unreachable;
    var digits: [32]u8 = undefined;
    var n: usize = 0;
    var i: usize = 0;
    const negative = sci[0] == '-';
    if (negative) i = 1;
    while (i < sci.len and sci[i] != 'e') : (i += 1) {
        if (sci[i] != '.') {
            digits[n] = sci[i];
            n += 1;
        }
    }
    while (n > 1 and digits[n - 1] == '0') n -= 1;
    const exp = std.fmt.parseInt(i32, sci[i + 1 ..], 10) catch 0;
    const decpt = exp + 1;
    var out: std.ArrayList(u8) = .initBuffer(buf);
    if (negative) out.appendAssumeCapacity('-');
    if (decpt <= -4 or decpt > 16) {
        out.appendAssumeCapacity(digits[0]);
        if (n > 1) {
            out.appendAssumeCapacity('.');
            out.appendSliceAssumeCapacity(digits[1..n]);
        }
        out.appendAssumeCapacity('e');
        out.appendAssumeCapacity(if (exp < 0) '-' else '+');
        const mag: u32 = @intCast(if (exp < 0) -exp else exp);
        if (mag < 10) out.appendAssumeCapacity('0');
        var ebuf: [8]u8 = undefined;
        out.appendSliceAssumeCapacity(std.fmt.bufPrint(&ebuf, "{d}", .{mag}) catch unreachable);
    } else if (decpt <= 0) {
        out.appendSliceAssumeCapacity("0.");
        var z: i32 = decpt;
        while (z < 0) : (z += 1) out.appendAssumeCapacity('0');
        out.appendSliceAssumeCapacity(digits[0..n]);
    } else {
        const d: usize = @intCast(decpt);
        if (d >= n) {
            out.appendSliceAssumeCapacity(digits[0..n]);
            for (0..d - n) |_| out.appendAssumeCapacity('0');
            out.appendSliceAssumeCapacity(".0");
        } else {
            out.appendSliceAssumeCapacity(digits[0..d]);
            out.appendAssumeCapacity('.');
            out.appendSliceAssumeCapacity(digits[d..n]);
        }
    }
    return out.items;
}

test "float repr" {
    var buf: [40]u8 = undefined;
    const cases = [_]struct { f64, []const u8 }{
        .{ 1e16, "1e+16" },                   .{ 1e15, "1000000000000000.0" },                        .{ 0.0001, "0.0001" },
        .{ 0.00001, "1e-05" },                .{ 100.0, "100.0" },                                    .{ 0.1, "0.1" },
        .{ 1.0 / 3.0, "0.3333333333333333" }, .{ 123456789012345678.0, "1.2345678901234568e+17" },    .{ -2.5, "-2.5" },
        .{ 5e-324, "5e-324" },                .{ 1.7976931348623157e308, "1.7976931348623157e+308" }, .{ 42.0, "42.0" },
        .{ std.math.nan(f64), "nan" },        .{ std.math.inf(f64), "inf" },                          .{ -std.math.inf(f64), "-inf" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], floatRepr(&buf, c[0]));
}

test "dumps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try json.parse(a, "{\"a\": [1, 2.50, \"\xc3\xa9\\n\xf0\x9f\x98\x80\x7f\"], \"b\": {}, \"c\": -0}");
    try std.testing.expectEqualStrings("{\"a\": [1, 2.5, \"\\u00e9\\n\\ud83d\\ude00\\u007f\"], \"b\": {}, \"c\": 0}", try json.stringify(a, r.ok, .{}));
    try std.testing.expectEqualStrings("{\"a\":[1,2.5,\"\xc3\xa9\\n\xf0\x9f\x98\x80\x7f\"],\"b\":{},\"c\":0}", try json.stringify(a, r.ok, .{ .ascii = false, .compact = true }));
}
