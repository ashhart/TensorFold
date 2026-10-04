//! Python's json.dumps as Hugging Face's tojson filter calls it, and request JSON as template values.
const std = @import("std");
const v = @import("value.zig");
const uni = @import("unicode.zig");
const Value = v.Value;

pub const Options = struct {
    ensure_ascii: bool = false,
    indent: ?[]const u8 = null,
    item_sep: []const u8 = ", ",
    key_sep: []const u8 = ": ",
    sort_keys: bool = false,
};

const Out = std.ArrayList(u8);

fn writeString(rt: *v.Rt, out: *Out, s: []const u8, ascii: bool) !void {
    const a = rt.a;
    try out.append(a, '"');
    var i: usize = 0;
    while (i < s.len) {
        const start = i;
        const cp = uni.next(s, &i);
        switch (cp) {
            '"' => try out.appendSlice(a, "\\\""),
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            8 => try out.appendSlice(a, "\\b"),
            12 => try out.appendSlice(a, "\\f"),
            else => if (cp < 0x20 or (ascii and cp == 0x7F)) {
                try out.print(a, "\\u{x:0>4}", .{cp});
            } else if (!ascii or cp < 0x80) {
                try out.appendSlice(a, s[start..i]);
            } else if (cp < 0x10000) {
                try out.print(a, "\\u{x:0>4}", .{cp});
            } else {
                const c = cp - 0x10000;
                try out.print(a, "\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF) });
            },
        }
    }
    try out.append(a, '"');
}

fn newline(rt: *v.Rt, out: *Out, o: Options, level: usize) !void {
    const indent = o.indent orelse return;
    try out.append(rt.a, '\n');
    for (0..level) |_| try out.appendSlice(rt.a, indent);
}

const Entry = struct { key: []const u8, value: Value };

fn byKey(_: void, x: Entry, y: Entry) bool {
    return std.mem.order(u8, x.key, y.key) == .lt;
}

fn write(rt: *v.Rt, out: *Out, value: Value, o: Options, level: usize) v.Error!void {
    const a = rt.a;
    switch (value) {
        .none => try out.appendSlice(a, "null"),
        .boolean => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .int => |i| try out.print(a, "{d}", .{i}),
        .big => |s| try out.appendSlice(a, s),
        .float => |f| try v.writeFloat(out, a, f, true),
        .str => |s| try writeString(rt, out, s.s, o.ensure_ascii),
        .list => |l| {
            if (l.seq != .list and l.seq != .tuple) return rt.fail("Object of type {s} is not JSON serializable", .{v.typeName(value)});
            if (l.items.len == 0) return out.appendSlice(a, "[]");
            try out.append(a, '[');
            for (l.items, 0..) |item, i| {
                if (i > 0) try out.appendSlice(a, o.item_sep);
                try newline(rt, out, o, level + 1);
                try write(rt, out, item, o, level + 1);
            }
            try newline(rt, out, o, level);
            try out.append(a, ']');
        },
        .dict => |d| {
            if (d.map.count() == 0) return out.appendSlice(a, "{}");
            const entries = try a.alloc(Entry, d.map.count());
            for (d.map.keys(), d.map.values(), entries) |k, val, *e| e.* = .{ .key = k, .value = val };
            if (o.sort_keys) std.mem.sort(Entry, entries, {}, byKey);
            try out.append(a, '{');
            for (entries, 0..) |e, i| {
                if (i > 0) try out.appendSlice(a, o.item_sep);
                try newline(rt, out, o, level + 1);
                try writeString(rt, out, e.key, o.ensure_ascii);
                try out.appendSlice(a, o.key_sep);
                try write(rt, out, e.value, o, level + 1);
            }
            try newline(rt, out, o, level);
            try out.append(a, '}');
        },
        else => return rt.fail("Object of type {s} is not JSON serializable", .{v.typeName(value)}),
    }
}

pub fn dumps(rt: *v.Rt, value: Value, o: Options) v.Error![]const u8 {
    var out: Out = .empty;
    try write(rt, &out, value, o, 0);
    return out.items;
}

/// Deeper request JSON is refused rather than recursed into.
pub const max_depth = 1000;

/// Request JSON as Python's json.loads sees it; non-finite numbers become Python infinities.
pub fn fromJson(rt: *v.Rt, j: std.json.Value) v.Error!Value {
    return convert(rt, j, 0);
}

fn convert(rt: *v.Rt, j: std.json.Value, depth: usize) v.Error!Value {
    if (depth > max_depth) return rt.fail("request JSON is nested more than {d} levels deep", .{max_depth});
    return switch (j) {
        .null => .none,
        .bool => |b| .{ .boolean = b },
        .integer => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .number_string => |s| if (std.mem.indexOfAny(u8, s, ".eE") == null) .{ .big = s } else .{ .float = if (s[0] == '-') -std.math.inf(f64) else std.math.inf(f64) },
        .string => |s| Value.string(s),
        .array => |arr| blk: {
            const items = try rt.a.alloc(Value, arr.items.len);
            for (arr.items, items) |x, *y| y.* = try convert(rt, x, depth + 1);
            break :blk try rt.list(items, .list);
        },
        .object => |obj| blk: {
            const d = try rt.a.create(v.Dict);
            d.* = .{ .map = .empty };
            try d.map.ensureTotalCapacity(rt.a, obj.count());
            var it = obj.iterator();
            while (it.next()) |e| d.map.putAssumeCapacity(e.key_ptr.*, try convert(rt, e.value_ptr.*, depth + 1));
            break :blk .{ .dict = d };
        },
    };
}
