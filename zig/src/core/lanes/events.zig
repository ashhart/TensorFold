//! The decision log in the Python recorder's format: one JSON line an event, keys sorted, floats as their bits.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Value = union(enum) {
    null,
    bool: bool,
    int: i64,
    uint: u64,
    str: []const u8,
    u32s: []const u32,
    i32s: []const i32,
    i64s: []const i64,
    u64s: []const u64,
    list: []const Value,
    obj: []const Field,
};

pub const Field = struct { k: []const u8, v: Value };

pub fn f(k: []const u8, v: Value) Field {
    return .{ .k = k, .v = v };
}

/// A double as its IEEE bits, the way the recorder writes floats.
pub fn bits(x: f64) Value {
    return .{ .uint = @bitCast(x) };
}

pub const Log = struct {
    arena: std.heap.ArenaAllocator,
    events: std.ArrayList([]Field) = .empty,

    pub fn init(gpa: Allocator) Log {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(log: *Log) void {
        log.arena.deinit();
    }

    /// Record an event (deep-copied); its index lets the loop fill a token it reads later.
    pub fn add(log: *Log, fields: []const Field) !usize {
        const a = log.arena.allocator();
        const copy = try a.alloc(Field, fields.len);
        for (copy, fields) |*c, x| c.* = .{ .k = x.k, .v = try dupe(a, x.v) };
        try log.events.append(a, copy);
        return log.events.items.len - 1;
    }

    pub fn set(log: *Log, event: usize, k: []const u8, v: Value) !void {
        for (log.events.items[event]) |*x| {
            if (std.mem.eql(u8, x.k, k)) {
                x.v = try dupe(log.arena.allocator(), v);
                return;
            }
        }
        return error.NoSuchField;
    }

    /// One event as the recorder's JSON line (no newline).
    pub fn line(log: *Log, gpa: Allocator, event: usize) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(gpa);
        defer out.deinit();
        try writeObj(gpa, &out.writer, log.events.items[event]);
        return out.toOwnedSlice();
    }
};

fn dupe(a: Allocator, v: Value) !Value {
    return switch (v) {
        .str => |s| .{ .str = try a.dupe(u8, s) },
        .u32s => |s| .{ .u32s = try a.dupe(u32, s) },
        .i32s => |s| .{ .i32s = try a.dupe(i32, s) },
        .i64s => |s| .{ .i64s = try a.dupe(i64, s) },
        .u64s => |s| .{ .u64s = try a.dupe(u64, s) },
        .list => |s| blk: {
            const out = try a.alloc(Value, s.len);
            for (out, s) |*o, x| o.* = try dupe(a, x);
            break :blk .{ .list = out };
        },
        .obj => |s| blk: {
            const out = try a.alloc(Field, s.len);
            for (out, s) |*o, x| o.* = .{ .k = x.k, .v = try dupe(a, x.v) };
            break :blk .{ .obj = out };
        },
        else => v,
    };
}

fn lessKey(_: void, a: Field, b: Field) bool {
    return std.mem.lessThan(u8, a.k, b.k);
}

fn writeObj(gpa: Allocator, w: *std.Io.Writer, fields: []const Field) !void {
    const sorted = try gpa.dupe(Field, fields);
    defer gpa.free(sorted);
    std.mem.sort(Field, sorted, {}, lessKey);
    try w.writeByte('{');
    for (sorted, 0..) |x, i| {
        if (i > 0) try w.writeByte(',');
        try writeStr(w, x.k);
        try w.writeByte(':');
        try writeValue(gpa, w, x.v);
    }
    try w.writeByte('}');
}

fn writeStr(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        0...9, 11...31 => try w.print("\\u{x:0>4}", .{c}),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

fn writeInts(w: *std.Io.Writer, comptime T: type, s: []const T) !void {
    try w.writeByte('[');
    for (s, 0..) |x, i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{d}", .{x});
    }
    try w.writeByte(']');
}

fn writeValue(gpa: Allocator, w: *std.Io.Writer, v: Value) anyerror!void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .int => |x| try w.print("{d}", .{x}),
        .uint => |x| try w.print("{d}", .{x}),
        .str => |s| try writeStr(w, s),
        .u32s => |s| try writeInts(w, u32, s),
        .i32s => |s| try writeInts(w, i32, s),
        .i64s => |s| try writeInts(w, i64, s),
        .u64s => |s| try writeInts(w, u64, s),
        .list => |s| {
            try w.writeByte('[');
            for (s, 0..) |x, i| {
                if (i > 0) try w.writeByte(',');
                try writeValue(gpa, w, x);
            }
            try w.writeByte(']');
        },
        .obj => |s| try writeObj(gpa, w, s),
    }
}

test "lines sort keys like json.dumps(sort_keys=True)" {
    const gpa = std.testing.allocator;
    var log = Log.init(gpa);
    defer log.deinit();
    const i = try log.add(&.{ f("stream", .{ .str = "g" }), f("ev", .{ .str = "plan" }), f("n", .{ .int = 2 }), f("tree", .{ .bool = false }), f("got", .{ .u32s = &.{ 1, 2 } }), f("cut", .null) });
    const text = try log.line(gpa, i);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("{\"cut\":null,\"ev\":\"plan\",\"got\":[1,2],\"n\":2,\"stream\":\"g\",\"tree\":false}", text);
}
