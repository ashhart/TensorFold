//! JSON values as Python's json module holds them: ordered objects, exact ints, WTF-8 strings.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const parse = @import("json_parse.zig").parse;
pub const parseText = @import("json_parse.zig").parseText;
pub const ParseResult = @import("json_parse.zig").Result;
pub const write = @import("json_write.zig").write;
pub const writeString = @import("json_write.zig").writeString;
pub const Options = @import("json_write.zig").Options;
pub const floatRepr = @import("json_write.zig").floatRepr;

pub const Object = std.array_hash_map.String(Value);

pub const Value = union(enum) {
    null,
    bool: bool,
    int: []const u8, // canonical decimal text: Python ints have no size limit
    float: f64,
    string: []const u8,
    array: []Value,
    object: *Object,

    pub fn get(v: Value, key: []const u8) ?Value {
        return if (v == .object) v.object.get(key) else null;
    }

    /// The field when present and not null (Python's ``body.get(key) is not None``).
    pub fn field(v: Value, key: []const u8) ?Value {
        const found = v.get(key) orelse return null;
        return if (found == .null) null else found;
    }

    pub fn has(v: Value, key: []const u8) bool {
        return v == .object and v.object.contains(key);
    }

    pub fn str(v: Value) ?[]const u8 {
        return if (v == .string) v.string else null;
    }

    pub fn items(v: Value) ?[]Value {
        return if (v == .array) v.array else null;
    }

    /// Python truthiness: None, False, 0, 0.0, "", [] and {} are false.
    pub fn truthy(v: Value) bool {
        return switch (v) {
            .null => false,
            .bool => |b| b,
            .int => |t| !std.mem.eql(u8, t, "0"),
            .float => |f| f != 0,
            .string => |s| s.len != 0,
            .array => |a| a.len != 0,
            .object => |o| o.count() != 0,
        };
    }

    /// A field that is a string, else null.
    pub fn strField(v: Value, key: []const u8) ?[]const u8 {
        const f = v.get(key) orelse return null;
        return if (f == .string) f.string else null;
    }

    /// ``value.get("type") == name``.
    pub fn typeIs(v: Value, name: []const u8) bool {
        const t = v.strField("type") orelse return false;
        return std.mem.eql(u8, t, name);
    }

    /// Python's ``v.get(k1) or v.get(k2) or ...`` when one is truthy.
    pub fn firstTruthy(v: Value, keys: []const []const u8) ?Value {
        for (keys) |k| if (v.get(k)) |f| if (f.truthy()) return f;
        return null;
    }

    /// A number as a double (True is 1, as Python compares them).
    pub fn float64(v: Value) ?f64 {
        return switch (v) {
            .bool => |b| if (b) 1 else 0,
            .int => |t| std.fmt.parseFloat(f64, t) catch null,
            .float => |f| f,
            else => null,
        };
    }

    /// The int as i64 when it is an int (bools excluded) that fits.
    pub fn int64(v: Value) ?i64 {
        if (v != .int) return null;
        return std.fmt.parseInt(i64, v.int, 10) catch null;
    }
};

pub fn truthyField(v: Value, key: []const u8) bool {
    return if (v.get(key)) |f| f.truthy() else false;
}

pub fn newObject(a: Allocator) !*Object {
    const o = try a.create(Object);
    o.* = .empty;
    return o;
}

/// A string value's text, else "".
pub fn strOr(v: ?Value) []const u8 {
    const x = v orelse return "";
    return if (x == .string) x.string else "";
}

pub fn intValue(a: Allocator, n: anytype) !Value {
    return .{ .int = try std.fmt.allocPrint(a, "{d}", .{n}) };
}

/// A shallow copy of an object (Python's ``{**d}``): the same values in a new ordered map.
pub fn copyObject(a: Allocator, o: *const Object) !*Object {
    const out = try a.create(Object);
    out.* = try o.clone(a);
    return out;
}

/// A deep copy into ``a`` (Python's ``copy.deepcopy`` of decoded JSON).
pub fn deepCopy(a: Allocator, v: Value) Allocator.Error!Value {
    return switch (v) {
        .null, .bool, .float => v,
        .int => |t| .{ .int = try a.dupe(u8, t) },
        .string => |s| .{ .string = try a.dupe(u8, s) },
        .array => |items_| blk: {
            const out = try a.alloc(Value, items_.len);
            for (items_, out) |item, *slot| slot.* = try deepCopy(a, item);
            break :blk .{ .array = out };
        },
        .object => |o| blk: {
            const out = try newObject(a);
            try out.ensureTotalCapacity(a, o.count());
            for (o.keys(), o.values()) |k, item| out.putAssumeCapacity(try a.dupe(u8, k), try deepCopy(a, item));
            break :blk .{ .object = out };
        },
    };
}

/// Python's ``==`` on decoded JSON: 1 == 1.0, True == 1, objects ignore order.
pub fn equal(x: Value, y: Value) bool {
    const nx = x.float64();
    const ny = y.float64();
    if (nx != null and ny != null) return nx.? == ny.?;
    return switch (x) {
        .null => y == .null,
        .string => |s| y == .string and std.mem.eql(u8, s, y.string),
        .array => |a| y == .array and a.len == y.array.len and for (a, y.array) |i, j| {
            if (!equal(i, j)) break false;
        } else true,
        .object => |o| y == .object and o.count() == y.object.count() and for (o.keys(), o.values()) |k, i| {
            const j = y.object.get(k) orelse break false;
            if (!equal(i, j)) break false;
        } else true,
        else => false,
    };
}

/// ``json.dumps(v, **options)`` into memory owned by ``a``.
pub fn stringify(a: Allocator, v: Value, options: Options) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    write(&out.writer, v, options) catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

/// A JSON string literal of ``s`` (``json.dumps(s)``).
pub fn quote(a: Allocator, s: []const u8, options: Options) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    writeString(&out.writer, s, options) catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

test {
    _ = @import("json_parse.zig");
    _ = @import("json_write.zig");
}
