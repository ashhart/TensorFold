//! Typed config.json field reads (no defaults or pins): each refusal names its source and field.
const std = @import("std");
const Value = std.json.Value;
pub const Object = std.json.ObjectMap;

pub const Error = error{BadConfigField};

/// Why the last read refused, e.g. "config.json text_config: hidden_size is missing", in the caller's storage.
pub const Problem = struct {
    bytes: [256]u8 = undefined,
    len: usize = 0,

    pub fn text(p: *const Problem) []const u8 {
        return p.bytes[0..p.len];
    }

    fn refuse(p: *Problem, at: At, comptime why: []const u8, args: anytype) Error {
        var w: std.Io.Writer = .fixed(&p.bytes);
        w.print("{s}: {s} " ++ why, .{ at.source, at.key } ++ args) catch {}; // full: the text is cut, not refused
        p.len = w.end;
        return error.BadConfigField;
    }
};

/// Where a value came from: the document part (`config.json text_config`) and the key.
pub const At = struct { source: []const u8, key: []const u8 };

/// An integer's inclusive bounds, checked before any narrowing.
pub const Ints = struct { min: i64 = 0, max: i64 = std.math.maxInt(i64) };

/// What a present `null` means for an optional key: refused, or the same as an absent key.
pub const Nulls = enum { refused, absent };

/// The JSON kind a refusal names.
fn kind(v: Value) []const u8 {
    return switch (v) {
        .null => "null",
        .bool => "a bool",
        .integer => "an integer",
        .float => "a float",
        .number_string => "a number past 64 bits",
        .string => "a string",
        .array => "an array",
        .object => "an object",
    };
}

/// An integer-tagged value inside `range`, then cast to `T`; a float, string or bool is refused, never truncated.
pub fn int(comptime T: type, v: Value, range: Ints, at: At, p: *Problem) Error!T {
    if (v != .integer) return p.refuse(at, "is {s}, not an integer", .{kind(v)});
    const n = v.integer;
    if (n < range.min or n > range.max) return p.refuse(at, "is {d}, outside {d}..{d}", .{ n, range.min, range.max });
    return std.math.cast(T, n) orelse p.refuse(at, "is {d}, past {s}", .{ n, @typeName(T) });
}

/// An integer or float as f64 (std.json keeps 1e999 a number string, refused here, so the result is finite).
pub fn float(v: Value, at: At, p: *Problem) Error!f64 {
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |f| f,
        else => p.refuse(at, "is {s}, not a number", .{kind(v)}),
    };
}

/// A JSON bool; 0 and 1 are not bools here.
pub fn flag(v: Value, at: At, p: *Problem) Error!bool {
    return if (v == .bool) v.bool else p.refuse(at, "is {s}, not a bool", .{kind(v)});
}

/// One object's fields, read under one source name into one Problem.
pub const Reader = struct {
    object: Object,
    source: []const u8,
    problem: *Problem,

    pub fn at(r: Reader, key: []const u8) At {
        return .{ .source = r.source, .key = key };
    }

    /// The key's value; absent is refused by name.
    pub fn value(r: Reader, key: []const u8) Error!Value {
        return r.object.get(key) orelse r.problem.refuse(r.at(key), "is missing", .{});
    }

    /// The key's value, null when absent (and when `null` with `.absent`); a `null` with `.refused` is refused.
    pub fn optional(r: Reader, key: []const u8, nulls: Nulls) Error!?Value {
        const v = r.object.get(key) orelse return null;
        if (v != .null) return v;
        return if (nulls == .absent) null else r.problem.refuse(r.at(key), "is null", .{});
    }

    pub fn requiredInt(r: Reader, comptime T: type, key: []const u8, range: Ints) Error!T {
        return int(T, try r.value(key), range, r.at(key), r.problem);
    }

    pub fn requiredFloat(r: Reader, key: []const u8) Error!f64 {
        return float(try r.value(key), r.at(key), r.problem);
    }

    pub fn optionalInt(r: Reader, comptime T: type, key: []const u8, nulls: Nulls, range: Ints) Error!?T {
        const v = try r.optional(key, nulls) orelse return null;
        return try int(T, v, range, r.at(key), r.problem);
    }

    pub fn optionalFloat(r: Reader, key: []const u8, nulls: Nulls) Error!?f64 {
        const v = try r.optional(key, nulls) orelse return null;
        return try float(v, r.at(key), r.problem);
    }

    pub fn optionalFlag(r: Reader, key: []const u8, nulls: Nulls) Error!?bool {
        const v = try r.optional(key, nulls) orelse return null;
        return try flag(v, r.at(key), r.problem);
    }
};

test {
    _ = @import("config_fields_test.zig");
}
