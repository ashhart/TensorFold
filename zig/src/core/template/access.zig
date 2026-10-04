//! The sandboxed environment's getattr and getitem, Python slicing, iteration and `in`.
const std = @import("std");
const v = @import("value.zig");
const uni = @import("unicode.zig");
const Value = v.Value;
const Rt = v.Rt;
const Error = v.Error;

/// Every public str method; any of them is a defined attribute even where calling it is unsupported.
pub const str_methods = [_][]const u8{ "capitalize", "casefold", "center", "count", "encode", "endswith", "expandtabs", "find", "format", "format_map", "index", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier", "islower", "isnumeric", "isprintable", "isspace", "istitle", "isupper", "join", "ljust", "lower", "lstrip", "maketrans", "partition", "removeprefix", "removesuffix", "replace", "rfind", "rindex", "rjust", "rpartition", "rsplit", "rstrip", "split", "splitlines", "startswith", "strip", "swapcase", "title", "translate", "upper", "zfill" };
const dict_safe = [_][]const u8{ "copy", "fromkeys", "get", "items", "keys", "values" };
const dict_unsafe = [_][]const u8{ "clear", "pop", "popitem", "setdefault", "update" };
const list_safe = [_][]const u8{ "copy", "count", "index" };
const list_unsafe = [_][]const u8{ "append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort" };
const int_methods = [_][]const u8{ "as_integer_ratio", "bit_count", "bit_length", "conjugate", "from_bytes", "is_integer", "to_bytes" };
const float_methods = [_][]const u8{ "as_integer_ratio", "conjugate", "fromhex", "hex", "is_integer" };

pub fn among(name: []const u8, set: []const []const u8) bool {
    for (set) |s| if (std.mem.eql(u8, s, name)) return true;
    return false;
}

fn method(rt: *Rt, self: Value, name: []const u8) Error!Value {
    const m = try rt.a.create(v.Method);
    m.* = .{ .self = self, .name = name };
    return .{ .method = m };
}

fn unsafe(rt: *Rt, obj: Value, name: []const u8) Error!Value {
    return rt.undef("access to attribute '{s}' of '{s}' object is unsafe.", .{ name, v.typeName(obj) });
}

fn noAttr(rt: *Rt, obj: Value, name: []const u8) Error!Value {
    return rt.undef("'{s} object' has no attribute '{s}'", .{ v.typeName(obj), name });
}

/// Python attributes as the sandbox exposes them: safe methods, then mapping items, else Undefined.
fn attribute(rt: *Rt, obj: Value, name: []const u8) Error!?Value {
    if (name.len > 0 and name[0] == '_') return try unsafe(rt, obj, name);
    switch (obj) {
        .str => if (among(name, &str_methods)) return try method(rt, obj, name),
        .dict => {
            if (among(name, &dict_safe)) return try method(rt, obj, name);
            if (among(name, &dict_unsafe)) return try unsafe(rt, obj, name);
        },
        .list => |l| switch (l.seq) {
            .list => {
                if (among(name, &list_safe)) return try method(rt, obj, name);
                if (among(name, &list_unsafe)) return try unsafe(rt, obj, name);
            },
            .tuple, .range => if (std.mem.eql(u8, name, "count") or std.mem.eql(u8, name, "index")) return try method(rt, obj, name),
            else => {},
        },
        .ns => |n| return n.map.get(name),
        .loop => |l| return try loopAttr(rt, l, name),
        .int, .big, .boolean => {
            const self: Value = if (obj == .boolean) .{ .int = @intFromBool(obj.boolean) } else obj;
            if (std.mem.eql(u8, name, "real") or std.mem.eql(u8, name, "numerator")) return self;
            if (std.mem.eql(u8, name, "imag")) return .{ .int = 0 };
            if (std.mem.eql(u8, name, "denominator")) return .{ .int = 1 };
            if (among(name, &int_methods)) return try method(rt, obj, name);
        },
        .float => {
            if (std.mem.eql(u8, name, "real")) return obj;
            if (std.mem.eql(u8, name, "imag")) return .{ .float = 0 };
            if (among(name, &float_methods)) return try method(rt, obj, name);
        },
        else => {},
    }
    return null;
}

fn loopAttr(rt: *Rt, l: *v.Loop, name: []const u8) Error!?Value {
    const i: i64 = @intCast(l.index0);
    const n: i64 = @intCast(l.items.len);
    const eq = std.mem.eql;
    if (eq(u8, name, "index0")) return .{ .int = i };
    if (eq(u8, name, "index")) return .{ .int = i + 1 };
    if (eq(u8, name, "revindex")) return .{ .int = n - i };
    if (eq(u8, name, "revindex0")) return .{ .int = n - i - 1 };
    if (eq(u8, name, "first")) return .{ .boolean = i == 0 };
    if (eq(u8, name, "last")) return .{ .boolean = i == n - 1 };
    if (eq(u8, name, "length")) return .{ .int = n };
    if (eq(u8, name, "depth")) return .{ .int = 1 };
    if (eq(u8, name, "depth0")) return .{ .int = 0 };
    if (eq(u8, name, "previtem")) return if (i > 0) l.items[l.index0 - 1] else try rt.undef("there is no previous item", .{});
    if (eq(u8, name, "nextitem")) return if (i + 1 < n) l.items[l.index0 + 1] else try rt.undef("there is no next item", .{});
    if (eq(u8, name, "cycle") or eq(u8, name, "changed")) return try method(rt, .{ .loop = l }, name);
    return null;
}

pub fn getattr(rt: *Rt, obj: Value, name: []const u8) Error!Value {
    if (v.isUndef(obj)) return rt.fail("{s}", .{v.undefMsg(obj)});
    if (try attribute(rt, obj, name)) |found| return found;
    if (obj == .dict) if (obj.dict.map.get(name)) |item| return item;
    return noAttr(rt, obj, name);
}

fn index(i: i64, n: usize) ?usize {
    const k = if (i < 0) i + @as(i64, @intCast(n)) else i;
    return if (k >= 0 and k < @as(i64, @intCast(n))) @intCast(k) else null;
}

fn intKey(key: Value) ?i64 {
    return switch (key) {
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        else => null,
    };
}

/// environment.getitem: subscription first, then a string key falls back to attributes.
pub fn getitem(rt: *Rt, obj: Value, key: Value) Error!Value {
    if (v.isUndef(obj)) return rt.fail("{s}", .{v.undefMsg(obj)});
    switch (obj) {
        .dict => |d| if (key == .str) if (d.map.get(key.str.s)) |item| return item,
        .list => |l| if (l.seq == .list or l.seq == .tuple or l.seq == .range) if (intKey(key)) |i| {
            if (index(i, l.items.len)) |k| return l.items[k];
        },
        .str => |s| if (intKey(key)) |i| {
            if (index(i, uni.count(s.s))) |k| {
                const at = uni.offset(s.s, k);
                var end = at;
                _ = uni.next(s.s, &end);
                return Value.string(s.s[at..end]);
            }
        },
        else => {},
    }
    if (key == .str) if (try attribute(rt, obj, key.str.s)) |found| return found;
    if (key == .str) return noAttr(rt, obj, key.str.s);
    return rt.undef("'{s} object' has no element {s}", .{ v.typeName(obj), try v.repr(rt, key) });
}

fn bound(rt: *Rt, x: Value) Error!?i64 {
    return switch (x) {
        .none => null,
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        else => rt.fail("slice indices must be integers or None or have an __index__ method", .{}),
    };
}

/// Python slice.indices: the positions a slice selects from a sequence of length `n`.
pub fn sliceIndices(rt: *Rt, n_: usize, start_: Value, stop_: Value, step_: Value) Error![]usize {
    const n: i64 = @intCast(n_);
    const step = (try bound(rt, step_)) orelse 1;
    if (step == 0) return rt.fail("slice step cannot be zero", .{});
    var start = (try bound(rt, start_)) orelse (if (step < 0) n - 1 else 0);
    var stop = (try bound(rt, stop_)) orelse (if (step < 0) -1 - n else n);
    if (start < 0) {
        start += n;
        if (start < 0) start = if (step < 0) -1 else 0;
    } else if (start >= n) start = if (step < 0) n - 1 else n;
    if (stop < 0) {
        stop += n;
        if (stop < 0) stop = if (step < 0) -1 else 0;
    } else if (stop >= n) stop = if (step < 0) n - 1 else n;
    var out: std.ArrayList(usize) = .empty;
    var i = start;
    while (if (step > 0) i < stop else i > stop) : (i += step) try out.append(rt.a, @intCast(i));
    return out.items;
}

pub fn slice(rt: *Rt, obj: Value, start: Value, stop: Value, step: Value) Error!Value {
    if (v.isUndef(obj)) return rt.fail("{s}", .{v.undefMsg(obj)});
    switch (obj) {
        .str => |s| {
            const cps = try codepoints(rt, s.s);
            var out: std.ArrayList(u8) = .empty;
            for (try sliceIndices(rt, cps.len, start, stop, step)) |k| try out.appendSlice(rt.a, cps[k]);
            return .{ .str = .{ .s = out.items, .safe = s.safe } };
        },
        .list => |l| if (l.seq == .list or l.seq == .tuple or l.seq == .range) {
            const picks = try sliceIndices(rt, l.items.len, start, stop, step);
            const items = try rt.a.alloc(Value, picks.len);
            for (picks, items) |k, *x| x.* = l.items[k];
            return rt.list(items, if (l.seq == .tuple) .tuple else .list);
        },
        else => {},
    }
    return rt.fail("'{s}' object is not subscriptable", .{v.typeName(obj)});
}

pub fn codepoints(rt: *Rt, s: []const u8) Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const start = i;
        _ = uni.next(s, &i);
        try out.append(rt.a, s[start..i]);
    }
    return out.items;
}

/// Python iter() materialized; Undefined iterates empty.
pub fn iterate(rt: *Rt, x: Value) Error![]const Value {
    return switch (x) {
        .missing, .undef => &.{},
        .str => |s| blk: {
            const parts = try codepoints(rt, s.s);
            const items = try rt.a.alloc(Value, parts.len);
            for (parts, items) |p, *it| it.* = .{ .str = .{ .s = p, .safe = s.safe } };
            break :blk items;
        },
        .list => |l| l.items,
        .dict => |d| blk: {
            const items = try rt.a.alloc(Value, d.map.count());
            for (d.map.keys(), items) |k, *it| it.* = Value.string(k);
            break :blk items;
        },
        .iter => |it| if (it.err) |msg| rt.fail("{s}", .{msg}) else it.items,
        else => rt.fail("'{s}' object is not iterable", .{v.typeName(x)}),
    };
}

/// Python `item in container`.
pub fn contains(rt: *Rt, container: Value, item: Value) Error!bool {
    switch (container) {
        .missing, .undef => return false,
        .str => |s| {
            if (item != .str) return rt.fail("'in <string>' requires string as left operand, not {s}", .{v.typeName(item)});
            return std.mem.indexOf(u8, s.s, item.str.s) != null;
        },
        .dict => |d| return switch (item) {
            .str => |k| d.map.contains(k.s),
            .list => |l| if (l.seq == .tuple) false else rt.fail("unhashable type: '{s}'", .{v.typeName(item)}),
            .dict, .ns => rt.fail("unhashable type: '{s}'", .{v.typeName(item)}),
            else => false,
        },
        .list, .iter => {
            for (try iterate(rt, container)) |x| if (v.eql(x, item)) return true;
            return false;
        },
        else => return rt.fail("argument of type '{s}' is not iterable", .{v.typeName(container)}),
    }
}

/// Python iterable unpacking into `n` targets.
pub fn unpack(rt: *Rt, x: Value, n: usize) Error![]const Value {
    const items = try iterate(rt, x);
    if (items.len < n) return rt.fail("not enough values to unpack (expected {d}, got {d})", .{ n, items.len });
    if (items.len > n) return rt.fail("too many values to unpack (expected {d})", .{n});
    return items;
}
