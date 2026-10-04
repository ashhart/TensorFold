//! Template values with the Python semantics Jinja2 inherits: equality, ordering, truth, str() and repr().
const std = @import("std");
const uni = @import("unicode.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ TemplateError, OutOfMemory };

pub const Map = std.array_hash_map.String(Value);

/// How a sequence prints: Python lists, tuples, ranges and dict views share one representation.
pub const Seq = enum { list, tuple, range, items, keys, values };

pub const List = struct { items: []const Value, seq: Seq = .list, range: [3]i64 = .{ 0, 0, 1 } };

pub const Dict = struct { map: Map };

pub const Namespace = struct { map: Map };

/// Jinja2's Undefined: prints empty, iterates empty, fails with `msg` on any other use.
pub const Undef = struct { msg: []const u8 };

/// A generator result: its items, or the error Python would raise once it is iterated.
pub const Iter = struct { items: []const Value, err: ?[]const u8 = null };

pub const Method = struct { self: Value, name: []const u8 };

pub const Func = enum { range, namespace, raise_exception, strftime_now, dict, cycler, joiner, lipsum };

pub const Str = struct { s: []const u8, safe: bool = false };

pub const Loop = struct {
    items: []const Value,
    index0: usize = 0,
    changed: ?[]const Value = null,
};

pub const Macro = struct { node: *const anyopaque, closure: *anyopaque, name: []const u8 };

pub const Value = union(enum) {
    missing,
    undef: *const Undef,
    none,
    boolean: bool,
    int: i64,
    big: []const u8,
    float: f64,
    str: Str,
    list: *const List,
    dict: *const Dict,
    ns: *Namespace,
    macro: *const Macro,
    loop: *Loop,
    method: *const Method,
    func: Func,
    iter: *const Iter,

    pub fn string(s: []const u8) Value {
        return .{ .str = .{ .s = s } };
    }
};

pub const Rt = struct {
    a: Allocator,
    msg: []const u8 = "",
    raised: bool = false,

    pub fn fail(rt: *Rt, comptime fmt: []const u8, args: anytype) Error {
        rt.msg = std.fmt.allocPrint(rt.a, fmt, args) catch return error.OutOfMemory;
        return error.TemplateError;
    }

    pub fn undef(rt: *Rt, comptime fmt: []const u8, args: anytype) Error!Value {
        const u = try rt.a.create(Undef);
        u.* = .{ .msg = try std.fmt.allocPrint(rt.a, fmt, args) };
        return .{ .undef = u };
    }

    pub fn list(rt: *Rt, items: []const Value, seq: Seq) Error!Value {
        const l = try rt.a.create(List);
        l.* = .{ .items = items, .seq = seq };
        return .{ .list = l };
    }

    pub fn iter(rt: *Rt, items: []const Value, err: ?[]const u8) Error!Value {
        const it = try rt.a.create(Iter);
        it.* = .{ .items = items, .err = err };
        return .{ .iter = it };
    }

    pub fn tuple2(rt: *Rt, x: Value, y: Value) Error!Value {
        const items = try rt.a.alloc(Value, 2);
        items[0] = x;
        items[1] = y;
        return rt.list(items, .tuple);
    }

    pub fn print(rt: *Rt, comptime fmt: []const u8, args: anytype) Error!Value {
        return Value.string(try std.fmt.allocPrint(rt.a, fmt, args));
    }
};

pub fn typeName(v: Value) []const u8 {
    return switch (v) {
        .missing, .undef => "Undefined",
        .none => "NoneType",
        .boolean => "bool",
        .int, .big => "int",
        .float => "float",
        .str => |s| if (s.safe) "Markup" else "str",
        .list => |l| switch (l.seq) {
            .list => "list",
            .tuple => "tuple",
            .range => "range",
            .items => "dict_items",
            .keys => "dict_keys",
            .values => "dict_values",
        },
        .dict => "dict",
        .ns => "Namespace",
        .macro => "Macro",
        .loop => "LoopContext",
        .method => "builtin_function_or_method",
        .func => "function",
        .iter => "generator",
    };
}

pub fn isUndef(v: Value) bool {
    return v == .undef or v == .missing;
}

pub fn truthy(v: Value) bool {
    return switch (v) {
        .missing, .undef, .none => false,
        .boolean => |b| b,
        .int => |i| i != 0,
        .big => true,
        .float => |f| f != 0,
        .str => |s| s.s.len > 0,
        .list => |l| l.items.len > 0,
        .dict => |d| d.map.count() > 0,
        .loop => |l| l.items.len > 0,
        .ns, .macro, .method, .func, .iter => true,
    };
}

/// Python len(); null where Python raises TypeError.
pub fn len(v: Value) ?usize {
    return switch (v) {
        .missing, .undef => 0,
        .str => |s| uni.count(s.s),
        .list => |l| l.items.len,
        .dict => |d| d.map.count(),
        .loop => |l| l.items.len,
        else => null,
    };
}

const Num = union(enum) { int: i64, float: f64 };

fn num(v: Value) ?Num {
    return switch (v) {
        .boolean => |b| .{ .int = @intFromBool(b) },
        .int => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        else => null,
    };
}

fn floatEqInt(f: f64, i: i64) bool {
    if (f != @floor(f) or @abs(f) >= 9.223372036854775808e18) return false;
    return @as(i64, @intFromFloat(f)) == i;
}

/// Python ==; Undefined equals only Undefined, and bool compares as an int.
pub fn eql(x: Value, y: Value) bool {
    if (isUndef(x) or isUndef(y)) return isUndef(x) and isUndef(y);
    if (num(x)) |a| if (num(y)) |b| return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| ai == bi,
            .float => |bf| floatEqInt(bf, ai),
        },
        .float => |af| switch (b) {
            .int => |bi| floatEqInt(af, bi),
            .float => |bf| af == bf,
        },
    };
    return switch (x) {
        .none => y == .none,
        .big => |a| y == .big and std.mem.eql(u8, a, y.big),
        .str => |a| y == .str and std.mem.eql(u8, a.s, y.str.s),
        .list => |a| y == .list and seqEql(a, y.list),
        .dict => |a| y == .dict and dictEql(&a.map, &y.dict.map),
        .ns => |a| y == .ns and a == y.ns,
        .macro => |a| y == .macro and a == y.macro,
        .loop => |a| y == .loop and a == y.loop,
        .method => |a| y == .method and a == y.method,
        .func => |a| y == .func and a == y.func,
        .iter => |a| y == .iter and a == y.iter,
        else => false,
    };
}

fn seqEql(a: *const List, b: *const List) bool {
    const kind = struct {
        fn of(s: Seq) u8 {
            return switch (s) {
                .list => 0,
                .tuple => 1,
                .range => 2,
                .items, .keys, .values => 3,
            };
        }
    };
    if (kind.of(a.seq) != kind.of(b.seq) or a.items.len != b.items.len) return false;
    for (a.items, b.items) |p, q| if (!eql(p, q)) return false;
    return true;
}

fn dictEql(a: *const Map, b: *const Map) bool {
    if (a.count() != b.count()) return false;
    var it = a.iterator();
    while (it.next()) |e| {
        const other = b.get(e.key_ptr.*) orelse return false;
        if (!eql(e.value_ptr.*, other)) return false;
    }
    return true;
}

/// Python ordering for <, <=, >, >= and sorting; null means unordered (NaN), mismatched types raise TypeError.
pub fn order(rt: *Rt, op: []const u8, x: Value, y: Value) Error!?std.math.Order {
    if (isUndef(x) or isUndef(y)) return rt.fail("{s}", .{undefMsg(if (isUndef(x)) x else y)});
    if (num(x)) |a| if (num(y)) |b| {
        const af: f64 = switch (a) {
            .int => |i| @floatFromInt(i),
            .float => |f| f,
        };
        const bf: f64 = switch (b) {
            .int => |i| @floatFromInt(i),
            .float => |f| f,
        };
        if (a == .int and b == .int) return std.math.order(a.int, b.int);
        if (std.math.isNan(af) or std.math.isNan(bf)) return null;
        return std.math.order(af, bf);
    };
    if (x == .str and y == .str) return std.mem.order(u8, x.str.s, y.str.s);
    if (x == .list and y == .list and x.list.seq == y.list.seq and (x.list.seq == .list or x.list.seq == .tuple)) {
        const a = x.list.items;
        const b = y.list.items;
        for (a[0..@min(a.len, b.len)], b[0..@min(a.len, b.len)]) |p, q| if (!eql(p, q)) return order(rt, op, p, q);
        return std.math.order(a.len, b.len);
    }
    return rt.fail("'{s}' not supported between instances of '{s}' and '{s}'", .{ op, typeName(x), typeName(y) });
}

pub fn undefMsg(v: Value) []const u8 {
    return switch (v) {
        .undef => |u| u.msg,
        else => "undefined value",
    };
}

/// Python's float repr: shortest round-trip digits, fixed notation for exponents -4 to 15.
pub fn writeFloat(out: *std.ArrayList(u8), a: Allocator, f: f64, json: bool) !void {
    if (std.math.isNan(f)) return out.appendSlice(a, if (json) "NaN" else "nan");
    if (std.math.isInf(f)) return out.appendSlice(a, if (f > 0) (if (json) "Infinity" else "inf") else (if (json) "-Infinity" else "-inf"));
    if (f == 0) return out.appendSlice(a, if (std.math.signbit(f)) "-0.0" else "0.0");
    var buf: [64]u8 = undefined;
    const sci = std.fmt.float.render(&buf, f, .{ .mode = .scientific }) catch unreachable;
    const e = std.mem.indexOfScalar(u8, sci, 'e').?;
    const exp = std.fmt.parseInt(i32, sci[e + 1 ..], 10) catch unreachable;
    var digits: [32]u8 = undefined;
    var nd: usize = 0;
    var neg = false;
    for (sci[0..e]) |c| switch (c) {
        '-' => neg = true,
        '0'...'9' => {
            digits[nd] = c;
            nd += 1;
        },
        else => {},
    };
    const d = digits[0..nd];
    const point = exp + 1;
    if (neg) try out.append(a, '-');
    if (point > -4 and point <= 16) {
        if (point <= 0) {
            try out.appendSlice(a, "0.");
            try out.appendNTimes(a, '0', @intCast(-point));
            try out.appendSlice(a, d);
        } else if (point >= d.len) {
            try out.appendSlice(a, d);
            try out.appendNTimes(a, '0', @as(usize, @intCast(point)) - d.len);
            try out.appendSlice(a, ".0");
        } else {
            try out.appendSlice(a, d[0..@intCast(point)]);
            try out.append(a, '.');
            try out.appendSlice(a, d[@intCast(point)..]);
        }
        return;
    }
    try out.append(a, d[0]);
    if (d.len > 1) {
        try out.append(a, '.');
        try out.appendSlice(a, d[1..]);
    }
    try out.print(a, "e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
}

/// Python str(); values whose str() embeds a memory address are refused.
pub fn toStr(rt: *Rt, v: Value) Error![]const u8 {
    return switch (v) {
        .missing, .undef => "",
        .str => |s| s.s,
        else => {
            var out: std.ArrayList(u8) = .empty;
            try write(rt, &out, v, false);
            return out.items;
        },
    };
}

pub fn repr(rt: *Rt, v: Value) Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try write(rt, &out, v, true);
    return out.items;
}

fn writeQuoted(out: *std.ArrayList(u8), a: Allocator, s: []const u8) !void {
    const quote: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try out.append(a, quote);
    var i: usize = 0;
    while (i < s.len) {
        const start = i;
        const cp = uni.next(s, &i);
        switch (cp) {
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            else => if (cp == quote) {
                try out.appendSlice(a, &.{ '\\', quote });
            } else if (cp < 0x20 or cp == 0x7F) {
                try out.print(a, "\\x{x:0>2}", .{cp});
            } else if (cp < 0x80 or uni.isPrintable(cp)) {
                try out.appendSlice(a, s[start..i]);
            } else if (cp < 0x100) {
                try out.print(a, "\\x{x:0>2}", .{cp});
            } else if (cp < 0x10000) {
                try out.print(a, "\\u{x:0>4}", .{cp});
            } else try out.print(a, "\\U{x:0>8}", .{cp}),
        }
    }
    try out.append(a, quote);
}

fn writeItems(rt: *Rt, out: *std.ArrayList(u8), items: []const Value) Error!void {
    for (items, 0..) |item, i| {
        if (i > 0) try out.appendSlice(rt.a, ", ");
        try write(rt, out, item, true);
    }
}

fn writeMap(rt: *Rt, out: *std.ArrayList(u8), map: *const Map) Error!void {
    try out.append(rt.a, '{');
    var it = map.iterator();
    var first = true;
    while (it.next()) |e| {
        if (!first) try out.appendSlice(rt.a, ", ");
        first = false;
        try writeQuoted(out, rt.a, e.key_ptr.*);
        try out.appendSlice(rt.a, ": ");
        try write(rt, out, e.value_ptr.*, true);
    }
    try out.append(rt.a, '}');
}

fn write(rt: *Rt, out: *std.ArrayList(u8), v: Value, quoted: bool) Error!void {
    const a = rt.a;
    switch (v) {
        .missing, .undef => if (quoted) try out.appendSlice(a, "Undefined"),
        .none => try out.appendSlice(a, "None"),
        .boolean => |b| try out.appendSlice(a, if (b) "True" else "False"),
        .int => |i| try out.print(a, "{d}", .{i}),
        .big => |s| try out.appendSlice(a, s),
        .float => |f| try writeFloat(out, a, f, false),
        .str => |s| if (!quoted) try out.appendSlice(a, s.s) else if (s.safe) {
            try out.appendSlice(a, "Markup(");
            try writeQuoted(out, a, s.s);
            try out.append(a, ')');
        } else try writeQuoted(out, a, s.s),
        .list => |l| switch (l.seq) {
            .list => {
                try out.append(a, '[');
                try writeItems(rt, out, l.items);
                try out.append(a, ']');
            },
            .tuple => {
                try out.append(a, '(');
                try writeItems(rt, out, l.items);
                if (l.items.len == 1) try out.append(a, ',');
                try out.append(a, ')');
            },
            .range => if (l.range[2] == 1) try out.print(a, "range({d}, {d})", .{ l.range[0], l.range[1] }) else try out.print(a, "range({d}, {d}, {d})", .{ l.range[0], l.range[1], l.range[2] }),
            .items, .keys, .values => {
                try out.print(a, "dict_{s}([", .{@tagName(l.seq)});
                try writeItems(rt, out, l.items);
                try out.appendSlice(a, "])");
            },
        },
        .dict => |d| try writeMap(rt, out, &d.map),
        .ns => |n| {
            try out.appendSlice(a, "<Namespace ");
            try writeMap(rt, out, &n.map);
            try out.append(a, '>');
        },
        .macro => |m| {
            try out.appendSlice(a, "<Macro ");
            try writeQuoted(out, a, m.name);
            try out.append(a, '>');
        },
        .loop => |l| try out.print(a, "<LoopContext {d}/{d}>", .{ l.index0 + 1, l.items.len }),
        .method, .func, .iter => return rt.fail("printing a {s} is not supported", .{typeName(v)}),
    }
}

test "python float repr" {
    const a = std.testing.allocator;
    const cases = [_]struct { f64, []const u8 }{
        .{ 1.0, "1.0" },       .{ 100.0, "100.0" },     .{ 1e16, "1e+16" },                                   .{ 1e15, "1000000000000000.0" }, .{ 1.5e-5, "1.5e-05" },
        .{ 0.0001, "0.0001" }, .{ 123.456, "123.456" }, .{ -0.5, "-0.5" },                                    .{ 1e22, "1e+22" },              .{ 5e-324, "5e-324" },
        .{ -0.0, "-0.0" },     .{ 0.1, "0.1" },         .{ 1.2345678901234568e17, "1.2345678901234568e+17" },
    };
    for (cases) |c| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        try writeFloat(&out, a, c[0], false);
        try std.testing.expectEqualStrings(c[1], out.items);
    }
}
