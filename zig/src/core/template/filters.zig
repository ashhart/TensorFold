//! Jinja2's filters and tests that chat templates use, with Hugging Face's tojson.
const std = @import("std");
const v = @import("value.zig");
const uni = @import("unicode.zig");
const access = @import("access.zig");
const methods = @import("methods.zig");
const pyjson = @import("pyjson.zig");
const eval = @import("eval.zig");
const Value = v.Value;
const Error = v.Error;
const Kw = methods.Kw;
const Ctx = eval.Ctx;
const bind = methods.bind;

pub const implemented = [_][]const u8{ "default", "d", "length", "count", "string", "trim", "upper", "lower", "capitalize", "safe", "tojson", "items", "dictsort", "join", "list", "map", "first", "last", "unique", "reverse", "sort", "select", "reject", "selectattr", "rejectattr", "indent", "replace", "escape", "e" };
const jinja_filters = [_][]const u8{ "abs", "attr", "batch", "capitalize", "center", "count", "d", "default", "dictsort", "e", "escape", "filesizeformat", "first", "float", "forceescape", "format", "groupby", "indent", "int", "items", "join", "last", "length", "list", "lower", "map", "max", "min", "pprint", "random", "reject", "rejectattr", "replace", "reverse", "round", "safe", "select", "selectattr", "slice", "sort", "string", "striptags", "sum", "title", "tojson", "trim", "truncate", "unique", "upper", "urlencode", "urlize", "wordcount", "wordwrap", "xmlattr" };
pub const tests = [_][]const u8{ "odd", "even", "divisibleby", "defined", "undefined", "filter", "test", "none", "boolean", "false", "true", "integer", "float", "string", "mapping", "number", "sequence", "iterable", "callable", "sameas", "escaped", "in", "==", "eq", "equalto", "!=", "ne", ">", "gt", "greaterthan", ">=", "ge", "<", "lt", "lessthan", "<=", "le" };
const jinja_tests = tests ++ [_][]const u8{ "lower", "upper" };

/// Python's soft_str: strings (Markup included) pass through, anything else becomes its str().
fn softStr(c: *Ctx, x: Value) Error!v.Str {
    return if (x == .str) x.str else .{ .s = try v.toStr(&c.rt, x) };
}

fn ignoreCase(c: *Ctx, x: Value) Error!Value {
    return if (x == .str) methods.lowerStr(&c.rt, x.str) else x;
}

/// make_attrgetter: dotted paths through environment.getitem, digit parts as integers.
fn attrPath(c: *Ctx, item: Value, attr: Value, default: ?Value) Error!Value {
    var x = item;
    if (attr == .str) {
        var parts = std.mem.splitScalar(u8, attr.str.s, '.');
        while (parts.next()) |p| {
            const all_digits = p.len > 0 and for (p) |ch| {
                if (!std.ascii.isDigit(ch)) break false;
            } else true;
            x = try access.getitem(&c.rt, x, if (all_digits) Value{ .int = std.fmt.parseInt(i64, p, 10) catch return c.rt.fail("index too large", .{}) } else Value.string(p));
            if (default) |d| if (v.isUndef(x)) {
                x = d;
            };
        }
    } else if (attr != .none) {
        x = try access.getitem(&c.rt, x, attr);
        if (default) |d| if (v.isUndef(x)) {
            x = d;
        };
    }
    return x;
}

/// A generator's items computed now; an error is kept until the result is iterated, as Python would raise it.
fn lazy(c: *Ctx, items: []const Value, err: anyerror) Error!Value {
    if (err == error.OutOfMemory) return error.OutOfMemory;
    return c.rt.iter(items, c.rt.msg);
}

const Sorter = struct {
    c: *Ctx,
    keys: []const Value,
    desc: bool,
    failed: bool = false,

    fn less(s: *Sorter, x: usize, y: usize) bool {
        const a = if (s.desc) s.keys[y] else s.keys[x];
        const b = if (s.desc) s.keys[x] else s.keys[y];
        const ord = v.order(&s.c.rt, "<", a, b) catch {
            s.failed = true;
            return false;
        };
        return ord == .lt;
    }
};

fn sorted(c: *Ctx, items: []const Value, keys: []const Value, desc: bool) Error![]Value {
    const idx = try c.rt.a.alloc(usize, items.len);
    for (idx, 0..) |*x, i| x.* = i;
    var s = Sorter{ .c = c, .keys = keys, .desc = desc };
    std.mem.sort(usize, idx, &s, Sorter.less);
    if (s.failed) return error.TemplateError;
    const out = try c.rt.a.alloc(Value, items.len);
    for (idx, out) |i, *x| x.* = items[i];
    return out;
}

fn selectOrReject(c: *Ctx, value: Value, pos: []const Value, kw: []const Kw, attr: bool, keep: bool) Error!Value {
    if (!v.truthy(value)) return c.rt.iter(&.{}, null);
    if (attr and pos.len == 0) return c.rt.fail("Missing parameter for attribute name", .{});
    const off: usize = @intFromBool(attr);
    var out: std.ArrayList(Value) = .empty;
    const items = try access.iterate(&c.rt, value);
    for (items) |item| {
        const x = (if (attr) attrPath(c, item, pos[0], null) else item) catch |err| return lazy(c, out.items, err);
        const hit = (if (pos.len > off) blk: {
            if (pos[off] != .str) break :blk c.rt.fail("test name must be a string", .{});
            break :blk testValue(c, pos[off].str.s, x, pos[off + 1 ..], kw);
        } else v.truthy(x)) catch |err| return lazy(c, out.items, err);
        if (hit == keep) try out.append(c.rt.a, item);
    }
    return c.rt.iter(out.items, null);
}

fn splitlines(c: *Ctx, s: []const u8) Error![]const []const u8 {
    var lines: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    var start: usize = 0;
    while (i < s.len) {
        const at = i;
        const cp = uni.next(s, &i);
        switch (cp) {
            '\n', 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029 => {},
            '\r' => if (i < s.len and s[i] == '\n') {
                i += 1;
            },
            else => continue,
        }
        try lines.append(c.rt.a, s[start..at]);
        start = i;
    }
    if (start < s.len) try lines.append(c.rt.a, s[start..]);
    return lines.items;
}

/// Python's `' ' * n` for an int or bool `n`: empty when negative.
fn spaces(rt: *v.Rt, n: Value) Error![]u8 {
    const p = try rt.a.alloc(u8, @intCast(@max(0, if (n == .int) n.int else @intFromBool(n.boolean))));
    @memset(p, ' ');
    return p;
}

fn indent(c: *Ctx, s: v.Str, width: Value, first: bool, blank: bool) Error!Value {
    const a = c.rt.a;
    const pad = switch (width) {
        .str => |w| w.s,
        .int, .boolean => try spaces(&c.rt, width),
        else => return c.rt.fail("indent width must be an int or str", .{}),
    };
    const lines = try splitlines(c, try std.mem.concat(a, u8, &.{ s.s, "\n" }));
    var out: std.ArrayList(u8) = .empty;
    for (lines, 0..) |line, i| {
        if (i > 0) try out.append(a, '\n');
        if (i > 0 and (blank or line.len > 0)) try out.appendSlice(a, pad);
        try out.appendSlice(a, line);
    }
    if (first) try out.insertSlice(a, 0, pad);
    return .{ .str = .{ .s = out.items, .safe = s.safe } };
}

pub fn filter(c: *Ctx, name: []const u8, value: Value, pos: []const Value, kw: []const Kw) Error!Value {
    const rt = &c.rt;
    const eq = std.mem.eql;
    if (eq(u8, name, "default") or eq(u8, name, "d")) {
        const b = try bind(rt, "default", &.{ "default_value", "boolean" }, 0, pos, kw);
        const boolean = if (b[1]) |x| v.truthy(x) else false;
        return if (v.isUndef(value) or (boolean and !v.truthy(value))) b[0] orelse Value.string("") else value;
    }
    if (eq(u8, name, "length") or eq(u8, name, "count")) {
        _ = try bind(rt, "length", &.{}, 0, pos, kw);
        return .{ .int = @intCast(v.len(value) orelse return rt.fail("object of type '{s}' has no len()", .{v.typeName(value)})) };
    }
    if (eq(u8, name, "string")) return .{ .str = try softStr(c, value) };
    if (eq(u8, name, "trim")) {
        const b = try bind(rt, "trim", &.{"chars"}, 0, pos, kw);
        return methods.strip(rt, try softStr(c, value), b[0]);
    }
    if (eq(u8, name, "upper")) return methods.upperStr(rt, try softStr(c, value));
    if (eq(u8, name, "lower")) return methods.lowerStr(rt, try softStr(c, value));
    if (eq(u8, name, "capitalize")) return methods.capitalizeStr(rt, try softStr(c, value));
    if (eq(u8, name, "safe")) return .{ .str = .{ .s = (try softStr(c, value)).s, .safe = true } };
    if (eq(u8, name, "escape") or eq(u8, name, "e")) return eval.escape(c, value);
    if (eq(u8, name, "tojson")) {
        const b = try bind(rt, "tojson", &.{ "ensure_ascii", "indent", "separators", "sort_keys" }, 0, pos, kw);
        var o = pyjson.Options{ .ensure_ascii = if (b[0]) |x| v.truthy(x) else false, .sort_keys = if (b[3]) |x| v.truthy(x) else false };
        if (b[1]) |ind| switch (ind) {
            .none => {},
            .str => |s| o.indent = s.s,
            .int, .boolean => o.indent = try spaces(rt, ind),
            else => return rt.fail("can't multiply sequence by non-int of type '{s}'", .{v.typeName(ind)}),
        };
        if (o.indent != null) o.item_sep = ",";
        if (b[2]) |sep| if (sep != .none) {
            const pair = try access.unpack(rt, sep, 2);
            if (pair[0] != .str or pair[1] != .str) return rt.fail("separators must be strings", .{});
            o.item_sep = pair[0].str.s;
            o.key_sep = pair[1].str.s;
        };
        return Value.string(try pyjson.dumps(rt, value, o));
    }
    if (eq(u8, name, "items")) {
        if (v.isUndef(value)) return rt.iter(&.{}, null);
        if (value != .dict) return rt.iter(&.{}, "Can only get item pairs from a mapping.");
        const items = try rt.a.alloc(Value, value.dict.map.count());
        for (value.dict.map.keys(), value.dict.map.values(), items) |k, x, *it| it.* = try rt.tuple2(Value.string(k), x);
        return rt.iter(items, null);
    }
    if (eq(u8, name, "dictsort")) {
        const b = try bind(rt, "dictsort", &.{ "case_sensitive", "by", "reverse" }, 0, pos, kw);
        if (value != .dict) return if (v.isUndef(value)) rt.fail("{s}", .{v.undefMsg(value)}) else rt.fail("'{s}' object has no attribute 'items'", .{v.typeName(value)});
        const by: usize = if (b[1]) |x| (if (x == .str and eq(u8, x.str.s, "value")) 1 else if (x == .str and eq(u8, x.str.s, "key")) 0 else return rt.fail("You can only sort by either \"key\" or \"value\"", .{})) else 0;
        const map = &value.dict.map;
        const items = try rt.a.alloc(Value, map.count());
        const keys = try rt.a.alloc(Value, map.count());
        for (map.keys(), map.values(), items, keys) |k, x, *it, *key| {
            it.* = try rt.tuple2(Value.string(k), x);
            const raw = if (by == 0) Value.string(k) else x;
            key.* = if (b[0] != null and v.truthy(b[0].?)) raw else try ignoreCase(c, raw);
        }
        return rt.list(try sorted(c, items, keys, if (b[2]) |r| v.truthy(r) else false), .list);
    }
    if (eq(u8, name, "join")) {
        const b = try bind(rt, "join", &.{ "d", "attribute" }, 0, pos, kw);
        const sep = if (b[0]) |d| try v.toStr(rt, d) else "";
        var out: std.ArrayList(u8) = .empty;
        for (try access.iterate(rt, value), 0..) |item, i| {
            if (i > 0) try out.appendSlice(rt.a, sep);
            const x = if (b[1]) |at| try attrPath(c, item, at, null) else item;
            try out.appendSlice(rt.a, try v.toStr(rt, x));
        }
        return Value.string(out.items);
    }
    if (eq(u8, name, "list")) return rt.list(try rt.a.dupe(Value, try access.iterate(rt, value)), .list);
    if (eq(u8, name, "map")) {
        if (!v.truthy(value)) return rt.iter(&.{}, null);
        var attribute: ?Value = null;
        var default: ?Value = null;
        if (pos.len == 0) for (kw) |k| {
            if (eq(u8, k.name, "attribute")) attribute = k.value else if (eq(u8, k.name, "default")) default = k.value else return rt.fail("Unexpected keyword argument '{s}'", .{k.name});
        };
        if (attribute == null and pos.len == 0) return rt.fail("map requires a filter argument", .{});
        if (attribute == null and pos[0] != .str) return rt.fail("filter name must be a string", .{});
        var out: std.ArrayList(Value) = .empty;
        for (try access.iterate(rt, value)) |item| {
            const x = (if (attribute) |at| attrPath(c, item, at, default) else callFilter(c, pos[0].str.s, item, pos[1..], kw)) catch |err| return lazy(c, out.items, err);
            try out.append(rt.a, x);
        }
        return rt.iter(out.items, null);
    }
    if (eq(u8, name, "first") or eq(u8, name, "last")) {
        _ = try bind(rt, "first", &.{}, 0, pos, kw);
        if (name[0] == 'l' and value == .iter) return rt.fail("'generator' object is not reversible", .{});
        const items = try access.iterate(rt, value);
        if (items.len == 0) return rt.undef("No {s} item, sequence was empty.", .{name});
        return if (name[0] == 'f') items[0] else items[items.len - 1];
    }
    if (eq(u8, name, "unique")) {
        const b = try bind(rt, "unique", &.{ "case_sensitive", "attribute" }, 0, pos, kw);
        var out: std.ArrayList(Value) = .empty;
        var seen: std.ArrayList(Value) = .empty;
        for (try access.iterate(rt, value)) |item| {
            var key = (if (b[1]) |at| attrPath(c, item, at, null) else item) catch |err| return lazy(c, out.items, err);
            if (!(b[0] != null and v.truthy(b[0].?))) key = ignoreCase(c, key) catch |err| return lazy(c, out.items, err);
            if (key == .list and key.list.seq != .tuple or key == .dict or key == .ns) {
                rt.msg = try std.fmt.allocPrint(rt.a, "unhashable type: '{s}'", .{v.typeName(key)});
                return lazy(c, out.items, error.TemplateError);
            }
            if (for (seen.items) |s| {
                if (v.eql(s, key)) break true;
            } else false) continue;
            try seen.append(rt.a, key);
            try out.append(rt.a, item);
        }
        return rt.iter(out.items, null);
    }
    if (eq(u8, name, "reverse")) {
        if (value == .str) {
            const parts = try access.codepoints(rt, value.str.s);
            var out: std.ArrayList(u8) = .empty;
            var i = parts.len;
            while (i > 0) : (i -= 1) try out.appendSlice(rt.a, parts[i - 1]);
            return .{ .str = .{ .s = out.items, .safe = value.str.safe } };
        }
        const items = access.iterate(rt, value) catch return rt.fail("argument must be iterable", .{});
        const out = try rt.a.dupe(Value, items);
        std.mem.reverse(Value, out);
        return if (value == .iter) rt.list(out, .list) else rt.iter(out, null);
    }
    if (eq(u8, name, "sort")) {
        const b = try bind(rt, "sort", &.{ "reverse", "case_sensitive", "attribute" }, 0, pos, kw);
        const items = try access.iterate(rt, value);
        const keys = try rt.a.alloc(Value, items.len);
        for (items, keys) |item, *key| {
            var parts: std.ArrayList(Value) = .empty;
            var attrs: []const Value = &.{.none};
            if (b[2]) |at| if (at == .str) {
                var list: std.ArrayList(Value) = .empty;
                var it = std.mem.splitScalar(u8, at.str.s, ',');
                while (it.next()) |p| try list.append(rt.a, Value.string(p));
                attrs = list.items;
            } else if (at != .none) {
                attrs = &.{at};
            };
            for (attrs) |at| {
                var x = try attrPath(c, item, at, null);
                if (!(b[1] != null and v.truthy(b[1].?))) x = try ignoreCase(c, x);
                try parts.append(rt.a, x);
            }
            key.* = try rt.list(parts.items, .list);
        }
        return rt.list(try sorted(c, items, keys, if (b[0]) |r| v.truthy(r) else false), .list);
    }
    if (eq(u8, name, "select")) return selectOrReject(c, value, pos, kw, false, true);
    if (eq(u8, name, "reject")) return selectOrReject(c, value, pos, kw, false, false);
    if (eq(u8, name, "selectattr")) return selectOrReject(c, value, pos, kw, true, true);
    if (eq(u8, name, "rejectattr")) return selectOrReject(c, value, pos, kw, true, false);
    if (eq(u8, name, "indent")) {
        const b = try bind(rt, "indent", &.{ "width", "first", "blank" }, 0, pos, kw);
        if (value != .str) return rt.fail("unsupported operand type(s) for +=: '{s}' and 'str'", .{v.typeName(value)});
        return indent(c, value.str, b[0] orelse Value{ .int = 4 }, if (b[1]) |x| v.truthy(x) else false, if (b[2]) |x| v.truthy(x) else false);
    }
    if (eq(u8, name, "replace")) {
        const b = try bind(rt, "replace", &.{ "old", "new", "count" }, 2, pos, kw);
        const count: []const Value = if (b[2]) |n| (if (n == .none) &.{} else &.{n}) else &.{};
        const m = v.Method{ .self = Value.string(try v.toStr(rt, value)), .name = "replace" };
        var args: std.ArrayList(Value) = .empty;
        try args.appendSlice(rt.a, &.{ Value.string(try v.toStr(rt, b[0].?)), Value.string(try v.toStr(rt, b[1].?)) });
        try args.appendSlice(rt.a, count);
        return methods.callMethod(rt, &m, args.items, &.{});
    }
    return rt.fail("filter '{s}' is not supported", .{name});
}

fn callFilter(c: *Ctx, name: []const u8, value: Value, pos: []const Value, kw: []const Kw) Error!Value {
    if (!access.among(name, &implemented)) return c.rt.fail("No filter named '{s}'.", .{name});
    return filter(c, name, value, pos, kw);
}

fn modulo(c: *Ctx, x: Value, n: i64) Error!bool {
    if (x == .float) return @mod(x.float, @as(f64, @floatFromInt(n))) == if (n == 2) @as(f64, 1) else 0;
    const i: i64 = switch (x) {
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        .missing, .undef => return c.rt.fail("{s}", .{v.undefMsg(x)}),
        else => return c.rt.fail("unsupported operand type(s) for %: '{s}' and 'int'", .{v.typeName(x)}),
    };
    return @mod(i, n) == if (n == 2) @as(i64, 1) else 0;
}

pub fn testValue(c: *Ctx, name: []const u8, value: Value, pos: []const Value, kw: []const Kw) Error!bool {
    const rt = &c.rt;
    const eq = std.mem.eql;
    const one = struct {
        fn arg(r: *v.Rt, n: []const u8, p: []const Value, k: []const Kw) Error!Value {
            if (p.len + k.len != 1 or k.len != 0) return r.fail("{s}() takes exactly one argument", .{n});
            return p[0];
        }
    };
    if (!(eq(u8, name, "divisibleby") or eq(u8, name, "sameas") or eq(u8, name, "in") or access.among(name, &.{ "==", "eq", "equalto", "!=", "ne", ">", "gt", "greaterthan", ">=", "ge", "<", "lt", "lessthan", "<=", "le" })) and pos.len + kw.len > 0)
        return rt.fail("test '{s}' takes no arguments", .{name});
    if (eq(u8, name, "defined")) return !v.isUndef(value);
    if (eq(u8, name, "undefined")) return v.isUndef(value);
    if (eq(u8, name, "none")) return value == .none;
    if (eq(u8, name, "boolean")) return value == .boolean;
    if (eq(u8, name, "false")) return value == .boolean and !value.boolean;
    if (eq(u8, name, "true")) return value == .boolean and value.boolean;
    if (eq(u8, name, "integer")) return value == .int or value == .big;
    if (eq(u8, name, "float")) return value == .float;
    if (eq(u8, name, "number")) return value == .int or value == .big or value == .float or value == .boolean;
    if (eq(u8, name, "string")) return value == .str;
    if (eq(u8, name, "mapping")) return value == .dict;
    if (eq(u8, name, "escaped")) return value == .str and value.str.safe;
    if (eq(u8, name, "sequence")) return switch (value) {
        .missing, .undef, .str, .dict => true,
        .list => |l| l.seq == .list or l.seq == .tuple or l.seq == .range,
        else => false,
    };
    if (eq(u8, name, "iterable")) return switch (value) {
        .missing, .undef, .str, .list, .dict, .iter, .loop => true,
        else => false,
    };
    if (eq(u8, name, "callable")) return switch (value) {
        .missing, .undef, .macro, .method, .func, .loop => true,
        else => false,
    };
    if (eq(u8, name, "odd")) return modulo(c, value, 2);
    if (eq(u8, name, "even")) return !try modulo(c, value, 2) and (value != .float or @mod(value.float, 2) == 0);
    if (eq(u8, name, "filter")) return value == .str and access.among(value.str.s, &jinja_filters);
    if (eq(u8, name, "test")) return value == .str and access.among(value.str.s, &jinja_tests);
    const other = try one.arg(rt, name, pos, kw);
    if (eq(u8, name, "divisibleby")) {
        const n = switch (other) {
            .int => |i| i,
            .boolean => |b| @as(i64, @intFromBool(b)),
            else => return rt.fail("divisibleby needs an integer", .{}),
        };
        if (n == 0) return rt.fail("integer division or modulo by zero", .{});
        if (value == .float) return @mod(value.float, @as(f64, @floatFromInt(n))) == 0;
        if (value != .int and value != .boolean) return rt.fail("unsupported operand type(s) for %", .{});
        return @mod(if (value == .int) value.int else @intFromBool(value.boolean), n) == 0;
    }
    if (eq(u8, name, "sameas")) return switch (value) {
        .none, .boolean => v.eql(value, other) and @as(std.meta.Tag(Value), value) == @as(std.meta.Tag(Value), other),
        .list, .dict, .ns, .macro, .loop => v.eql(value, other) and std.meta.eql(value, other),
        else => rt.fail("sameas on {s} values is not supported", .{v.typeName(value)}),
    };
    if (eq(u8, name, "in")) return access.contains(rt, other, value);
    const ops = [_]struct { []const u8, @import("ast.zig").CmpOp }{ .{ "==", .eq }, .{ "eq", .eq }, .{ "equalto", .eq }, .{ "!=", .ne }, .{ "ne", .ne }, .{ ">", .gt }, .{ "gt", .gt }, .{ "greaterthan", .gt }, .{ ">=", .gteq }, .{ "ge", .gteq }, .{ "<", .lt }, .{ "lt", .lt }, .{ "lessthan", .lt }, .{ "<=", .lteq }, .{ "le", .lteq } };
    for (ops) |o| if (eq(u8, name, o[0])) return eval.compare(c, o[1], value, other);
    return rt.fail("No test named '{s}'.", .{name});
}
