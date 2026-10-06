//! Python methods templates call on strings, dicts and lists, the loop helpers, and the template globals.
const std = @import("std");
const builtin = @import("builtin");
const v = @import("value.zig");
const uni = @import("unicode.zig");
const access = @import("access.zig");
const Value = v.Value;
const Rt = v.Rt;
const Error = v.Error;

pub const Kw = struct { name: []const u8, value: Value };

/// Binds Python parameters by position then keyword; the first `min` are required.
pub fn bind(rt: *Rt, comptime fname: []const u8, comptime names: []const []const u8, min: usize, pos: []const Value, kw: []const Kw) Error![names.len]?Value {
    var out: [names.len]?Value = @splat(null);
    if (pos.len > names.len) return rt.fail(fname ++ "() takes at most {d} arguments ({d} given)", .{ names.len, pos.len });
    if (names.len == 0) {
        if (kw.len > 0) return rt.fail(fname ++ "() takes no keyword arguments", .{});
        return out;
    }
    for (pos, 0..) |p, i| out[i] = p;
    for (kw) |k| {
        const i = for (names, 0..) |n, j| {
            if (std.mem.eql(u8, n, k.name)) break j;
        } else return rt.fail(fname ++ "() got an unexpected keyword argument '{s}'", .{k.name});
        if (out[i] != null) return rt.fail(fname ++ "() got multiple values for argument '{s}'", .{k.name});
        out[i] = k.value;
    }
    for (out[0..min], names[0..min]) |o, n| if (o == null) return rt.fail(fname ++ "() missing required argument '{s}'", .{n});
    return out;
}

fn noneOr(x: ?Value) Value {
    return x orelse .none;
}

fn str(rt: *Rt, x: Value, what: []const u8) Error![]const u8 {
    if (x != .str) return rt.fail("{s} must be str, not {s}", .{ what, v.typeName(x) });
    return x.str.s;
}

fn int(rt: *Rt, x: Value) Error!i64 {
    return switch (x) {
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        else => rt.fail("'{s}' object cannot be interpreted as an integer", .{v.typeName(x)}),
    };
}

/// CPython's ADJUST_INDICES: a [start, end) code point window, possibly empty or inverted.
fn window(rt: *Rt, s: []const u8, start_: ?Value, end_: ?Value) Error![2]i64 {
    const n: i64 = @intCast(uni.count(s));
    var start: i64 = if (start_) |x| (if (x == .none) 0 else try int(rt, x)) else 0;
    var end: i64 = if (end_) |x| (if (x == .none) n else try int(rt, x)) else n;
    if (end > n) end = n else if (end < 0) {
        end += n;
        if (end < 0) end = 0;
    }
    if (start < 0) {
        start += n;
        if (start < 0) start = 0;
    }
    return .{ start, end };
}

/// The window's bytes when it holds at least `need` code points.
fn hay(s: []const u8, w: [2]i64, need: usize) ?struct { []const u8, usize } {
    if (w[1] - w[0] < @as(i64, @intCast(need))) return null;
    const from = uni.offset(s, @intCast(w[0]));
    return .{ s[from..uni.offset(s, @intCast(w[1]))], from };
}

fn isWs(s: []const u8, i: usize) ?usize {
    var j = i;
    return if (uni.isSpace(uni.next(s, &j))) j else null;
}

fn stripChars(rt: *Rt, s: []const u8, chars: ?Value, left: bool, right: bool) Error![]const u8 {
    const set: ?[]const u8 = if (chars) |c| (if (c == .none) null else try str(rt, c, "strip arg")) else null;
    const Hit = struct {
        fn at(text: []const u8, i: usize, cs: ?[]const u8) ?usize {
            var j = i;
            const cp = uni.next(text, &j);
            if (cs) |chars_set| {
                var k: usize = 0;
                while (k < chars_set.len) if (uni.next(chars_set, &k) == cp) return j;
                return null;
            }
            return if (uni.isSpace(cp)) j else null;
        }
    };
    var start: usize = 0;
    var end = s.len;
    if (left) while (start < end) {
        start = Hit.at(s, start, set) orelse break;
    };
    if (right) while (end > start) {
        var b = end - 1;
        while (b > start and s[b] & 0xC0 == 0x80) b -= 1;
        if (Hit.at(s, b, set) == null) break;
        end = b;
    };
    return s[start..end];
}

fn listOf(rt: *Rt, parts: []const []const u8) Error!Value {
    const items = try rt.a.alloc(Value, parts.len);
    for (parts, items) |p, *x| x.* = Value.string(p);
    return rt.list(items, .list);
}

fn split(rt: *Rt, s: []const u8, sep_: ?Value, max_: ?Value, from_right: bool) Error!Value {
    const sep: ?[]const u8 = if (sep_) |x| (if (x == .none) null else try str(rt, x, "separator")) else null;
    var max: i64 = if (max_) |m| try int(rt, m) else -1;
    if (max < 0) max = std.math.maxInt(i64);
    var parts: std.ArrayList([]const u8) = .empty;
    if (sep) |d| {
        if (d.len == 0) return rt.fail("empty separator", .{});
        if (!from_right) {
            var rest = s;
            while (max > 0) : (max -= 1) {
                const at = std.mem.indexOf(u8, rest, d) orelse break;
                try parts.append(rt.a, rest[0..at]);
                rest = rest[at + d.len ..];
            }
            try parts.append(rt.a, rest);
        } else {
            var rest = s;
            while (max > 0) : (max -= 1) {
                const at = std.mem.lastIndexOf(u8, rest, d) orelse break;
                try parts.append(rt.a, rest[at + d.len ..]);
                rest = rest[0..at];
            }
            try parts.append(rt.a, rest);
            std.mem.reverse([]const u8, parts.items);
        }
        return listOf(rt, parts.items);
    }
    if (!from_right) {
        var i: usize = 0;
        while (max > 0) : (max -= 1) {
            while (i < s.len) i = isWs(s, i) orelse break;
            if (i == s.len) break;
            const j = i;
            while (i < s.len and isWs(s, i) == null) _ = uni.next(s, &i);
            try parts.append(rt.a, s[j..i]);
        }
        if (i < s.len) {
            while (i < s.len) i = isWs(s, i) orelse break;
            if (i < s.len) try parts.append(rt.a, s[i..]);
        }
    } else {
        const cps = try access.codepoints(rt, s);
        var i = cps.len;
        const sp = struct {
            fn is(c: []const u8) bool {
                return isWs(c, 0) != null;
            }
        };
        while (max > 0) : (max -= 1) {
            while (i > 0 and sp.is(cps[i - 1])) i -= 1;
            if (i == 0) break;
            const j = i;
            while (i > 0 and !sp.is(cps[i - 1])) i -= 1;
            try parts.append(rt.a, s[uni.offset(s, i)..uni.offset(s, j)]);
        }
        if (i > 0) {
            while (i > 0 and sp.is(cps[i - 1])) i -= 1;
            if (i > 0) try parts.append(rt.a, s[0..uni.offset(s, i)]);
        }
        std.mem.reverse([]const u8, parts.items);
    }
    return listOf(rt, parts.items);
}

fn replace(rt: *Rt, s: []const u8, old: []const u8, new: []const u8, count_: ?Value) Error![]const u8 {
    var count: i64 = if (count_) |c| try int(rt, c) else -1;
    if (count < 0) count = std.math.maxInt(i64);
    var out: std.ArrayList(u8) = .empty;
    if (old.len == 0) {
        var i: usize = 0;
        while (true) {
            if (count > 0) {
                try out.appendSlice(rt.a, new);
                count -= 1;
            }
            if (i >= s.len) break;
            const start = i;
            _ = uni.next(s, &i);
            try out.appendSlice(rt.a, s[start..i]);
            if (count == 0) {
                try out.appendSlice(rt.a, s[i..]);
                break;
            }
        }
        return out.items;
    }
    var rest = s;
    while (count > 0) : (count -= 1) {
        const at = std.mem.indexOf(u8, rest, old) orelse break;
        try out.appendSlice(rt.a, rest[0..at]);
        try out.appendSlice(rt.a, new);
        rest = rest[at + old.len ..];
    }
    try out.appendSlice(rt.a, rest);
    return out.items;
}

fn find(rt: *Rt, s: []const u8, sub: []const u8, start: ?Value, end: ?Value, last: bool) Error!i64 {
    const h = hay(s, try window(rt, s, start, end), uni.count(sub)) orelse return -1;
    const at = (if (last) std.mem.lastIndexOf(u8, h[0], sub) else std.mem.indexOf(u8, h[0], sub)) orelse return -1;
    return @intCast(uni.count(s[0 .. h[1] + at]));
}

fn affix(rt: *Rt, s: []const u8, arg: Value, start: ?Value, end: ?Value, suffix: bool) Error!bool {
    const w = try window(rt, s, start, end);
    const opts: []const Value = if (arg == .list and arg.list.seq == .tuple) arg.list.items else &.{arg};
    for (opts) |o| {
        const p = try str(rt, o, if (suffix) "endswith arg" else "startswith arg");
        const h = hay(s, w, uni.count(p)) orelse continue;
        if (if (suffix) std.mem.endsWith(u8, h[0], p) else std.mem.startsWith(u8, h[0], p)) return true;
    }
    return false;
}

fn caseMap(rt: *Rt, s: v.Str, up: bool) Error!Value {
    const out = try (if (up) uni.upper(rt.a, s.s) else uni.lower(rt.a, s.s));
    return .{ .str = .{ .s = out, .safe = s.safe } };
}

pub fn lowerStr(rt: *Rt, s: v.Str) Error!Value {
    return caseMap(rt, s, false);
}

pub fn upperStr(rt: *Rt, s: v.Str) Error!Value {
    return caseMap(rt, s, true);
}

pub fn capitalizeStr(rt: *Rt, s: v.Str) Error!Value {
    return .{ .str = .{ .s = try uni.capitalize(rt.a, s.s), .safe = s.safe } };
}

pub fn strip(rt: *Rt, s: v.Str, chars: ?Value) Error!Value {
    return .{ .str = .{ .s = try stripChars(rt, s.s, chars, true, true), .safe = s.safe } };
}

fn strMethod(rt: *Rt, s: v.Str, name: []const u8, pos: []const Value, kw: []const Kw) Error!Value {
    const eq = std.mem.eql;
    if (s.safe and !(eq(u8, name, "startswith") or eq(u8, name, "endswith") or eq(u8, name, "find") or eq(u8, name, "rfind") or eq(u8, name, "count")))
        return rt.fail("Markup.{s}() is not supported", .{name});
    const t = s.s;
    if (eq(u8, name, "startswith") or eq(u8, name, "endswith")) {
        const b = try bind(rt, "startswith", &.{ "prefix", "start", "end" }, 1, pos, kw);
        return .{ .boolean = try affix(rt, t, b[0].?, b[1], b[2], name[0] == 'e') };
    }
    if (eq(u8, name, "split") or eq(u8, name, "rsplit")) {
        const b = try bind(rt, "split", &.{ "sep", "maxsplit" }, 0, pos, kw);
        return split(rt, t, b[0], b[1], name[0] == 'r');
    }
    if (eq(u8, name, "strip") or eq(u8, name, "lstrip") or eq(u8, name, "rstrip")) {
        const b = try bind(rt, "strip", &.{"chars"}, 0, pos, kw);
        return Value.string(try stripChars(rt, t, b[0], name[0] != 'r', name[0] != 'l'));
    }
    if (eq(u8, name, "replace")) {
        const b = try bind(rt, "replace", &.{ "old", "new", "count" }, 2, pos, kw);
        return Value.string(try replace(rt, t, try str(rt, b[0].?, "replace arg"), try str(rt, b[1].?, "replace arg"), b[2]));
    }
    if (eq(u8, name, "find") or eq(u8, name, "rfind") or eq(u8, name, "index") or eq(u8, name, "rindex")) {
        const b = try bind(rt, "find", &.{ "sub", "start", "end" }, 1, pos, kw);
        const at = try find(rt, t, try str(rt, b[0].?, "find arg"), b[1], b[2], name[0] == 'r');
        if (at < 0 and (name[0] == 'i' or name[1] == 'i')) return rt.fail("substring not found", .{});
        return .{ .int = at };
    }
    if (eq(u8, name, "count")) {
        const b = try bind(rt, "count", &.{ "sub", "start", "end" }, 1, pos, kw);
        const sub = try str(rt, b[0].?, "count arg");
        const h = hay(t, try window(rt, t, b[1], b[2]), uni.count(sub)) orelse return .{ .int = 0 };
        return .{ .int = @intCast(if (sub.len == 0) uni.count(h[0]) + 1 else std.mem.count(u8, h[0], sub)) };
    }
    if (eq(u8, name, "lower") or eq(u8, name, "upper")) {
        _ = try bind(rt, "lower", &.{}, 0, pos, kw);
        return caseMap(rt, s, name[0] == 'u');
    }
    if (eq(u8, name, "capitalize")) {
        _ = try bind(rt, "capitalize", &.{}, 0, pos, kw);
        return capitalizeStr(rt, s);
    }
    if (eq(u8, name, "removeprefix") or eq(u8, name, "removesuffix")) {
        const b = try bind(rt, "removeprefix", &.{"affix"}, 1, pos, kw);
        const x = try str(rt, b[0].?, "removeprefix arg");
        if (name[6] == 'p') return Value.string(if (std.mem.startsWith(u8, t, x)) t[x.len..] else t);
        return Value.string(if (x.len > 0 and std.mem.endsWith(u8, t, x)) t[0 .. t.len - x.len] else t);
    }
    if (eq(u8, name, "join")) {
        const b = try bind(rt, "join", &.{"iterable"}, 1, pos, kw);
        var out: std.ArrayList(u8) = .empty;
        for (try access.iterate(rt, b[0].?), 0..) |item, i| {
            if (item != .str) return rt.fail("sequence item {d}: expected str instance, {s} found", .{ i, v.typeName(item) });
            if (i > 0) try out.appendSlice(rt.a, t);
            try out.appendSlice(rt.a, item.str.s);
        }
        return Value.string(out.items);
    }
    return rt.fail("str.{s}() is not supported", .{name});
}

fn dictMethod(rt: *Rt, d: *const v.Dict, self: Value, name: []const u8, pos: []const Value, kw: []const Kw) Error!Value {
    const eq = std.mem.eql;
    if (eq(u8, name, "get")) {
        const b = try bind(rt, "get", &.{ "key", "default" }, 1, pos, kw);
        return switch (b[0].?) {
            .str => |k| d.map.get(k.s) orelse noneOr(b[1]),
            .list, .dict, .ns => rt.fail("unhashable type: '{s}'", .{v.typeName(b[0].?)}),
            else => noneOr(b[1]),
        };
    }
    _ = try bind(rt, "items", &.{}, 0, pos, kw);
    if (eq(u8, name, "items")) {
        const items = try rt.a.alloc(Value, d.map.count());
        for (d.map.keys(), d.map.values(), items) |k, x, *it| it.* = try rt.tuple2(Value.string(k), x);
        return rt.list(items, .items);
    }
    if (eq(u8, name, "keys")) return rt.list(try access.iterate(rt, self), .keys);
    if (eq(u8, name, "values")) return rt.list(d.map.values(), .values);
    if (eq(u8, name, "copy")) return self;
    return rt.fail("dict.{s}() is not supported", .{name});
}

fn seqMethod(rt: *Rt, l: *const v.List, self: Value, name: []const u8, pos: []const Value, kw: []const Kw) Error!Value {
    if (std.mem.eql(u8, name, "count")) {
        const b = try bind(rt, "count", &.{"value"}, 1, pos, kw);
        var n: i64 = 0;
        for (l.items) |x| n += @intFromBool(v.eql(x, b[0].?));
        return .{ .int = n };
    }
    if (std.mem.eql(u8, name, "index")) {
        const b = try bind(rt, "index", &.{"value"}, 1, pos, kw);
        for (l.items, 0..) |x, i| if (v.eql(x, b[0].?)) return .{ .int = @intCast(i) };
        return rt.fail("{s} is not in list", .{try v.repr(rt, b[0].?)});
    }
    _ = try bind(rt, "copy", &.{}, 0, pos, kw);
    return self;
}

fn loopMethod(rt: *Rt, l: *v.Loop, name: []const u8, pos: []const Value, kw: []const Kw) Error!Value {
    if (kw.len > 0) return rt.fail("loop.{s}() takes no keyword arguments", .{name});
    if (std.mem.eql(u8, name, "cycle")) {
        if (pos.len == 0) return rt.fail("no items for cycling given", .{});
        return pos[l.index0 % pos.len];
    }
    const now = try rt.a.dupe(Value, pos);
    if (l.changed) |prev| if (prev.len == now.len and for (prev, now) |p, q| {
        if (!v.eql(p, q)) break false;
    } else true) return .{ .boolean = false };
    l.changed = now;
    return .{ .boolean = true };
}

pub fn callMethod(rt: *Rt, m: *const v.Method, pos: []const Value, kw: []const Kw) Error!Value {
    return switch (m.self) {
        .str => |s| strMethod(rt, s, m.name, pos, kw),
        .dict => |d| dictMethod(rt, d, m.self, m.name, pos, kw),
        .list => |l| seqMethod(rt, l, m.self, m.name, pos, kw),
        .loop => |l| loopMethod(rt, l, m.name, pos, kw),
        else => rt.fail("{s}.{s}() is not supported", .{ v.typeName(m.self), m.name }),
    };
}

/// dict(*args, **kwargs) as namespace() and dict() build them.
fn mapping(rt: *Rt, comptime fname: []const u8, pos: []const Value, kw: []const Kw) Error!v.Map {
    var map: v.Map = .empty;
    if (pos.len > 1) return rt.fail(fname ++ " expected at most 1 argument, got {d}", .{pos.len});
    if (pos.len == 1) switch (pos[0]) {
        .dict => |d| {
            var it = d.map.iterator();
            while (it.next()) |e| try map.put(rt.a, e.key_ptr.*, e.value_ptr.*);
        },
        else => for (try access.iterate(rt, pos[0])) |pair| {
            const kv = try access.unpack(rt, pair, 2);
            if (kv[0] != .str) return rt.fail("only string keys are supported", .{});
            try map.put(rt.a, kv[0].str.s, kv[1]);
        },
    };
    for (kw) |k| try map.put(rt.a, k.name, k.value);
    return map;
}

fn range(rt: *Rt, pos: []const Value, kw: []const Kw) Error!Value {
    if (kw.len > 0) return rt.fail("range() takes no keyword arguments", .{});
    if (pos.len == 0 or pos.len > 3) return rt.fail("range expected 1 to 3 arguments, got {d}", .{pos.len});
    var r = [3]i64{ 0, 0, 1 };
    if (pos.len == 1) r[1] = try int(rt, pos[0]) else for (pos, 0..) |p, i| {
        r[i] = try int(rt, p);
    }
    if (r[2] == 0) return rt.fail("range() arg 3 must not be zero", .{});
    const span = if (r[2] > 0) r[1] - r[0] else r[0] - r[1];
    const step = if (r[2] > 0) r[2] else -r[2];
    const n: i64 = if (span <= 0) 0 else @divFloor(span - 1, step) + 1;
    if (n > 100000) return rt.fail("Range too big. The sandbox blocks ranges larger than MAX_RANGE (100000).", .{});
    const items = try rt.a.alloc(Value, @intCast(n));
    for (items, 0..) |*x, i| x.* = .{ .int = r[0] + @as(i64, @intCast(i)) * r[2] };
    const l = try rt.a.create(v.List);
    l.* = .{ .items = items, .seq = .range, .range = r };
    return .{ .list = l };
}

pub fn callFunc(rt: *Rt, f: v.Func, pos: []const Value, kw: []const Kw) Error!Value {
    switch (f) {
        .range => return range(rt, pos, kw),
        .namespace => {
            const ns = try rt.a.create(v.Namespace);
            ns.* = .{ .map = try mapping(rt, "Namespace", pos, kw) };
            return .{ .ns = ns };
        },
        .dict => {
            const d = try rt.a.create(v.Dict);
            d.* = .{ .map = try mapping(rt, "dict", pos, kw) };
            return .{ .dict = d };
        },
        .raise_exception => {
            const b = try bind(rt, "raise_exception", &.{"message"}, 1, pos, kw);
            rt.msg = try v.toStr(rt, b[0].?);
            rt.raised = true;
            return error.TemplateError;
        },
        .strftime_now => {
            const b = try bind(rt, "strftime_now", &.{"format"}, 1, pos, kw);
            return Value.string(try strftimeNow(rt, try str(rt, b[0].?, "format")));
        },
        .cycler, .joiner, .lipsum => return rt.fail("{s}() is not supported", .{@tagName(f)}),
    }
}

const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };
extern "c" fn localtime_r(t: *const std.c.time_t, out: *Tm) ?*Tm;
extern "c" fn strftime(buf: [*]u8, max: usize, fmt: [*:0]const u8, tm: *const Tm) usize;

/// datetime.now().strftime(): %f, %z and %Z are Python's own (naive time), the rest is the C library's.
fn strftimeNow(rt: *Rt, fmt: []const u8) Error![]const u8 {
    if (!builtin.link_libc) return rt.fail("strftime_now needs libc", .{});
    var tv: std.c.timeval = undefined;
    _ = std.c.gettimeofday(&tv, null);
    var tm: Tm = undefined;
    if (localtime_r(&tv.sec, &tm) == null) return rt.fail("localtime failed", .{});
    var pre: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < fmt.len) : (i += 1) {
        if (fmt[i] != '%' or i + 1 >= fmt.len) {
            try pre.append(rt.a, fmt[i]);
            continue;
        }
        i += 1;
        switch (fmt[i]) {
            'f' => try pre.print(rt.a, "{d:0>6}", .{@as(u64, @intCast(tv.usec))}),
            'z', 'Z' => {},
            else => try pre.appendSlice(rt.a, &.{ '%', fmt[i] }),
        }
    }
    try pre.append(rt.a, 0);
    var size: usize = 256;
    while (size <= 1 << 20) : (size *= 2) {
        const buf = try rt.a.alloc(u8, size);
        const n = strftime(buf.ptr, size, @ptrCast(pre.items.ptr), &tm);
        if (n > 0 or pre.items.len == 1) return buf[0..n];
    }
    return "";
}
