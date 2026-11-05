//! Kimi XTML's JSON literals and tool-result ordering, adapted from Moonshot AI's K3.1 release encoding_k3.py.

// Copyright (c) 2026 Moonshot AI; release license: LICENSES/Kimi-K3.1-Pro.txt.
const std = @import("std");
const json = @import("json");
const V = json.Value;
const A = std.mem.Allocator;

pub fn fieldIs(v: V, key: []const u8, want: []const u8) bool {
    return std.mem.eql(u8, v.strField(key) orelse return false, want);
}

pub const Argument = struct { key: []const u8, kind: []const u8, text: []const u8 };
pub const Arguments = struct { items: []const Argument = &.{}, raw: ?[]const u8 = null };

pub fn sorted(a: A, v: V) error{OutOfMemory}!V {
    if (v == .array) {
        const out = try a.alloc(V, v.array.len);
        for (v.array, out) |item, *dest| dest.* = try sorted(a, item);
        return .{ .array = out };
    }
    if (v != .object) return v;
    const keys = try a.dupe([]const u8, v.object.keys());
    std.mem.sort([]const u8, keys, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    const out = try json.newObject(a);
    for (keys) |key| try out.put(a, key, try sorted(a, v.object.get(key).?));
    return .{ .object = out };
}

pub fn kind(v: V) []const u8 {
    return switch (v) {
        .bool => "boolean",
        .null => "null",
        .int, .float => "number",
        .string => "string",
        .object => "object",
        .array => "array",
    };
}

pub fn text(a: A, v: V) ![]const u8 {
    return if (v == .string) v.string else json.stringify(a, v, .{ .ascii = false });
}

fn space(s: []const u8, at: *usize) void {
    while (at.* < s.len and std.mem.indexOfScalar(u8, " \t\r\n", s[at.*]) != null) at.* += 1;
}

/// One raw JSON value's end; its original non-string bytes must survive argument rendering.
fn valueEnd(s: []const u8, start: usize) ?usize {
    var at = start;
    var depth: usize = 0;
    var quoted = false;
    while (at < s.len) : (at += 1) {
        const c = s[at];
        if (quoted) {
            if (c == '\\') {
                at += 1;
                if (at >= s.len) return null;
            } else if (c == '"') {
                quoted = false;
                if (depth == 0) return at + 1;
            }
            continue;
        }
        switch (c) {
            '"' => quoted = true,
            '{', '[' => depth += 1,
            '}', ']' => {
                if (depth == 0) break;
                depth -= 1;
                if (depth == 0) return at + 1;
            },
            ',', ' ', '\t', '\r', '\n' => if (depth == 0) {
                break;
            },
            else => {},
        }
    }
    return if (quoted or depth != 0 or at == start) null else at;
}

/// Python's JSONDecoder(strict=False) permits literal control bytes in strings.
fn decode(a: A, s: []const u8) !?V {
    var escaped: std.ArrayList(u8) = .empty;
    var quoted = false;
    var slash = false;
    for (s) |c| {
        if (quoted and !slash and c < 32) {
            const hex = "0123456789abcdef";
            try escaped.appendSlice(a, &.{ '\\', 'u', '0', '0', hex[c >> 4], hex[c & 15] });
        } else try escaped.append(a, c);
        if (c == '"' and !slash) quoted = !quoted;
        slash = quoted and c == '\\' and !slash;
    }
    return switch (try json.parse(a, escaped.items)) {
        .ok => |v| v,
        .err => null,
    };
}

fn objectArgs(a: A, s: []const u8) !?[]const Argument {
    var at: usize = 0;
    space(s, &at);
    if (at >= s.len or s[at] != '{') return null;
    at += 1;
    space(s, &at);
    var out: std.ArrayList(Argument) = .empty;
    if (at < s.len and s[at] == '}') return out.items;
    while (at < s.len) {
        const key_end = valueEnd(s, at) orelse return null;
        const key = (try decode(a, s[at..key_end])) orelse return null;
        if (key != .string) return null;
        at = key_end;
        space(s, &at);
        if (at >= s.len or s[at] != ':') return null;
        at += 1;
        space(s, &at);
        const end = valueEnd(s, at) orelse return null;
        const v = (try decode(a, s[at..end])) orelse return null;
        try out.append(a, .{ .key = key.string, .kind = kind(v), .text = if (v == .string) v.string else s[at..end] });
        at = end;
        space(s, &at);
        if (at < s.len and s[at] == '}') return out.items;
        if (at >= s.len or s[at] != ',') return null;
        at += 1;
        space(s, &at);
    }
    return null;
}

pub fn arguments(a: A, v: V) !Arguments {
    if (v == .null) return .{};
    if (v == .string) {
        if (v.string.len == 0) return .{};
        return if (try objectArgs(a, v.string)) |items| .{ .items = items } else .{ .raw = v.string };
    }
    if (v != .object) return error.BadKimiArguments;
    const out = try a.alloc(Argument, v.object.count());
    for (v.object.keys(), v.object.values(), out) |key, item, *dest|
        dest.* = .{ .key = key, .kind = kind(item), .text = try text(a, item) };
    return .{ .items = out };
}

pub const Match = struct { position: usize, name: ?[]const u8 };

pub fn matched(calls: []const V, id: V) ?Match {
    if (id != .string) return null;
    for (calls, 0..) |call, i| {
        const key = call.strField("id") orelse continue;
        if (!std.mem.eql(u8, key, id.string)) continue;
        const function = call.get("function") orelse call;
        return .{ .position = i + 1, .name = function.strField("name") };
    }
    return null;
}

/// A fully matched run is ordered by call id; unmatched runs keep their original order and names.
pub fn orderResults(a: A, messages: []const V) ![]const V {
    var out: std.ArrayList(V) = .empty;
    var calls: []const V = &.{};
    var at: usize = 0;
    while (at < messages.len) {
        const m = messages[at];
        if (fieldIs(m, "role", "assistant")) calls = if (m.get("tool_calls")) |v| v.items() orelse &.{} else &.{};
        if (!fieldIs(m, "role", "tool")) {
            try out.append(a, m);
            at += 1;
            continue;
        }
        const start = at;
        var all = true;
        while (at < messages.len and fieldIs(messages[at], "role", "tool")) : (at += 1)
            all = all and matched(calls, messages[at].get("tool_call_id") orelse messages[at].get("id") orelse .null) != null;
        if (!all) {
            try out.appendSlice(a, messages[start..at]);
            continue;
        }
        const run = try a.dupe(V, messages[start..at]);
        const Context = struct {
            calls: []const V,
            fn less(c: @This(), x: V, y: V) bool {
                return matched(c.calls, x.get("tool_call_id") orelse x.get("id").?).?.position <
                    matched(c.calls, y.get("tool_call_id") orelse y.get("id").?).?.position;
            }
        };
        for (1..run.len) |i| {
            const item = run[i];
            var j = i;
            while (j > 0 and Context.less(.{ .calls = calls }, item, run[j - 1])) : (j -= 1) run[j] = run[j - 1];
            run[j] = item;
        }
        for (run) |item| {
            const hit = matched(calls, item.get("tool_call_id") orelse item.get("id").?).?;
            var resolved = item;
            if (hit.name) |name| {
                resolved = .{ .object = try json.copyObject(a, item.object) };
                try resolved.object.put(a, "tool", .{ .string = name });
                if (resolved.has("name")) try resolved.object.put(a, "name", .{ .string = name });
            }
            try out.append(a, resolved);
        }
    }
    return out.items;
}
